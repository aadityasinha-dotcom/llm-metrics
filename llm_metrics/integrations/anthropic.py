"""Trace Anthropic calls without touching the call sites.

    from anthropic import Anthropic
    from llm_metrics.integrations.anthropic import wrap_anthropic

    client = wrap_anthropic(Anthropic())
    client.messages.create(model="claude-sonnet-5", max_tokens=1024, messages=[...])

Every message becomes a ``generation`` observation with the model, the
messages, the reply, the token counts, and the latency, nesting under an
enclosing ``@observe`` trace if there is one. ``Anthropic`` and
``AsyncAnthropic``; ``messages.create`` with and without ``stream=True``; and
the ``messages.stream()`` helper.

Token accounting is normalised to match the OpenAI wrapper
--------------------------------------------------------
Anthropic reports ``input_tokens`` *excluding* whatever was served from the
prompt cache, alongside ``cache_read_input_tokens`` and
``cache_creation_input_tokens``. OpenAI reports ``prompt_tokens`` *including*
cached tokens. A dashboard summing both would silently under-count Anthropic
prompts, so this module records:

* ``prompt_tokens`` — the whole prompt: fresh + cache reads + cache writes
* ``cached_tokens`` — the cache-read subset, as for OpenAI
* ``metadata["usage"]["cache_creation_input_tokens"]`` — the cache-write
  subset, which Anthropic bills at a premium and OpenAI has no equivalent for

Like the OpenAI module this never imports ``anthropic`` — everything is
duck-typed, so the ``[anthropic]`` extra is for the user, not for us.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from llm_metrics.decorator import finish_span, open_span
from llm_metrics.integrations import _transport
from llm_metrics.integrations._stream import StreamState, TracedAsyncStream, TracedStream
from llm_metrics.models import ObservationType

__all__ = ["wrap_anthropic"]

_MARKER = "__llm_metrics_wrapped__"
_NAME = "anthropic.messages"

_TRACKED_PARAMS = (
    "model",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "stream",
    "service_tier",
)

_HEADERS = _transport.HeaderSpec(
    request_id=("request-id", "x-request-id"),
    rate_limit={
        "limit_requests": "anthropic-ratelimit-requests-limit",
        "remaining_requests": "anthropic-ratelimit-requests-remaining",
        "reset_requests": "anthropic-ratelimit-requests-reset",
        "limit_tokens": "anthropic-ratelimit-tokens-limit",
        "remaining_tokens": "anthropic-ratelimit-tokens-remaining",
        "reset_tokens": "anthropic-ratelimit-tokens-reset",
        "limit_input_tokens": "anthropic-ratelimit-input-tokens-limit",
        "remaining_input_tokens": "anthropic-ratelimit-input-tokens-remaining",
        "limit_output_tokens": "anthropic-ratelimit-output-tokens-limit",
        "remaining_output_tokens": "anthropic-ratelimit-output-tokens-remaining",
    },
)


def wrap_anthropic(client: Any) -> Any:
    """Instrument an Anthropic client in place and return it.

    Safe to call twice. Never raises; an unrecognised client comes back
    uninstrumented.
    """
    with contextlib.suppress(Exception):  # rule 2
        _patch_create(client)
    with contextlib.suppress(Exception):  # rule 2
        _patch_stream(client)
    _transport.install(client, _HEADERS)
    return client


def _patch_create(client: Any) -> None:
    messages = client.messages
    original = messages.create
    if getattr(original, _MARKER, False):
        return
    if _transport.is_async_callable(original):
        wrapper: Any = _async_create(original)
    else:
        wrapper = _sync_create(original)
    setattr(wrapper, _MARKER, True)
    messages.create = wrapper


def _patch_stream(client: Any) -> None:
    messages = client.messages
    original = messages.stream
    if getattr(original, _MARKER, False):
        return
    # ``stream`` is a plain function on both clients: it returns a context
    # manager rather than awaiting anything. The manager decides sync/async.
    wrapper = _stream_manager_wrapper(original)
    setattr(wrapper, _MARKER, True)
    messages.stream = wrapper


# --------------------------------------------------------------------------- #
# messages.create
# --------------------------------------------------------------------------- #


def _sync_create(original: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span = _open(kwargs)
        if span is None:
            return original(*args, **kwargs)
        try:
            with _transport.active(span.observation):
                response = original(*args, **kwargs)
        except BaseException as exc:
            finish_span(span, exc=exc)
            raise
        if kwargs.get("stream"):
            return TracedStream(response, _AnthropicStreamState(span))
        _complete(span, response, kwargs)
        return response

    return _copy_identity(wrapper, original)


def _async_create(original: Any) -> Any:
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span = _open(kwargs)
        if span is None:
            return await original(*args, **kwargs)
        try:
            with _transport.active(span.observation):
                response = await original(*args, **kwargs)
        except BaseException as exc:
            finish_span(span, exc=exc)
            raise
        if kwargs.get("stream"):
            return TracedAsyncStream(response, _AnthropicStreamState(span))
        _complete(span, response, kwargs)
        return response

    return _copy_identity(wrapper, original)


# --------------------------------------------------------------------------- #
# messages.stream — a context manager yielding a MessageStream
# --------------------------------------------------------------------------- #


def _stream_manager_wrapper(original: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span = _open(dict(kwargs, stream=True))
        manager = original(*args, **kwargs)
        if span is None:
            return manager
        return _TracedStreamManager(manager, span)

    return _copy_identity(wrapper, original)


class _TracedStreamManager:
    """Wraps ``MessageStreamManager`` / ``AsyncMessageStreamManager``.

    The inner ``MessageStream`` accumulates the message itself and exposes it
    as ``current_message_snapshot``, so the span is settled from that on exit
    rather than by intercepting every event. Time to first token is not
    available on this path; use ``messages.create(stream=True)`` for it.
    """

    def __init__(self, manager: Any, span: Any) -> None:
        self._manager = manager
        self._span = span
        self._stream: Any = None
        self._finished = False

    def __enter__(self) -> Any:
        try:
            with _transport.active(self._span.observation):
                self._stream = self._manager.__enter__()
        except BaseException as exc:
            self._settle(exc)
            raise
        return self._stream

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        self._settle(exc if isinstance(exc, BaseException) else None)
        return self._manager.__exit__(exc_type, exc, tb)

    async def __aenter__(self) -> Any:
        try:
            with _transport.active(self._span.observation):
                self._stream = await self._manager.__aenter__()
        except BaseException as exc:
            self._settle(exc)
            raise
        return self._stream

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        self._settle(exc if isinstance(exc, BaseException) else None)
        return await self._manager.__aexit__(exc_type, exc, tb)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._manager, item)

    def _settle(self, exc: BaseException | None) -> None:
        if self._finished:
            return
        self._finished = True
        if exc is not None:
            finish_span(self._span, exc=exc)
            return
        message = None
        with contextlib.suppress(Exception):  # rule 2
            message = getattr(self._stream, "current_message_snapshot", None)
        _complete(self._span, message, {})


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #


def _copy_identity(wrapper: Any, original: Any) -> Any:
    for attr in ("__name__", "__qualname__", "__doc__", "__module__"):
        with contextlib.suppress(AttributeError, TypeError):
            setattr(wrapper, attr, getattr(original, attr))
    return wrapper


def _open(kwargs: Mapping[str, Any]) -> Any:
    try:
        return open_span(
            _NAME,
            ObservationType.GENERATION,
            input_value=_extract_input(kwargs),
            metadata=_extract_params(kwargs),
        )
    except Exception:  # noqa: BLE001 - rule 2
        return None


def _complete(span: Any, message: Any, kwargs: Mapping[str, Any]) -> None:
    try:
        observation = span.observation
        observation.model = _get(message, "model") or kwargs.get("model")
        _apply_usage(observation, _get(message, "usage"))
        _apply_facts(observation, message)
        output = _extract_output(message)
    except Exception:  # noqa: BLE001 - rule 2
        output = None
    finish_span(span, output=output, summarise_output=False)


class _AnthropicStreamState(StreamState):
    """Reads the raw event stream from ``messages.create(stream=True)``.

    ``message_start`` carries the model, id, and input-side usage;
    ``content_block_start`` names tools; ``content_block_delta`` carries text;
    ``message_delta`` carries the stop reason and output tokens.
    """

    __slots__ = ("facts", "model", "tool_calls", "usage")

    def __init__(self, span: Any) -> None:
        super().__init__(span)
        self.model: str | None = None
        self.usage: dict[str, Any] = {}
        self.facts: dict[str, Any] = {}
        self.tool_calls: list[str] = []

    def _observe(self, event: Any) -> None:
        kind = _get(event, "type")
        if kind == "message_start":
            message = _get(event, "message")
            self.model = _get(message, "model")
            self._merge_usage(_get(message, "usage"))
            response_id = _get(message, "id")
            if response_id:
                self.facts["response_id"] = response_id
        elif kind == "content_block_start":
            block = _get(event, "content_block")
            if _get(block, "type") == "tool_use" and _get(block, "name"):
                self.mark_first_token()
                self.tool_calls.append(str(_get(block, "name")))
        elif kind == "content_block_delta":
            delta = _get(event, "delta")
            text = _get(delta, "text")
            if isinstance(text, str) and text:
                self.mark_first_token()
                self.text.append(text)
            elif _get(delta, "type") in ("thinking_delta", "input_json_delta"):
                self.mark_first_token()
        elif kind == "message_delta":
            reason = _get(_get(event, "delta"), "stop_reason")
            if reason:
                self.facts["stop_reason"] = reason
            self._merge_usage(_get(event, "usage"))

    def _merge_usage(self, usage: Any) -> None:
        if usage is None:
            return
        for key in _USAGE_KEYS:
            value = _as_int(_get(usage, key))
            if value is not None:
                self.usage[key] = value
        self.completion_tokens = self.usage.get("output_tokens")

    def _apply(self, observation: Any) -> None:
        observation.model = self.model
        _apply_usage(observation, self.usage)
        if self.tool_calls:
            self.facts["tool_calls"] = list(self.tool_calls)
        if "stop_reason" in self.facts:
            self.facts["finish_reason"] = _finish_reason(self.facts.pop("stop_reason"))
        observation.metadata.update(self.facts)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

#: Anthropic's stop reasons, mapped onto the vocabulary the OpenAI wrapper
#: records so ``finish_reason`` means the same thing across providers.
_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _finish_reason(stop_reason: Any) -> str:
    return _STOP_REASONS.get(str(stop_reason), str(stop_reason))


def _plain(obj: Any) -> Any:
    for attr in ("model_dump", "to_dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return _prune(method())
            except Exception:  # noqa: BLE001 - rule 2
                continue
    return obj


def _prune(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _prune(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def _extract_input(kwargs: Mapping[str, Any]) -> Any:
    messages = kwargs.get("messages")
    system = kwargs.get("system")
    if system is None:
        return _plain(messages) if messages is not None else None
    # The system prompt is part of what the model saw; keep it with the
    # messages rather than losing it to a metadata field nobody reads.
    return {"system": _plain(system), "messages": _plain(messages)}


def _extract_params(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {key: kwargs[key] for key in _TRACKED_PARAMS if key in kwargs}
    thinking = kwargs.get("thinking")
    if isinstance(thinking, Mapping):
        budget = thinking.get("budget_tokens")
        metadata["thinking"] = thinking.get("type")
        if budget is not None:
            metadata["thinking_budget_tokens"] = budget
    tools = kwargs.get("tools")
    if tools:
        names = [_get(tool, "name") for tool in tools]
        metadata["tools"] = [n for n in names if n]
    return metadata


def _apply_usage(observation: Any, usage: Any) -> None:
    """See the module docstring for why the prompt total is reconstructed."""
    if usage is None:
        return
    fresh = _as_int(_get(usage, "input_tokens"))
    cache_read = _as_int(_get(usage, "cache_read_input_tokens"))
    cache_write = _as_int(_get(usage, "cache_creation_input_tokens"))
    if fresh is not None:
        observation.prompt_tokens = fresh + (cache_read or 0) + (cache_write or 0)
    observation.completion_tokens = _as_int(_get(usage, "output_tokens"))
    if cache_read is not None:
        observation.cached_tokens = cache_read
    if cache_write:
        observation.metadata["usage"] = {"cache_creation_input_tokens": cache_write}


def _apply_facts(observation: Any, message: Any) -> None:
    facts: dict[str, Any] = {}
    response_id = _get(message, "id")
    if response_id:
        facts["response_id"] = response_id
    stop_reason = _get(message, "stop_reason")
    if stop_reason:
        facts["finish_reason"] = _finish_reason(stop_reason)
        if stop_reason == "refusal":
            facts["refusal"] = True
    names = [
        str(_get(block, "name"))
        for block in _get(message, "content") or ()
        if _get(block, "type") == "tool_use" and _get(block, "name")
    ]
    if names:
        facts["tool_calls"] = names
    observation.metadata.update(facts)


def _extract_output(message: Any) -> Any:
    content = _get(message, "content")
    if content is None:
        return _plain(message) if message is not None else None
    blocks = [_plain(block) for block in content]
    if len(blocks) == 1 and _get(blocks[0], "type") == "text":
        return _get(blocks[0], "text")  # the common case reads as plain text
    return blocks
