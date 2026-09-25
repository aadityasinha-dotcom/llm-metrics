"""Trace OpenAI calls without touching the call sites.

    from openai import OpenAI
    from llm_metrics.integrations.openai import wrap_openai

    client = wrap_openai(OpenAI())
    client.chat.completions.create(model="gpt-4o", messages=[...])

Every completion becomes a ``generation`` observation carrying the model, the
messages, the response, the token counts, and the latency. If a trace is
already open — say the caller is inside an ``@observe`` function — the
generation nests under it; otherwise it becomes its own trace.

What is recorded beyond the basics
----------------------------------
* **Token detail.** ``cached_tokens`` and ``reasoning_tokens`` as first-class
  fields, since providers price them differently from the totals they sit
  inside. Audio and prediction token counts go under ``metadata["usage"]``.
* **How it ended.** ``finish_reason`` (a rising ``"length"`` rate means silent
  truncation), whether the model refused, and which tools it asked for.
* **Provider identity.** ``response_id``, ``system_fingerprint`` (changes when
  the backend model is silently rolled) and ``service_tier``.
* **Streams.** Time to first token, output tokens per second, and whether the
  caller drained the stream or abandoned it. See :mod:`._stream`.
* **Headers.** Request id, rate-limit headroom, upstream processing time and
  the number of HTTP attempts the ``openai`` client made. See :mod:`._transport`.

Why this module never imports ``openai``
----------------------------------------
Everything here is duck-typed against the client object. That means the SDK
picks up new API surfaces and new ``openai`` releases without a version pin, a
user who calls :func:`wrap_openai` on something unexpected gets a no-op rather
than an ``ImportError``, and our test suite does not need the real package to
cover the logic. The ``[openai]`` extra exists for the *user's* benefit, not
because this module needs it.

What is deliberately not done
-----------------------------
* **No cost.** Token counts go up; pricing is applied server-side (rule 3).
* **No injected request parameters.** Token usage on a streamed response
  requires ``stream_options={"include_usage": True}``, and setting that behind
  the user's back appends a final chunk with an empty ``choices`` list. Plenty
  of real code does ``chunk.choices[0]`` unguarded and would start raising
  ``IndexError`` the moment it was wrapped. Pass it yourself if you want stream
  token counts; without it the stream is traced with everything except usage.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from llm_metrics.decorator import finish_span, open_span
from llm_metrics.integrations import _transport
from llm_metrics.integrations._stream import StreamState, TracedAsyncStream, TracedStream
from llm_metrics.models import ObservationType

__all__ = ["wrap_openai"]

#: ``(dotted path on the client, observation name)``. Missing paths are skipped,
#: so one table covers every ``openai`` version and both client classes.
_TARGETS: tuple[tuple[str, str], ...] = (
    ("chat.completions.create", "openai.chat.completions"),
    ("responses.create", "openai.responses"),
    ("embeddings.create", "openai.embeddings"),
)

_MARKER = "__llm_metrics_wrapped__"

#: Request parameters worth keeping. The messages go to ``input``; these
#: describe *how* the call was made and belong in metadata.
_TRACKED_PARAMS = (
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "n",
    "stream",
    "seed",
    "response_format",
    "reasoning_effort",
    "service_tier",
)

_HEADERS = _transport.HeaderSpec(
    request_id=("x-request-id",),
    processing_ms=("openai-processing-ms",),
    rate_limit={
        "limit_requests": "x-ratelimit-limit-requests",
        "limit_tokens": "x-ratelimit-limit-tokens",
        "remaining_requests": "x-ratelimit-remaining-requests",
        "remaining_tokens": "x-ratelimit-remaining-tokens",
        "reset_requests": "x-ratelimit-reset-requests",
        "reset_tokens": "x-ratelimit-reset-tokens",
    },
)


def wrap_openai(client: Any) -> Any:
    """Instrument an OpenAI client in place and return it.

    Works on ``OpenAI`` and ``AsyncOpenAI``, and is safe to call twice — the
    second call is a no-op rather than double-wrapping.

    Never raises. A client shape this does not recognise comes back
    uninstrumented rather than broken.
    """
    for path, name in _TARGETS:
        try:
            _patch(client, path, name)
        except Exception:  # noqa: BLE001 - rule 2: an un-patchable path is skipped
            continue
    _transport.install(client, _HEADERS)
    return client


def _patch(client: Any, path: str, name: str) -> None:
    *parents, attribute = path.split(".")
    owner: Any = client
    for step in parents:
        owner = getattr(owner, step)  # raises if this API does not exist here

    original = getattr(owner, attribute)
    if getattr(original, _MARKER, False):
        return  # already wrapped

    if _transport.is_async_callable(original):
        wrapper: Any = _async_wrapper(original, name)
    else:
        wrapper = _sync_wrapper(original, name)

    setattr(wrapper, _MARKER, True)
    setattr(owner, attribute, wrapper)


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #


def _sync_wrapper(original: Any, name: str) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span = _open(name, kwargs)
        if span is None:
            return original(*args, **kwargs)
        try:
            with _transport.active(span.observation):
                response = original(*args, **kwargs)
        except BaseException as exc:
            finish_span(span, exc=exc)
            raise
        if kwargs.get("stream"):
            return TracedStream(response, _OpenAIStreamState(span))
        _complete(span, response, kwargs)
        return response

    return _copy_identity(wrapper, original)


def _async_wrapper(original: Any, name: str) -> Any:
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span = _open(name, kwargs)
        if span is None:
            return await original(*args, **kwargs)
        try:
            with _transport.active(span.observation):
                response = await original(*args, **kwargs)
        except BaseException as exc:
            finish_span(span, exc=exc)
            raise
        if kwargs.get("stream"):
            return TracedAsyncStream(response, _OpenAIStreamState(span))
        _complete(span, response, kwargs)
        return response

    return _copy_identity(wrapper, original)


def _copy_identity(wrapper: Any, original: Any) -> Any:
    """``functools.wraps`` without inheriting ``__wrapped__``.

    Keeping ``__wrapped__`` would make ``inspect.unwrap`` on our wrapper reach
    the original coroutine function, so a second :func:`wrap_openai` on an
    already-wrapped sync client could pick the async path.
    """
    for attr in ("__name__", "__qualname__", "__doc__", "__module__"):
        with contextlib.suppress(AttributeError, TypeError):
            setattr(wrapper, attr, getattr(original, attr))
    return wrapper


def _open(name: str, kwargs: Mapping[str, Any]) -> Any:
    """Open the generation span. Returns ``None`` to mean "do not instrument"."""
    try:
        return open_span(
            name,
            ObservationType.GENERATION,
            input_value=_extract_input(kwargs),
            metadata=_extract_params(kwargs),
        )
    except Exception:  # noqa: BLE001 - rule 2
        return None


def _complete(span: Any, response: Any, kwargs: Mapping[str, Any]) -> None:
    """Record model, tokens, and output from a non-streamed response."""
    try:
        observation = span.observation
        observation.model = _get(response, "model") or kwargs.get("model")
        _apply_usage(observation, _get(response, "usage"))
        _apply_response_facts(observation, response)
        output = _extract_output(response)
    except Exception:  # noqa: BLE001 - rule 2: still close the span below
        output = None
    # summarise_output=False: the shape was already reduced to something small
    # and meaningful; the generic summariser would only flatten it further.
    finish_span(span, output=output, summarise_output=False)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class _OpenAIStreamState(StreamState):
    """Reads chat-completion chunks and Responses API events."""

    __slots__ = ("facts", "model", "tool_calls", "usage")

    def __init__(self, span: Any) -> None:
        super().__init__(span)
        self.model: str | None = None
        self.usage: Any = None
        self.facts: dict[str, Any] = {}
        #: ``index -> name``. The name arrives on the first delta for an
        #: index; later deltas for it carry only argument fragments.
        self.tool_calls: dict[int, str] = {}

    def _observe(self, chunk: Any) -> None:
        self.model = self.model or _get(chunk, "model")

        usage = _get(chunk, "usage")
        if usage is not None:
            self.usage = usage
            self.completion_tokens = _completion_tokens(usage)
        _collect_identity(self.facts, chunk)

        for choice in _get(chunk, "choices") or ():
            reason = _get(choice, "finish_reason")
            if reason:
                self.facts["finish_reason"] = reason
            delta = _get(choice, "delta")
            if delta is None:
                continue
            piece = _get(delta, "content")
            if isinstance(piece, str) and piece:
                self.mark_first_token()
                self.text.append(piece)
            if _get(delta, "refusal"):
                self.mark_first_token()
                self.facts["refusal"] = True
            for call in _get(delta, "tool_calls") or ():
                self.mark_first_token()
                index = _get(call, "index")
                name = _get(_get(call, "function"), "name")
                if isinstance(index, int) and name:
                    self.tool_calls.setdefault(index, name)

        # Responses API: text arrives as events with a string ``delta``, and
        # the final ``response.completed`` event carries the whole response.
        piece = _get(chunk, "delta")
        if isinstance(piece, str) and piece:
            self.mark_first_token()
            self.text.append(piece)
        final = _get(chunk, "response")
        if final is not None and _get(chunk, "type") == "response.completed":
            self.usage = _get(final, "usage") or self.usage
            self.completion_tokens = _completion_tokens(self.usage)
            self.model = _get(final, "model") or self.model
            _collect_response_api_facts(self.facts, final)

    def _apply(self, observation: Any) -> None:
        observation.model = self.model
        _apply_usage(observation, self.usage)
        if self.tool_calls:
            self.facts["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        observation.metadata.update(self.facts)


# --------------------------------------------------------------------------- #
# Extraction — tolerant of both attribute objects and plain dicts
# --------------------------------------------------------------------------- #


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _plain(obj: Any) -> Any:
    """Reduce a provider model object to plain data if it knows how.

    The dump is pruned of nulls. A chat message dumps with ``refusal``,
    ``audio``, ``function_call``, ``tool_calls`` and friends all set to
    ``None``; carrying that to the server costs payload on every generation
    and buys a wall of nulls on the dashboard. Only *dumped* values are pruned
    — arguments the caller passed are never rewritten.
    """
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
    """The prompt, under whichever name this endpoint uses for it."""
    for key in ("messages", "input", "prompt"):
        if key in kwargs:
            return _plain(kwargs[key])
    return None


def _extract_params(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {key: kwargs[key] for key in _TRACKED_PARAMS if key in kwargs}
    tools = kwargs.get("tools")
    if tools:
        # Tool schemas are large and static. The names are the useful part.
        with_names = [_get(_get(tool, "function"), "name") or _get(tool, "name") for tool in tools]
        metadata["tools"] = [n for n in with_names if n]
    return metadata


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _first_int(obj: Any, *names: str) -> int | None:
    for name in names:
        value = _as_int(_get(obj, name))
        if value is not None:
            return value
    return None


def _completion_tokens(usage: Any) -> int | None:
    return _first_int(usage, "completion_tokens", "output_tokens")


def _apply_usage(observation: Any, usage: Any) -> None:
    """Chat completions say prompt/completion; the Responses API says input/output.

    The detail objects follow the same split. Cached and reasoning tokens
    become first-class fields; the rarer counts stay in ``metadata["usage"]``
    so the wire shape does not grow a column per provider feature.
    """
    if usage is None:
        return
    observation.prompt_tokens = _first_int(usage, "prompt_tokens", "input_tokens")
    observation.completion_tokens = _completion_tokens(usage)

    prompt_details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
    output_details = _get(usage, "completion_tokens_details") or _get(
        usage, "output_tokens_details"
    )
    observation.cached_tokens = _first_int(prompt_details, "cached_tokens")
    observation.reasoning_tokens = _first_int(output_details, "reasoning_tokens")

    extra: dict[str, int] = {}
    for key, source, name in (
        ("audio_input_tokens", prompt_details, "audio_tokens"),
        ("audio_output_tokens", output_details, "audio_tokens"),
        ("accepted_prediction_tokens", output_details, "accepted_prediction_tokens"),
        ("rejected_prediction_tokens", output_details, "rejected_prediction_tokens"),
    ):
        value = _as_int(_get(source, name))
        if value:
            extra[key] = value
    if extra:
        observation.metadata["usage"] = extra


def _collect_identity(facts: dict[str, Any], response: Any) -> None:
    """Fields that identify *this* response on the provider's side."""
    for key in ("system_fingerprint", "service_tier"):
        value = _get(response, key)
        if value:
            facts[key] = value
    response_id = _get(response, "id")
    if isinstance(response_id, str) and response_id and "response_id" not in facts:
        facts["response_id"] = response_id


def _apply_response_facts(observation: Any, response: Any) -> None:
    facts: dict[str, Any] = {}
    _collect_identity(facts, response)

    choices = _get(response, "choices")
    if choices:
        reasons = [_get(choice, "finish_reason") for choice in choices]
        reasons = [r for r in reasons if r]
        if reasons:
            facts["finish_reason"] = reasons[0] if len(set(reasons)) == 1 else reasons
        names: list[str] = []
        for choice in choices:
            message = _get(choice, "message")
            if _get(message, "refusal"):
                facts["refusal"] = True
            for call in _get(message, "tool_calls") or ():
                name = _get(_get(call, "function"), "name")
                if name:
                    names.append(name)
        if names:
            facts["tool_calls"] = names
    else:
        _collect_response_api_facts(facts, response)

    observation.metadata.update(facts)


def _collect_response_api_facts(facts: dict[str, Any], response: Any) -> None:
    """The Responses API reports completion state on the response itself."""
    status = _get(response, "status")
    if status == "incomplete":
        facts["finish_reason"] = _get(_get(response, "incomplete_details"), "reason") or status
    elif status:
        facts["finish_reason"] = "stop" if status == "completed" else status

    names: list[str] = []
    for item in _get(response, "output") or ():
        kind = _get(item, "type")
        if kind == "function_call" and _get(item, "name"):
            names.append(str(_get(item, "name")))
        elif kind == "message":
            for part in _get(item, "content") or ():
                if _get(part, "type") == "refusal":
                    facts["refusal"] = True
    if names:
        facts["tool_calls"] = names


def _extract_output(response: Any) -> Any:
    """The part of the response worth showing on a trace.

    Whole responses carry request echoes, ids, and system fingerprints that add
    bulk without adding meaning, so this reaches for the message instead.
    """
    choices = _get(response, "choices")
    if choices:
        messages = [_plain(_get(choice, "message")) for choice in choices]
        messages = [m for m in messages if m is not None]
        if messages:
            return messages[0] if len(messages) == 1 else messages

    text = _get(response, "output_text")
    if isinstance(text, str) and text:
        return text

    output = _get(response, "output")
    if output is not None:
        return _plain(output)

    data = _get(response, "data")
    if data is not None:
        # Embeddings: the vectors are useless on a dashboard and enormous.
        return f"<{len(data)} embedding(s)>"

    return _plain(response)
