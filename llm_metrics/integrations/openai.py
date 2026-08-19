"""Trace OpenAI calls without touching the call sites.

    from openai import OpenAI
    from llm_metrics.integrations.openai import wrap_openai

    client = wrap_openai(OpenAI())
    client.chat.completions.create(model="gpt-4o", messages=[...])

Every completion becomes a ``generation`` observation carrying the model, the
messages, the response, the token counts, and the latency. If a trace is
already open — say the caller is inside an ``@observe`` function — the
generation nests under it; otherwise it becomes its own trace.

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
import inspect
from collections.abc import Mapping
from typing import Any

from llm_metrics.decorator import finish_span, open_span
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
    "n",
    "stream",
    "seed",
    "response_format",
    "reasoning_effort",
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
    return client


def _patch(client: Any, path: str, name: str) -> None:
    *parents, attribute = path.split(".")
    owner: Any = client
    for step in parents:
        owner = getattr(owner, step)  # raises if this API does not exist here

    original = getattr(owner, attribute)
    if getattr(original, _MARKER, False):
        return  # already wrapped

    # openai decorates `create` with functools.wraps, so the bound method is
    # not itself a coroutine function even on AsyncOpenAI. Unwrap before asking.
    if inspect.iscoroutinefunction(inspect.unwrap(original)):
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
            response = original(*args, **kwargs)
        except BaseException as exc:
            finish_span(span, exc=exc)
            raise
        if kwargs.get("stream"):
            return _TracedStream(response, span)
        _complete(span, response, kwargs)
        return response

    return _copy_identity(wrapper, original)


def _async_wrapper(original: Any, name: str) -> Any:
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span = _open(name, kwargs)
        if span is None:
            return await original(*args, **kwargs)
        try:
            response = await original(*args, **kwargs)
        except BaseException as exc:
            finish_span(span, exc=exc)
            raise
        if kwargs.get("stream"):
            return _TracedAsyncStream(response, span)
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
        prompt, completion = _extract_usage(response)
        observation.prompt_tokens = prompt
        observation.completion_tokens = completion
        output = _extract_output(response)
    except Exception:  # noqa: BLE001 - rule 2: still close the span below
        output = None
    # summarise_output=False: the shape was already reduced to something small
    # and meaningful; the generic summariser would only flatten it further.
    finish_span(span, output=output, summarise_output=False)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class _StreamState:
    """Accumulates a streamed completion into one observation's worth of data."""

    __slots__ = ("chunks", "completion_tokens", "model", "prompt_tokens", "text")

    def __init__(self) -> None:
        self.text: list[str] = []
        self.chunks = 0
        self.model: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    def observe_chunk(self, chunk: Any) -> None:
        try:
            self.chunks += 1
            self.model = self.model or _get(chunk, "model")

            usage = _get(chunk, "usage")
            if usage is not None:
                prompt, completion = _usage_fields(usage)
                self.prompt_tokens = prompt if prompt is not None else self.prompt_tokens
                self.completion_tokens = (
                    completion if completion is not None else self.completion_tokens
                )

            for choice in _get(chunk, "choices") or ():
                delta = _get(choice, "delta")
                piece = _get(delta, "content") if delta is not None else None
                if isinstance(piece, str):
                    self.text.append(piece)

            # Responses API events carry their text on the event itself.
            piece = _get(chunk, "delta")
            if isinstance(piece, str):
                self.text.append(piece)
        except Exception:  # noqa: BLE001 - rule 2: a weird chunk is not fatal
            pass

    def apply(self, span: Any) -> None:
        try:
            observation = span.observation
            observation.model = self.model
            observation.prompt_tokens = self.prompt_tokens
            observation.completion_tokens = self.completion_tokens
            observation.metadata["stream_chunks"] = self.chunks
            output: Any = "".join(self.text) if self.text else None
        except Exception:  # noqa: BLE001 - rule 2
            output = None
        finish_span(span, output=output, summarise_output=False)


class _TracedStream:
    """Proxy around ``openai.Stream`` that closes the span when iteration ends.

    A proxy rather than a generator: user code reaches for ``.response``,
    ``.close()``, and ``with client...`` on the object it gets back, and a bare
    generator has none of those. ``__getattr__`` forwards everything this class
    does not define.
    """

    def __init__(self, stream: Any, span: Any) -> None:
        self._stream = stream
        self._span = span
        self._state = _StreamState()
        self._finished = False
        # Initialised here, not in __iter__: openai's Stream supports next()
        # directly, and __getattr__ would otherwise forward the lookup to the
        # wrapped stream and raise a confusing AttributeError.
        self._iterator: Any = None

    def __iter__(self) -> _TracedStream:
        self._ensure_iterator()
        return self

    def _ensure_iterator(self) -> None:
        if self._iterator is None:
            self._iterator = iter(self._stream)

    def __next__(self) -> Any:
        self._ensure_iterator()
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._settle()
            raise
        except BaseException as exc:
            self._settle(exc)
            raise
        self._state.observe_chunk(chunk)
        return chunk

    def __enter__(self) -> _TracedStream:
        self._stream.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        # Leaving the block ends the generation even if it was never drained.
        self._settle()
        return self._stream.__exit__(*exc)

    def close(self) -> None:
        self._settle()
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._stream, item)

    def _settle(self, exc: BaseException | None = None) -> None:
        """Close the span exactly once, however the stream ended."""
        if self._finished:
            return
        self._finished = True
        if exc is not None:
            finish_span(self._span, exc=exc)
        else:
            self._state.apply(self._span)


class _TracedAsyncStream:
    """``_TracedStream`` for ``AsyncStream``."""

    def __init__(self, stream: Any, span: Any) -> None:
        self._stream = stream
        self._span = span
        self._state = _StreamState()
        self._finished = False
        self._iterator: Any = None

    def __aiter__(self) -> _TracedAsyncStream:
        self._ensure_iterator()
        return self

    def _ensure_iterator(self) -> None:
        if self._iterator is None:
            self._iterator = self._stream.__aiter__()

    async def __anext__(self) -> Any:
        self._ensure_iterator()
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._settle()
            raise
        except BaseException as exc:
            self._settle(exc)
            raise
        self._state.observe_chunk(chunk)
        return chunk

    async def __aenter__(self) -> _TracedAsyncStream:
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        self._settle()
        return await self._stream.__aexit__(*exc)

    async def close(self) -> None:
        self._settle()
        close = getattr(self._stream, "close", None)
        if close is not None:
            await close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._stream, item)

    def _settle(self, exc: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        if exc is not None:
            finish_span(self._span, exc=exc)
        else:
            self._state.apply(self._span)


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


def _usage_fields(usage: Any) -> tuple[int | None, int | None]:
    """Chat completions say prompt/completion; the responses API says input/output."""
    prompt = _get(usage, "prompt_tokens")
    if prompt is None:
        prompt = _get(usage, "input_tokens")
    completion = _get(usage, "completion_tokens")
    if completion is None:
        completion = _get(usage, "output_tokens")
    return _as_int(prompt), _as_int(completion)


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _extract_usage(response: Any) -> tuple[int | None, int | None]:
    return _usage_fields(_get(response, "usage"))


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
