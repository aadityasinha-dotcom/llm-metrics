"""Shared machinery for tracing a streamed provider response.

A provider stream is consumed by the *caller*, not by the SDK, so the span has
to stay open until iteration ends — normally, by exception, or by the caller
walking away — and close exactly once. The proxies here own that lifecycle and
delegate the provider-specific reading of each chunk to a :class:`StreamState`.

Timing that only a stream can give
----------------------------------
Total latency hides the number users feel. A stream that starts instantly and
one that stalls for four seconds can finish at the same moment. So the state
records the time to first token and, from there, the output rate:

* ``time_to_first_token_ms`` — from the request leaving to the first chunk
  that carried content
* ``output_tokens_per_second`` — completion tokens over the time between the
  first token and the end of the stream; the provider's generation speed,
  independent of queueing
* ``stream_completed`` — ``False`` when the caller closed the stream early
"""

from __future__ import annotations

import time
from typing import Any

from llm_metrics.decorator import finish_span

__all__ = ["StreamState", "TracedAsyncStream", "TracedStream"]


class StreamState:
    """Accumulates a streamed completion into one observation's worth of data.

    Subclasses implement :meth:`_observe` for their provider's chunk shape and
    :meth:`_apply` to write what they gathered onto the observation. Both are
    guarded here, so a subclass can read chunks optimistically.
    """

    __slots__ = (
        "chunks",
        "completion_tokens",
        "ended_at",
        "first_token_at",
        "span",
        "text",
    )

    def __init__(self, span: Any) -> None:
        self.span = span
        self.chunks = 0
        self.text: list[str] = []
        self.first_token_at: float | None = None
        self.ended_at: float | None = None
        #: Subclasses set this when the provider reports it, so the rate can
        #: be derived here without knowing where each provider keeps usage.
        self.completion_tokens: int | None = None

    # ----------------------------------------------------- subclass surface

    def _observe(self, chunk: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _apply(self, observation: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def mark_first_token(self) -> None:
        """Call from :meth:`_observe` on the first chunk that carries content."""
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    # ----------------------------------------------------------- lifecycle

    def observe_chunk(self, chunk: Any) -> None:
        try:
            self.chunks += 1
            self._observe(chunk)
        except Exception:  # noqa: BLE001 - rule 2: a weird chunk is not fatal
            pass

    def finish(self, exc: BaseException | None = None, *, completed: bool = True) -> None:
        """Close the span with everything gathered so far."""
        if exc is not None:
            finish_span(self.span, exc=exc)
            return
        output: Any = None
        try:
            self.ended_at = time.perf_counter()
            observation = self.span.observation
            self._apply(observation)
            metadata = observation.metadata
            metadata["stream_chunks"] = self.chunks
            metadata["stream_completed"] = completed
            if self.first_token_at is not None:
                started = self.span.started
                metadata["time_to_first_token_ms"] = round(
                    (self.first_token_at - started) * 1000.0, 3
                )
                generating = self.ended_at - self.first_token_at
                tokens = self.completion_tokens
                if tokens is not None and tokens > 0 and generating > 0:
                    metadata["output_tokens_per_second"] = round(tokens / generating, 2)
            output = "".join(self.text) if self.text else None
        except Exception:  # noqa: BLE001 - rule 2
            output = None
        finish_span(self.span, output=output, summarise_output=False)


class TracedStream:
    """Proxy around a provider's sync stream that closes the span on the way out.

    A proxy rather than a generator: user code reaches for ``.response``,
    ``.close()``, and ``with client...`` on the object it gets back, and a bare
    generator has none of those. ``__getattr__`` forwards everything this class
    does not define.
    """

    def __init__(self, stream: Any, state: StreamState) -> None:
        self._stream = stream
        self._state = state
        self._finished = False
        # Initialised here, not in __iter__: provider streams support next()
        # directly, and __getattr__ would otherwise forward the lookup to the
        # wrapped stream and raise a confusing AttributeError.
        self._iterator: Any = None

    def __iter__(self) -> TracedStream:
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
            self._settle(completed=True)
            raise
        except BaseException as exc:
            self._settle(exc)
            raise
        self._state.observe_chunk(chunk)
        return chunk

    def __enter__(self) -> TracedStream:
        self._stream.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        # Leaving the block ends the generation even if it was never drained.
        self._settle(completed=False)
        return self._stream.__exit__(*exc)

    def close(self) -> None:
        self._settle(completed=False)
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._stream, item)

    def _settle(self, exc: BaseException | None = None, *, completed: bool = False) -> None:
        """Close the span exactly once, however the stream ended."""
        if self._finished:
            return
        self._finished = True
        self._state.finish(exc, completed=completed)


class TracedAsyncStream:
    """:class:`TracedStream` for an async stream."""

    def __init__(self, stream: Any, state: StreamState) -> None:
        self._stream = stream
        self._state = state
        self._finished = False
        self._iterator: Any = None

    def __aiter__(self) -> TracedAsyncStream:
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
            self._settle(completed=True)
            raise
        except BaseException as exc:
            self._settle(exc)
            raise
        self._state.observe_chunk(chunk)
        return chunk

    async def __aenter__(self) -> TracedAsyncStream:
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        self._settle(completed=False)
        return await self._stream.__aexit__(*exc)

    async def close(self) -> None:
        self._settle(completed=False)
        close = getattr(self._stream, "close", None)
        if close is not None:
            await close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._stream, item)

    def _settle(self, exc: BaseException | None = None, *, completed: bool = False) -> None:
        if self._finished:
            return
        self._finished = True
        self._state.finish(exc, completed=completed)
