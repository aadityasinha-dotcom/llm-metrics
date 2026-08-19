"""``@observe`` — the SDK's entire public surface for instrumenting code.

    @observe()
    def answer(question: str) -> str:
        ...

Wrapping a function records one :class:`~llm_metrics.models.Observation`: how
long it took, what went in, what came out, and whether it raised. If no trace
is open, a root :class:`~llm_metrics.models.Trace` is created and closed around
it; if one is, the observation nests under whatever is currently open.

The load-bearing rule
---------------------
**The wrapped function runs, and its result or exception reaches the caller,
no matter what goes wrong in here.** Every piece of SDK work is behind a
try/except that degrades to a plain call. If ``_begin`` fails the function is
invoked directly with no instrumentation at all; if ``_finish`` fails the
result is still returned and the exception still propagates unchanged. An
observability decorator that can break the function it wraps is worse than no
decorator, and this one sits on every interesting call path in the app.

Four call shapes
----------------
Coroutines, generators, and async generators each need their own wrapper.
Wrapping a generator function like a plain one would time only how long it
took to *build* the generator — microseconds — and record no output, which is
exactly wrong for a streamed completion that ran for eight seconds. The
generator wrappers keep the observation open until the stream is exhausted or
abandoned, and re-enter the trace context around each resumption so anything
the body instruments still nests correctly.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any, TypeVar, cast, overload

from llm_metrics import _runtime, context
from llm_metrics.models import Observation, ObservationType, Trace

__all__ = ["finish_span", "observe", "open_span", "summarise"]

F = TypeVar("F", bound=Callable[..., Any])

#: Never captured as input, whatever the signature calls them.
_SKIPPED_PARAMS = frozenset({"self", "cls"})

#: Depth past which captured values are summarised rather than walked. Deep
#: structures are usually frameworks' internals, not the user's prompt.
_MAX_DEPTH = 3
_MAX_ITEMS = 100


def _default_name(fn: Callable[..., Any]) -> str:
    """``module.Class.method`` where available, so names stay unambiguous."""
    for attr in ("__qualname__", "__name__"):
        value = getattr(fn, attr, None)
        if isinstance(value, str) and value:
            return value
    return "observed"


class _Spec:
    """Everything decided once, at decoration time, so the call path stays thin."""

    __slots__ = (
        "as_type",
        "capture_input",
        "capture_output",
        "metadata",
        "name",
        "signature",
    )

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None,
        as_type: str,
        capture_input: bool | None,
        capture_output: bool | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        self.name: str = name or _default_name(fn)
        self.as_type = as_type
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.metadata = dict(metadata) if metadata else None
        # inspect.signature costs ~50us, so it is paid once here rather than
        # on every call. Binding against it later is cheap.
        try:
            self.signature: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):  # builtins, some C callables
            self.signature = None


class _Span:
    """One in-flight observation plus the context scopes it opened."""

    __slots__ = ("observation", "stack", "started", "trace")

    def __init__(self, observation: Observation, trace: Trace | None) -> None:
        self.observation = observation
        #: Set only when *this* span created the root trace and therefore owns
        #: closing and emitting it.
        self.trace = trace
        self.stack = ExitStack()
        self.started = time.perf_counter()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Make this span's trace and observation current for a block.

        Entered once around a plain call, and once per resumption for
        generators — which is why it builds fresh scopes each time instead of
        holding tokens open.
        """
        with ExitStack() as stack:
            if self.trace is not None:
                stack.enter_context(context.use_trace(self.trace))
            stack.enter_context(context.use_observation(self.observation))
            yield


# --------------------------------------------------------------------------- #
# Value capture
# --------------------------------------------------------------------------- #


def _summarise(value: Any, limit: int, depth: int = 0) -> Any:
    """Reduce an arbitrary value to something small and JSON-friendly.

    Structure is preserved where it is cheap to do so, because the structure of
    a messages list is most of what makes a trace readable. Everything else
    degrades to a bounded string rather than being dropped.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(value, limit)
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if depth >= _MAX_DEPTH:
        return _clip(_safe_repr(value), limit)
    if isinstance(value, Mapping):
        out = {}
        for i, (key, item) in enumerate(value.items()):
            if i >= _MAX_ITEMS:
                out["..."] = f"+{len(value) - _MAX_ITEMS} more"
                break
            out[str(key)] = _summarise(item, limit, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        summarised: list[Any] = [_summarise(i, limit, depth + 1) for i in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            summarised.append(f"...+{len(items) - _MAX_ITEMS} more")
        return summarised
    return _clip(_safe_repr(value), limit)


def summarise(value: Any) -> Any:
    """Bound an arbitrary value using the current settings. Never raises.

    Exposed for integrations, which receive framework payloads of unbounded
    size and shape and need the same treatment the decorator gives arguments.
    """
    try:
        return _summarise(value, _runtime.current_settings().max_value_chars)
    except Exception:  # noqa: BLE001 - rule 2
        return None


def _safe_repr(value: Any) -> str:
    """``repr`` that cannot take the host app down.

    User objects have user-written ``__repr__`` methods, and those raise.
    """
    try:
        return repr(value)
    except Exception:  # noqa: BLE001 - rule 2
        return f"<unreprable {type(value).__name__}>"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} chars truncated]"


def _capture_arguments(
    spec: _Spec, args: Sequence[Any], kwargs: Mapping[str, Any], limit: int
) -> Any:
    """Bind the call against the signature so inputs read as named parameters."""
    if spec.signature is None:
        return _summarise({"args": list(args), "kwargs": dict(kwargs)}, limit)
    try:
        bound = spec.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        # The call does not match the signature. Python is about to raise the
        # real error; do not pre-empt it with one of ours.
        return _summarise({"args": list(args), "kwargs": dict(kwargs)}, limit)
    return {
        name: _summarise(value, limit)
        for name, value in bound.arguments.items()
        if name not in _SKIPPED_PARAMS
    }


# --------------------------------------------------------------------------- #
# Span lifecycle — every function here swallows its own failures
# --------------------------------------------------------------------------- #


def open_span(
    name: str,
    as_type: str,
    *,
    input_value: Any = None,
    metadata: Mapping[str, Any] | None = None,
    trace: Trace | None = None,
    parent_id: str | None = None,
) -> _Span | None:
    """Open a span directly, without a function to introspect.

    The shared core of :func:`observe` and the integrations, which instrument
    third-party call sites where there is no signature to bind against.
    Returns ``None`` to mean "carry on uninstrumented" — never raises.

    Args:
        trace: Override the ambient trace. Needed by callback-style
            integrations such as LangChain, which are handed an explicit run
            tree and may be invoked from a thread where the contextvars are
            empty.
        parent_id: Override the ambient parent, for the same reason.
    """
    try:
        if not _runtime.is_enabled():
            return None

        if trace is None:
            trace = context.current_trace()
        created_trace = None
        if trace is None:
            # No trace open, so this call is the root of one.
            created_trace = Trace(name=name)
            trace = created_trace

        observation = Observation(
            trace_id=trace.id,
            name=name,
            type=as_type,
            parent_id=parent_id if parent_id is not None else context.current_parent_id(),
            metadata=dict(metadata) if metadata else {},
        )
        observation.input = input_value
        return _Span(observation, created_trace)
    except Exception:  # noqa: BLE001 - rule 2: fall back to no instrumentation
        return None


def _begin(spec: _Spec, args: Sequence[Any], kwargs: Mapping[str, Any]) -> _Span | None:
    """Open a span for a decorated call."""
    try:
        if not _runtime.is_enabled():
            return None
        settings = _runtime.current_settings()
        capture = spec.capture_input if spec.capture_input is not None else settings.capture_input
        return open_span(
            spec.name,
            spec.as_type,
            input_value=(
                _capture_arguments(spec, args, kwargs, settings.max_value_chars)
                if capture
                else None
            ),
            metadata=spec.metadata,
        )
    except Exception:  # noqa: BLE001 - rule 2
        return None


def finish_span(
    span: _Span,
    output: Any = None,
    exc: BaseException | None = None,
    *,
    capture_output: bool | None = None,
    summarise_output: bool = True,
) -> None:
    """Close a span and queue it. Never raises.

    Args:
        summarise_output: ``False`` when the caller has already reduced the
            output itself — integrations extract a small, meaningful shape from
            a provider response rather than letting the generic summariser
            flatten it.
    """
    try:
        observation = span.observation
        observation.latency_ms = (time.perf_counter() - span.started) * 1000.0
        if exc is not None:
            observation.fail(exc)
        elif output is not None:
            settings = _runtime.current_settings()
            capture = capture_output if capture_output is not None else settings.capture_output
            if capture:
                observation.output = (
                    _summarise(output, settings.max_value_chars) if summarise_output else output
                )
        observation.end()

        # Trace first: it is the parent entity, and the buffer preserves order.
        if span.trace is not None:
            span.trace.end()
            _runtime.emit(span.trace)
        _runtime.emit(observation)
    except Exception:  # noqa: BLE001 - rule 2
        pass
    finally:
        # Must happen even if the above blew up, or the contextvar scopes leak
        # and every later call in this task nests under a dead observation.
        with contextlib.suppress(Exception):  # rule 2
            span.stack.close()


def _finish(span: _Span, spec: _Spec, output: Any = None, exc: BaseException | None = None) -> None:
    """``finish_span`` with the decorator's per-function capture setting."""
    finish_span(span, output, exc, capture_output=spec.capture_output)


# --------------------------------------------------------------------------- #
# The decorator
# --------------------------------------------------------------------------- #


@overload
def observe(fn: F) -> F: ...


@overload
def observe(
    *,
    name: str | None = ...,
    as_type: str = ...,
    capture_input: bool | None = ...,
    capture_output: bool | None = ...,
    metadata: Mapping[str, Any] | None = ...,
) -> Callable[[F], F]: ...


def observe(
    fn: F | None = None,
    *,
    name: str | None = None,
    as_type: str = ObservationType.SPAN,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> F | Callable[[F], F]:
    """Record a trace observation for each call of the wrapped function.

    Works bare (``@observe``) or called (``@observe(name="rag")``), on sync and
    async functions, and on generators and async generators.

    Args:
        name: Defaults to the function's qualified name.
        as_type: One of :class:`~llm_metrics.models.ObservationType` — use
            ``"generation"`` for LLM calls, ``"tool"`` for tool invocations.
        capture_input: Record the arguments. Defaults to the global setting.
        capture_output: Record the return value. Defaults to the global setting.
        metadata: Static metadata attached to every call.
    """

    def decorate(func: F) -> F:
        spec = _Spec(func, name, as_type, capture_input, capture_output, metadata)

        if inspect.isasyncgenfunction(func):
            return cast(F, _wrap_async_gen(func, spec))
        if inspect.iscoroutinefunction(func):
            return cast(F, _wrap_async(func, spec))
        if inspect.isgeneratorfunction(func):
            return cast(F, _wrap_gen(func, spec))
        return cast(F, _wrap_sync(func, spec))

    if fn is not None:  # bare @observe
        return decorate(fn)
    return decorate


def _wrap_sync(func: Callable[..., Any], spec: _Spec) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span = _begin(spec, args, kwargs)
        if span is None:
            return func(*args, **kwargs)
        try:
            with span.activate():
                result = func(*args, **kwargs)
        except BaseException as exc:
            # BaseException, not Exception: a cancelled or interrupted call is
            # still a finished observation, and the scope must close either way.
            _finish(span, spec, exc=exc)
            raise
        _finish(span, spec, output=result)
        return result

    return wrapper


def _wrap_async(func: Callable[..., Awaitable[Any]], spec: _Spec) -> Callable[..., Any]:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span = _begin(spec, args, kwargs)
        if span is None:
            return await func(*args, **kwargs)
        try:
            with span.activate():
                result = await func(*args, **kwargs)
        except BaseException as exc:
            _finish(span, spec, exc=exc)
            raise
        _finish(span, spec, output=result)
        return result

    return wrapper


def _wrap_gen(func: Callable[..., Iterator[Any]], spec: _Spec) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
        span = _begin(spec, args, kwargs)
        if span is None:
            yield from func(*args, **kwargs)
            return

        chunks: list[Any] = []
        try:
            iterator = func(*args, **kwargs)
            while True:
                # Active only while the body runs, not while the consumer has
                # control — otherwise the consumer's own calls would nest here.
                with span.activate():
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        break
                chunks.append(chunk)
                yield chunk
        except GeneratorExit:
            # The consumer walked away. That is a normal end to a stream, not
            # a failure — record what was produced and re-raise so the
            # underlying generator still gets closed.
            _finish(span, spec, output=chunks)
            raise
        except BaseException as exc:
            _finish(span, spec, exc=exc)
            raise
        _finish(span, spec, output=chunks)

    return wrapper


def _wrap_async_gen(func: Callable[..., AsyncIterator[Any]], spec: _Spec) -> Callable[..., Any]:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        span = _begin(spec, args, kwargs)
        if span is None:
            async for chunk in func(*args, **kwargs):
                yield chunk
            return

        chunks: list[Any] = []
        try:
            iterator = func(*args, **kwargs).__aiter__()
            while True:
                with span.activate():
                    try:
                        chunk = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                chunks.append(chunk)
                yield chunk
        except GeneratorExit:
            _finish(span, spec, output=chunks)
            raise
        except BaseException as exc:
            _finish(span, spec, exc=exc)
            raise
        _finish(span, spec, output=chunks)

    return wrapper
