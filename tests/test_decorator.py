"""``@observe`` — tree shape, call shapes, and the never-break-the-caller rule.

Two things are being tested here and they pull in opposite directions. One is
that the decorator records accurate observations: right parent, right latency,
right input and output. The other is that when any of that recording goes
wrong, the wrapped function still behaves exactly as if the decorator were not
there. The second matters more, so it gets the harsher tests.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator

import pytest

from llmobserve import _runtime, observe
from llmobserve.buffer import EventBuffer
from llmobserve.decorator import _summarise
from llmobserve.models import ObservationStatus, ObservationType


class Collector:
    """Stands in for the buffer's flush target, keeping every event."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def __call__(self, payload: list[dict[str, object]], deadline: object = None) -> None:
        self.events.extend(payload)

    # --- helpers the tests read through -----------------------------------

    @property
    def observations(self) -> list[dict[str, object]]:
        return [e for e in self.events if e.get("type") != "trace"]

    @property
    def traces(self) -> list[dict[str, object]]:
        return [e for e in self.events if e.get("type") == "trace"]

    def named(self, name: str) -> dict[str, object]:
        matches = [e for e in self.observations if e["name"] == name]
        assert len(matches) == 1, f"expected exactly one {name!r}, got {len(matches)}"
        return matches[0]

    def names(self) -> list[str]:
        return [str(e["name"]) for e in self.observations]


class Pipeline:
    """The collector plus the buffer feeding it, flushed on demand."""

    def __init__(self, sink: Collector, buffer: EventBuffer) -> None:
        self.sink = sink
        self.buffer = buffer

    def flush(self) -> Collector:
        self.buffer.flush_once()
        return self.sink


@pytest.fixture
def pipeline() -> Iterator[Pipeline]:
    """A decorator pipeline that keeps its events in memory."""
    sink = Collector()
    buffer = EventBuffer(sink, flush_at=10_000, flush_interval=300.0, start=False)
    _runtime.configure(sink=buffer, enabled=True, capture_input=True, capture_output=True)
    try:
        yield Pipeline(sink, buffer)
    finally:
        _runtime.shutdown(timeout=1.0)


@pytest.fixture
def events(pipeline: Pipeline) -> Pipeline:
    """Alias for tests that only ever read flushed events."""
    return pipeline


# --------------------------------------------------------------------------- #
# Shape of what gets recorded
# --------------------------------------------------------------------------- #


def test_a_plain_call_produces_one_trace_and_one_observation(events: Pipeline) -> None:
    @observe()
    def answer(question: str) -> str:
        return f"re: {question}"

    assert answer("why") == "re: why"

    sink = events.flush()
    assert len(sink.traces) == 1
    assert len(sink.observations) == 1

    obs = sink.observations[0]
    assert obs["name"] == "test_a_plain_call_produces_one_trace_and_one_observation.<locals>.answer"
    assert obs["trace_id"] == sink.traces[0]["id"]
    assert obs["status"] == ObservationStatus.OK
    assert obs["input"] == {"question": "why"}
    assert obs["output"] == "re: why"
    assert "parent_id" not in obs, "a root observation has no parent"


def test_bare_and_called_decorator_forms_both_work(events: Pipeline) -> None:
    @observe
    def bare() -> str:
        return "a"

    @observe()
    def called() -> str:
        return "b"

    @observe(name="renamed", as_type=ObservationType.GENERATION)
    def configured() -> str:
        return "c"

    assert (bare(), called(), configured()) == ("a", "b", "c")

    sink = events.flush()
    assert "renamed" in sink.names()
    assert sink.named("renamed")["type"] == ObservationType.GENERATION


def test_latency_reflects_the_wrapped_call(events: Pipeline) -> None:
    @observe()
    def slow() -> None:
        time.sleep(0.05)

    slow()

    latency = events.flush().observations[0]["latency_ms"]
    assert isinstance(latency, float)
    assert 40 < latency < 500, f"latency_ms was {latency}"


def test_metadata_is_attached_to_every_call(events: Pipeline) -> None:
    @observe(metadata={"tier": "premium"})
    def call() -> None:
        pass

    call()
    call()

    sink = events.flush()
    assert [e["metadata"] for e in sink.observations] == [{"tier": "premium"}] * 2


def test_self_is_not_captured_as_input(events: Pipeline) -> None:
    class Agent:
        @observe()
        def run(self, prompt: str) -> str:
            return prompt

    Agent().run("hi")

    assert events.flush().observations[0]["input"] == {"prompt": "hi"}


def test_capture_can_be_switched_off_per_decorator(events: Pipeline) -> None:
    @observe(capture_input=False, capture_output=False)
    def secret(password: str) -> str:
        return "token"

    secret("hunter2")

    obs = events.flush().observations[0]
    assert "input" not in obs
    assert "output" not in obs


def test_capture_can_be_switched_off_globally(pipeline: Pipeline) -> None:
    _runtime.configure(sink=pipeline.buffer, capture_input=False, capture_output=False)

    @observe()
    def call(secret: str) -> str:
        return "result"

    call("hunter2")

    obs = pipeline.flush().observations[0]
    assert "input" not in obs
    assert "output" not in obs


# --------------------------------------------------------------------------- #
# Nesting
# --------------------------------------------------------------------------- #


def test_nested_calls_build_a_parent_chain(events: Pipeline) -> None:
    @observe(name="inner")
    def inner() -> str:
        return "leaf"

    @observe(name="middle")
    def middle() -> str:
        return inner()

    @observe(name="outer")
    def outer() -> str:
        return middle()

    outer()

    sink = events.flush()
    assert len(sink.traces) == 1, "nested calls must share one trace"

    o, m, i = sink.named("outer"), sink.named("middle"), sink.named("inner")
    assert {o["trace_id"], m["trace_id"], i["trace_id"]} == {sink.traces[0]["id"]}
    assert "parent_id" not in o
    assert m["parent_id"] == o["id"]
    assert i["parent_id"] == m["id"]


def test_sequential_children_are_siblings(events: Pipeline) -> None:
    @observe(name="child")
    def child(i: int) -> int:
        return i

    @observe(name="parent")
    def parent() -> None:
        for i in range(3):
            child(i)

    parent()

    sink = events.flush()
    parent_id = sink.named("parent")["id"]
    children = [e for e in sink.observations if e["name"] == "child"]
    assert [c["parent_id"] for c in children] == [parent_id] * 3


def test_a_failing_child_does_not_capture_its_siblings(events: Pipeline) -> None:
    @observe(name="boom")
    def boom() -> None:
        raise ValueError("nope")

    @observe(name="after")
    def after() -> None:
        pass

    @observe(name="parent")
    def parent() -> None:
        with pytest.raises(ValueError, match="nope"):
            boom()
        after()

    parent()

    sink = events.flush()
    parent_id = sink.named("parent")["id"]
    assert sink.named("boom")["parent_id"] == parent_id
    assert sink.named("after")["parent_id"] == parent_id, (
        "the sibling after a failure nested under the failure"
    )


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_an_exception_is_recorded_and_re_raised_unchanged(events: Pipeline) -> None:
    sentinel = RuntimeError("model overloaded")

    @observe()
    def failing() -> None:
        raise sentinel

    with pytest.raises(RuntimeError) as caught:
        failing()

    assert caught.value is sentinel, "the original exception object must reach the caller"

    obs = events.flush().observations[0]
    assert obs["status"] == ObservationStatus.ERROR
    assert obs["status_message"] == "RuntimeError: model overloaded"


def test_no_traceback_is_shipped(events: Pipeline) -> None:
    """Tracebacks carry source lines and locals nobody consented to send."""

    @observe()
    def failing() -> None:
        api_key = "sk-secret"  # noqa: F841 - deliberately a local
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        failing()

    assert "sk-secret" not in str(events.flush().observations[0])


def test_keyboard_interrupt_still_closes_the_observation(events: Pipeline) -> None:
    """BaseException, not Exception — a cancelled call is still a finished one."""

    @observe()
    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()

    assert events.flush().observations[0]["status"] == ObservationStatus.ERROR


# --------------------------------------------------------------------------- #
# async
# --------------------------------------------------------------------------- #


def test_async_functions_are_traced(events: Pipeline) -> None:
    @observe()
    async def fetch(url: str) -> str:
        await asyncio.sleep(0.01)
        return f"body of {url}"

    assert asyncio.run(fetch("/x")) == "body of /x"

    obs = events.flush().observations[0]
    assert obs["input"] == {"url": "/x"}
    assert obs["output"] == "body of /x"
    assert isinstance(obs["latency_ms"], float)


def test_async_nesting_across_await(events: Pipeline) -> None:
    @observe(name="inner")
    async def inner() -> str:
        await asyncio.sleep(0.01)
        return "leaf"

    @observe(name="outer")
    async def outer() -> str:
        return await inner()

    asyncio.run(outer())

    sink = events.flush()
    assert sink.named("inner")["parent_id"] == sink.named("outer")["id"]


def test_gathered_calls_are_siblings_not_a_chain(events: Pipeline) -> None:
    """The property contextvars buy: concurrency must not distort the tree."""

    @observe(name="leaf")
    async def leaf(i: int) -> int:
        await asyncio.sleep(0.02 - i * 0.005)  # finish out of order
        return i

    @observe(name="root")
    async def root() -> None:
        await asyncio.gather(*(leaf(i) for i in range(3)))

    asyncio.run(root())

    sink = events.flush()
    root_id = sink.named("root")["id"]
    leaves = [e for e in sink.observations if e["name"] == "leaf"]
    assert len(leaves) == 3
    assert [leaf["parent_id"] for leaf in leaves] == [root_id] * 3


def test_async_exceptions_propagate_unchanged(events: Pipeline) -> None:
    @observe()
    async def failing() -> None:
        raise ValueError("async boom")

    with pytest.raises(ValueError, match="async boom"):
        asyncio.run(failing())

    assert events.flush().observations[0]["status"] == ObservationStatus.ERROR


# --------------------------------------------------------------------------- #
# Generators — streaming
# --------------------------------------------------------------------------- #


def test_a_generator_is_timed_over_the_whole_stream(events: Pipeline) -> None:
    """Wrapping a generator like a plain function would time only how long it
    took to build the generator, which for a streamed completion is nonsense."""

    @observe()
    def stream() -> Iterator[str]:
        for i in range(3):
            time.sleep(0.02)
            yield f"chunk-{i}"

    assert list(stream()) == ["chunk-0", "chunk-1", "chunk-2"]

    obs = events.flush().observations[0]
    latency = obs["latency_ms"]
    assert isinstance(latency, float)
    assert latency > 50, f"latency_ms was {latency} — the stream was not timed"
    assert obs["output"] == ["chunk-0", "chunk-1", "chunk-2"]


def test_nothing_is_recorded_until_a_generator_is_consumed(events: Pipeline) -> None:
    @observe()
    def stream() -> Iterator[str]:
        yield "only"

    generator = stream()
    assert events.flush().observations == [], "an unstarted generator recorded a call"

    list(generator)
    assert len(events.flush().observations) == 1


def test_an_abandoned_generator_records_what_it_produced(events: Pipeline) -> None:
    @observe()
    def stream() -> Iterator[int]:
        yield from range(100)

    for chunk in stream():
        if chunk == 2:
            break  # closes the generator

    obs = events.flush().observations[0]
    assert obs["status"] == ObservationStatus.OK, "abandoning a stream is not a failure"
    assert obs["output"] == [0, 1, 2]


def test_a_generator_that_raises_mid_stream_is_recorded(events: Pipeline) -> None:
    @observe()
    def stream() -> Iterator[str]:
        yield "ok"
        raise RuntimeError("stream died")

    with pytest.raises(RuntimeError, match="stream died"):
        list(stream())

    obs = events.flush().observations[0]
    assert obs["status"] == ObservationStatus.ERROR
    assert obs["status_message"] == "RuntimeError: stream died"


def test_calls_inside_a_generator_body_nest_correctly(events: Pipeline) -> None:
    @observe(name="tool")
    def tool(i: int) -> int:
        return i

    @observe(name="stream")
    def stream() -> Iterator[int]:
        for i in range(2):
            yield tool(i)

    list(stream())

    sink = events.flush()
    stream_id = sink.named("stream")["id"]
    tools = [e for e in sink.observations if e["name"] == "tool"]
    assert [t["parent_id"] for t in tools] == [stream_id] * 2


def test_async_generators_are_traced(events: Pipeline) -> None:
    @observe()
    async def stream() -> AsyncIterator[int]:
        for i in range(3):
            await asyncio.sleep(0.01)
            yield i

    async def consume() -> list[int]:
        return [chunk async for chunk in stream()]

    assert asyncio.run(consume()) == [0, 1, 2]

    obs = events.flush().observations[0]
    assert obs["output"] == [0, 1, 2]
    assert isinstance(obs["latency_ms"], float)
    assert obs["latency_ms"] > 20


# --------------------------------------------------------------------------- #
# Value capture
# --------------------------------------------------------------------------- #


def test_long_values_are_truncated(events: Pipeline) -> None:
    @observe()
    def echo(prompt: str) -> str:
        return prompt

    echo("x" * 10_000)

    captured = events.flush().observations[0]["input"]
    assert isinstance(captured, dict)
    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert len(prompt) < 3_000
    assert "truncated" in prompt


def test_message_structure_survives_capture(events: Pipeline) -> None:
    """The shape of a messages list is most of what makes a trace readable."""

    @observe()
    def complete(messages: list[dict[str, str]]) -> dict[str, str]:
        return {"role": "assistant", "content": "hi"}

    complete([{"role": "user", "content": "hello"}])

    obs = events.flush().observations[0]
    assert obs["input"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert obs["output"] == {"role": "assistant", "content": "hi"}


def test_an_object_with_a_hostile_repr_does_not_break_the_call(events: Pipeline) -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("no repr for you")

    @observe()
    def call(thing: object) -> str:
        return "fine"

    assert call(Hostile()) == "fine", "a broken __repr__ must not break the call"

    captured = events.flush().observations[0]["input"]
    assert isinstance(captured, dict)
    assert "unreprable" in str(captured["thing"])


def test_summarise_bounds_wide_collections() -> None:
    summarised = _summarise(list(range(500)), limit=100)
    assert isinstance(summarised, list)
    assert len(summarised) == 101
    assert "400 more" in str(summarised[-1])


def test_summarise_stops_descending_at_depth() -> None:
    deep = {"a": {"b": {"c": {"d": {"e": "bottom"}}}}}
    summarised = _summarise(deep, limit=500)
    assert "bottom" in str(summarised), "the value should still be represented"


# --------------------------------------------------------------------------- #
# Rule 2: the decorator must never break the function it wraps
# --------------------------------------------------------------------------- #


def test_the_call_still_runs_when_the_sdk_is_disabled(pipeline: Pipeline) -> None:
    _runtime.configure(sink=pipeline.buffer, enabled=False)

    @observe()
    def call(x: int) -> int:
        return x * 2

    assert call(21) == 42
    assert pipeline.flush().events == [], "a disabled SDK must not record"


def test_the_call_still_runs_when_the_sink_is_broken(pipeline: Pipeline) -> None:
    def explode(_payload: object, _deadline: object = None) -> None:
        raise RuntimeError("sink is down")

    broken = EventBuffer(explode, flush_at=1, flush_interval=300.0, start=False)
    _runtime.configure(sink=broken, enabled=True)

    @observe()
    def call(x: int) -> int:
        return x * 2

    assert call(21) == 42
    broken.flush_once()  # must not propagate


def test_the_call_still_runs_when_span_creation_fails(
    monkeypatch: pytest.MonkeyPatch, events: Pipeline
) -> None:
    """If _begin blows up, the function runs with no instrumentation at all."""

    def broken_settings() -> object:
        raise RuntimeError("settings exploded")

    monkeypatch.setattr(_runtime, "current_settings", broken_settings)

    @observe()
    def call(x: int) -> int:
        return x * 2

    assert call(21) == 42
    assert events.flush().observations == []


def test_exceptions_still_propagate_when_span_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_settings() -> object:
        raise RuntimeError("settings exploded")

    monkeypatch.setattr(_runtime, "current_settings", broken_settings)

    @observe()
    def call() -> None:
        raise ValueError("the real error")

    with pytest.raises(ValueError, match="the real error"):
        call()


def test_the_call_still_runs_when_emit_fails(
    monkeypatch: pytest.MonkeyPatch, events: Pipeline
) -> None:
    def broken_emit(_event: object) -> None:
        raise RuntimeError("emit exploded")

    monkeypatch.setattr(_runtime, "emit", broken_emit)

    @observe()
    def call(x: int) -> int:
        return x * 2

    assert call(21) == 42


def test_a_failed_finish_does_not_leak_the_context_scope(
    monkeypatch: pytest.MonkeyPatch, events: Pipeline
) -> None:
    """The nastiest failure mode: if _finish dies before closing the scope,
    every later call in the task nests under a dead observation."""
    from llmobserve import context

    calls = {"n": 0}

    def flaky_emit(event: object) -> None:
        calls["n"] += 1
        raise RuntimeError("emit exploded")

    monkeypatch.setattr(_runtime, "emit", flaky_emit)

    @observe()
    def call() -> None:
        pass

    call()
    assert calls["n"] > 0, "emit should have been attempted"
    assert context.current_parent_id() is None, "the observation scope leaked"
    assert context.current_trace() is None, "the trace scope leaked"


def test_functools_wraps_metadata_is_preserved() -> None:
    @observe()
    def documented(x: int) -> int:
        """A docstring worth keeping."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring worth keeping."


def test_decorating_a_builtin_without_a_signature_does_not_explode(events: Pipeline) -> None:
    """inspect.signature raises for some C callables; capture must degrade."""
    wrapped = observe()(len)
    assert wrapped([1, 2, 3]) == 3
    assert len(events.flush().observations) == 1


def test_settings_only_configure_does_not_build_a_pipeline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adjusting a capture flag must not start a thread or warn about keys.

    ``_sink()`` builds the default pipeline lazily on first emit, so eager
    construction here would only produce a stray flush thread and a "no API
    key" warning for a caller who never asked for either.
    """
    monkeypatch.setattr(_runtime, "_buffer", None)
    monkeypatch.setattr(_runtime, "_client", None)

    _runtime.configure(capture_input=False, enabled=False)

    assert _runtime._buffer is None, "a settings-only call built a pipeline"
    assert "no API key" not in capsys.readouterr().err


def test_configure_with_transport_args_does_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_runtime, "_buffer", None)
    monkeypatch.setattr(_runtime, "_client", None)
    try:
        _runtime.configure(api_key="k", host="https://example.invalid")
        assert _runtime._buffer is not None
    finally:
        _runtime.shutdown(timeout=1.0)
