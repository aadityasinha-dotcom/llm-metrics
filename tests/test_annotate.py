"""Attribution, scores, sampling, redaction, deploy stamping, and stats.

These are the pieces that turn a token ledger into an observability tool:
who a request served, which prompt version it used, whether it was any good,
and whether the SDK itself is healthy. Each is tested through the public API
and the flushed wire payload, since that is what the server will see.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

import llm_metrics
from llm_metrics import _runtime, context, observe, score, update_observation, update_trace
from llm_metrics.buffer import EventBuffer
from llm_metrics.models import ScoreSource


class Collector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, payload: list[dict[str, Any]], deadline: object = None) -> None:
        self.events.extend(payload)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == kind]

    @property
    def traces(self) -> list[dict[str, Any]]:
        return self.of_type("trace")

    @property
    def observations(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") not in ("trace", "score")]

    @property
    def scores(self) -> list[dict[str, Any]]:
        return self.of_type("score")


class Pipeline:
    def __init__(self, sink: Collector, buffer: EventBuffer) -> None:
        self.sink = sink
        self.buffer = buffer

    def flush(self) -> Collector:
        self.buffer.flush_once()
        return self.sink


@pytest.fixture
def pipeline() -> Iterator[Pipeline]:
    sink = Collector()
    buffer = EventBuffer(sink, flush_at=10_000, flush_interval=300.0, start=False)
    _runtime.configure(
        sink=buffer,
        enabled=True,
        capture_input=True,
        capture_output=True,
        environment=None,
        release=None,
        sample_rate=1.0,
        redact=None,
    )
    try:
        yield Pipeline(sink, buffer)
    finally:
        _runtime.configure(environment=None, release=None, sample_rate=1.0, redact=None)
        _runtime.shutdown(timeout=1.0)


# --------------------------------------------------------------------------- #
# update_trace / update_observation
# --------------------------------------------------------------------------- #


def test_update_trace_annotates_the_ambient_trace(pipeline: Pipeline) -> None:
    @observe()
    def handle() -> None:
        assert update_trace(
            name="qa",
            user_id="u-1",
            session_id="s-9",
            tags=["beta", "qa", "beta"],
            metadata={"tenant": "acme"},
        )

    handle()

    (trace,) = pipeline.flush().traces
    assert trace["name"] == "qa"
    assert trace["user_id"] == "u-1"
    assert trace["session_id"] == "s-9"
    assert trace["tags"] == ["beta", "qa"], "tags are de-duplicated and ordered"
    assert trace["metadata"] == {"tenant": "acme"}


def test_update_trace_from_a_nested_call_reaches_the_root(pipeline: Pipeline) -> None:
    @observe()
    def inner() -> None:
        update_trace(user_id="deep")

    @observe()
    def outer() -> None:
        inner()

    outer()

    (trace,) = pipeline.flush().traces
    assert trace["user_id"] == "deep"


def test_update_trace_outside_a_trace_is_a_harmless_no_op(pipeline: Pipeline) -> None:
    assert update_trace(user_id="nobody") is False
    assert pipeline.flush().events == []


def test_update_observation_fills_in_hand_rolled_generations(pipeline: Pipeline) -> None:
    @observe(as_type="generation")
    def call_some_provider() -> str:
        assert update_observation(
            model="mystery-1",
            prompt_name="qa",
            prompt_version=3,
            prompt_tokens=120,
            completion_tokens=40,
            cached_tokens=100,
            reasoning_tokens=12,
            metadata={"region": "eu"},
        )
        return "ok"

    call_some_provider()

    (obs,) = pipeline.flush().observations
    assert obs["model"] == "mystery-1"
    assert obs["prompt_name"] == "qa"
    assert obs["prompt_version"] == "3", "versions are strings on the wire"
    assert obs["prompt_tokens"] == 120
    assert obs["completion_tokens"] == 40
    assert obs["cached_tokens"] == 100
    assert obs["reasoning_tokens"] == 12
    assert obs["metadata"]["region"] == "eu"


def test_update_observation_targets_the_innermost_call(pipeline: Pipeline) -> None:
    @observe()
    def inner() -> None:
        update_observation(name="renamed-inner")

    @observe()
    def outer() -> None:
        inner()

    outer()

    names = sorted(o["name"] for o in pipeline.flush().observations)
    assert names == ["outer", "renamed-inner"] or names == [
        "renamed-inner",
        "test_update_observation_targets_the_innermost_call.<locals>.outer",
    ]


def test_update_observation_outside_a_call_is_a_no_op(pipeline: Pipeline) -> None:
    assert update_observation(model="x") is False


def test_a_snapshot_carries_the_observation_across_a_thread(pipeline: Pipeline) -> None:
    import threading

    @observe()
    def work() -> None:
        snap = context.snapshot()

        def worker() -> None:
            with context.adopt(snap):
                update_observation(metadata={"from": "thread"})

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    work()

    (obs,) = pipeline.flush().observations
    assert obs["metadata"] == {"from": "thread"}


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def test_score_attaches_to_the_ambient_trace_and_observation(pipeline: Pipeline) -> None:
    @observe()
    def answer() -> None:
        assert score("faithful", True, comment="cites the doc", source=ScoreSource.HEURISTIC)

    answer()

    sink = pipeline.flush()
    (trace,) = sink.traces
    (obs,) = sink.observations
    (s,) = sink.scores
    assert s["name"] == "faithful"
    assert s["value"] is True
    assert s["comment"] == "cites the doc"
    assert s["source"] == "heuristic"
    assert s["trace_id"] == trace["id"]
    assert s["observation_id"] == obs["id"]
    assert "timestamp" in s


def test_score_after_the_fact_by_trace_id(pipeline: Pipeline) -> None:
    captured: list[str] = []

    @observe()
    def answer() -> None:
        captured.append(context.current_trace_id() or "")

    answer()
    score_id = score("thumbs", -1, trace_id=captured[0])

    (s,) = pipeline.flush().scores
    assert s["id"] == score_id
    assert s["trace_id"] == captured[0]
    assert "observation_id" not in s
    assert s["value"] == -1


def test_score_with_nothing_to_attach_to_records_nothing(pipeline: Pipeline) -> None:
    assert score("thumbs", 1) is None
    assert pipeline.flush().scores == []


def test_score_is_inert_when_disabled(pipeline: Pipeline) -> None:
    _runtime.configure(enabled=False)
    try:
        assert score("thumbs", 1, trace_id="t") is None
    finally:
        _runtime.configure(enabled=True)


def test_no_pricing_field_appears_on_any_new_event(pipeline: Pipeline) -> None:
    import json

    @observe()
    def answer() -> None:
        update_trace(user_id="u")
        update_observation(prompt_tokens=1, completion_tokens=1, cached_tokens=1)
        score("q", 0.5)

    answer()

    assert "cost" not in json.dumps(pipeline.flush().events).lower()


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #


def test_sample_rate_zero_drops_the_whole_tree(pipeline: Pipeline) -> None:
    _runtime.configure(sample_rate=0.0)

    @observe()
    def child() -> str:
        score("q", 1)
        return "still runs"

    @observe()
    def root() -> str:
        return child()

    assert root() == "still runs", "sampling never changes what the caller gets"
    assert pipeline.flush().events == []


def test_sample_rate_one_keeps_everything(pipeline: Pipeline) -> None:
    _runtime.configure(sample_rate=1.0)

    @observe()
    def root() -> None:
        pass

    root()
    assert len(pipeline.flush().traces) == 1


def test_sampling_is_decided_once_per_trace(pipeline: Pipeline) -> None:
    """Either the whole tree arrives or none of it. Holes would be worse
    than absence: a parent with missing children reads as a bug."""
    _runtime.configure(sample_rate=0.5)

    @observe()
    def child() -> None:
        pass

    @observe()
    def root() -> None:
        child()
        child()

    for _ in range(40):
        root()

    sink = pipeline.flush()
    trace_ids = {t["id"] for t in sink.traces}
    by_trace: dict[str, int] = {}
    for obs in sink.observations:
        by_trace[obs["trace_id"]] = by_trace.get(obs["trace_id"], 0) + 1
    assert set(by_trace) == trace_ids
    assert all(count == 3 for count in by_trace.values())
    assert 0 < len(trace_ids) < 40, "a coin that always lands the same way is broken"


def test_sample_rate_is_clamped(pipeline: Pipeline) -> None:
    _runtime.configure(sample_rate=7.0)
    assert _runtime.current_settings().sample_rate == 1.0
    _runtime.configure(sample_rate=-1.0)
    assert _runtime.current_settings().sample_rate == 0.0


def test_sample_rate_from_the_environment() -> None:
    os.environ["LLM_METRICS_SAMPLE_RATE"] = "0.25"
    try:
        assert _runtime._env_sample_rate() == 0.25
        os.environ["LLM_METRICS_SAMPLE_RATE"] = "not a number"
        assert _runtime._env_sample_rate() == 1.0, "garbage falls back to keeping everything"
    finally:
        del os.environ["LLM_METRICS_SAMPLE_RATE"]


# --------------------------------------------------------------------------- #
# redaction
# --------------------------------------------------------------------------- #


def test_redactor_runs_over_inputs_and_outputs(pipeline: Pipeline) -> None:
    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, str):
            return value.replace("secret", "[redacted]")
        return value

    _runtime.configure(redact=scrub)

    @observe()
    def echo(text: str) -> str:
        return f"you said: {text}"

    echo("my secret")

    (obs,) = pipeline.flush().observations
    assert obs["input"] == {"text": "my [redacted]"}
    assert obs["output"] == "you said: my [redacted]"


def test_a_redactor_that_raises_drops_the_value_rather_than_shipping_it(
    pipeline: Pipeline,
) -> None:
    def broken(value: Any) -> Any:
        raise RuntimeError("regex exploded")

    _runtime.configure(redact=broken)

    @observe()
    def echo(text: str) -> str:
        return text

    assert echo("pii") == "pii", "the caller is unaffected"

    (obs,) = pipeline.flush().observations
    assert "input" not in obs
    assert "output" not in obs


def test_a_redactor_returning_none_drops_the_value(pipeline: Pipeline) -> None:
    _runtime.configure(redact=lambda value: None)

    @observe()
    def echo(text: str) -> str:
        return text

    echo("pii")

    (obs,) = pipeline.flush().observations
    assert "input" not in obs
    assert "output" not in obs


def test_redactor_can_be_removed(pipeline: Pipeline) -> None:
    _runtime.configure(redact=lambda value: None)
    _runtime.configure(redact=None)
    assert _runtime.current_settings().redact is None


# --------------------------------------------------------------------------- #
# environment / release
# --------------------------------------------------------------------------- #


def test_environment_and_release_are_stamped_on_created_traces(pipeline: Pipeline) -> None:
    _runtime.configure(environment="prod", release="abc123")

    @observe()
    def root() -> None:
        pass

    root()

    (trace,) = pipeline.flush().traces
    assert trace["environment"] == "prod"
    assert trace["release"] == "abc123"


def test_environment_and_release_are_omitted_when_unset(pipeline: Pipeline) -> None:
    @observe()
    def root() -> None:
        pass

    root()

    (trace,) = pipeline.flush().traces
    assert "environment" not in trace
    assert "release" not in trace
    assert "tags" not in trace, "an empty tag list is not worth the bytes"


def test_environment_from_the_env_var() -> None:
    os.environ["LLM_METRICS_ENVIRONMENT"] = "  staging "
    os.environ["LLM_METRICS_RELEASE"] = ""
    try:
        assert _runtime._env_str("LLM_METRICS_ENVIRONMENT") == "staging"
        assert _runtime._env_str("LLM_METRICS_RELEASE") is None, "blank means unset"
    finally:
        del os.environ["LLM_METRICS_ENVIRONMENT"]
        del os.environ["LLM_METRICS_RELEASE"]


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def test_stats_reflect_the_running_pipeline(pipeline: Pipeline) -> None:
    @observe()
    def root() -> None:
        pass

    root()
    before = llm_metrics.stats()
    assert before.queued == 2, "one trace and one observation"
    assert before.flushed == 0

    pipeline.flush()
    after = llm_metrics.stats()
    assert after.flushed == 2
    assert after.healthy
    assert after.sent_batches is None, "a custom sink owns transport; there is no client"


def test_stats_report_overflow_drops() -> None:
    sink = Collector()
    buffer = EventBuffer(sink, max_size=2, flush_at=10_000, flush_interval=300.0, start=False)
    _runtime.configure(sink=buffer, enabled=True)
    try:

        @observe()
        def root() -> None:
            pass

        for _ in range(5):
            root()

        view = llm_metrics.stats()
        assert view.dropped_on_overflow == 8
        assert not view.healthy
    finally:
        _runtime.shutdown(timeout=1.0)


def test_stats_are_zero_before_anything_happens() -> None:
    _runtime.shutdown(timeout=1.0)
    view = llm_metrics.stats()
    assert view.queued == 0
    assert view.healthy


def test_public_surface_lists_the_new_api() -> None:
    for name in ("update_trace", "update_observation", "score", "stats", "Score", "ScoreSource"):
        assert name in llm_metrics.__all__
