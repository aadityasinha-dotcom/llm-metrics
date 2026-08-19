"""LangChain callback handler.

Most of these drive real LangChain runnables — fake chat models, real chains,
real tools — because the handler's whole job is to reconstruct a tree from
LangChain's ``run_id``/``parent_run_id`` pairs, and hand-written callback
sequences would be testing my guess at those pairs rather than the real ones.

The synthetic tests that remain cover the shapes LangChain does not readily
produce on demand: evicted runs, orphaned parents, hostile payloads.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest

from llm_metrics import _runtime, observe
from llm_metrics.buffer import EventBuffer
from llm_metrics.models import ObservationStatus, ObservationType

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from llm_metrics.integrations.langchain import LlmMetricsTracer

# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class Collector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, payload: list[dict[str, Any]], deadline: object = None) -> None:
        self.events.extend(payload)


class Pipeline:
    def __init__(self, sink: Collector, buffer: EventBuffer) -> None:
        self.sink = sink
        self.buffer = buffer

    def flush(self) -> list[dict[str, Any]]:
        self.buffer.flush_once()
        return self.sink.events

    def observations(self) -> list[dict[str, Any]]:
        return [e for e in self.flush() if e["type"] != "trace"]

    def traces(self) -> list[dict[str, Any]]:
        return [e for e in self.flush() if e["type"] == "trace"]

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.observations() if e["type"] == kind]

    def one(self, kind: str) -> dict[str, Any]:
        matches = self.of_type(kind)
        assert len(matches) == 1, f"expected one {kind}, got {[m['name'] for m in matches]}"
        return matches[0]

    def tree(self) -> dict[str, list[str]]:
        """``parent name -> child names``, for asserting on shape."""
        by_id = {e["id"]: e for e in self.observations()}
        shape: dict[str, list[str]] = {}
        for event in self.observations():
            parent = by_id.get(event.get("parent_id", ""))
            shape.setdefault(parent["name"] if parent else "<root>", []).append(event["name"])
        return shape


@pytest.fixture
def pipeline() -> Iterator[Pipeline]:
    sink = Collector()
    buffer = EventBuffer(sink, flush_at=10_000, flush_interval=300.0, start=False)
    _runtime.configure(sink=buffer, enabled=True)
    try:
        yield Pipeline(sink, buffer)
    finally:
        _runtime.shutdown(timeout=1.0)


def chat_model(*replies: str, usage: dict[str, int] | None = None) -> GenericFakeChatModel:
    messages = [
        AIMessage(content=reply, usage_metadata=usage) if usage else AIMessage(content=reply)
        for reply in replies
    ]
    return GenericFakeChatModel(messages=iter(messages))


# --------------------------------------------------------------------------- #
# Real runs
# --------------------------------------------------------------------------- #


def test_a_chat_model_call_becomes_a_generation(pipeline: Pipeline) -> None:
    llm = chat_model("Paris.", usage={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14})

    result = llm.invoke("capital of France?", config={"callbacks": [LlmMetricsTracer()]})

    assert result.content == "Paris.", "the caller must get the real result"

    generation = pipeline.one(ObservationType.GENERATION)
    assert generation["name"] == "GenericFakeChatModel"
    assert generation["prompt_tokens"] == 11
    assert generation["completion_tokens"] == 3
    assert generation["output"] == "Paris."
    assert generation["input"] == [{"role": "human", "content": "capital of France?"}]
    assert isinstance(generation["latency_ms"], float)


def test_a_chain_produces_a_nested_tree(pipeline: Pipeline) -> None:
    prompt = ChatPromptTemplate.from_template("Answer: {question}")
    chain = prompt | chat_model("Paris.") | StrOutputParser()

    result = chain.invoke({"question": "capital?"}, config={"callbacks": [LlmMetricsTracer()]})

    assert result == "Paris."
    assert len(pipeline.traces()) == 1, "one chain invocation is one trace"

    names = [e["name"] for e in pipeline.observations()]
    assert "GenericFakeChatModel" in names
    assert pipeline.of_type(ObservationType.GENERATION), "the model call must be a generation"

    # Every observation except the root hangs off another one.
    roots = [e for e in pipeline.observations() if "parent_id" not in e]
    assert len(roots) == 1, f"expected a single root, got {[r['name'] for r in roots]}"


def test_the_tree_matches_langchains_run_hierarchy(pipeline: Pipeline) -> None:
    """Children must hang off their real parent, not off whatever ran last."""

    def outer(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    chain = RunnableLambda(outer, name="outer") | RunnableLambda(
        lambda payload: str(payload), name="inner"
    )

    chain.invoke({"x": 1}, config={"callbacks": [LlmMetricsTracer()]})

    shape = pipeline.tree()
    assert len(shape["<root>"]) == 1, "exactly one root"
    root = shape["<root>"][0]
    assert set(shape.get(root, [])) == {"outer", "inner"}, (
        f"outer and inner should be siblings under {root}, got {shape}"
    )


def test_tools_become_tool_observations(pipeline: Pipeline) -> None:
    @tool
    def lookup(city: str) -> str:
        """Look a city up."""
        return f"{city} is nice"

    result = lookup.invoke({"city": "Paris"}, config={"callbacks": [LlmMetricsTracer()]})

    assert result == "Paris is nice"
    observation = pipeline.one(ObservationType.TOOL)
    assert observation["name"] == "lookup"
    assert observation["output"] == "Paris is nice"


def test_retrievers_become_retrieval_observations(pipeline: Pipeline) -> None:
    from langchain_core.retrievers import BaseRetriever

    class Fake(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
            return [Document(page_content=f"about {query}"), Document(page_content="second")]

    docs = Fake().invoke("paris", config={"callbacks": [LlmMetricsTracer()]})

    assert len(docs) == 2
    observation = pipeline.one(ObservationType.RETRIEVAL)
    assert observation["metadata"]["documents"] == 2
    assert observation["output"][0]["page_content"] == "about paris"


def test_a_langchain_error_is_recorded_and_re_raised(pipeline: Pipeline) -> None:
    def explode(_payload: Any) -> Any:
        raise ValueError("chain broke")

    with pytest.raises(ValueError, match="chain broke"):
        RunnableLambda(explode, name="boom").invoke(
            {"x": 1}, config={"callbacks": [LlmMetricsTracer()]}
        )

    failed = [e for e in pipeline.observations() if e["status"] == ObservationStatus.ERROR]
    assert failed, "the failure was not recorded"
    assert "ValueError: chain broke" in failed[0]["status_message"]


def test_a_chain_nests_under_an_enclosing_observe_trace(pipeline: Pipeline) -> None:
    tracer = LlmMetricsTracer()
    chain = ChatPromptTemplate.from_template("{q}") | chat_model("Paris.")

    @observe(name="handler")
    def handle() -> Any:
        return chain.invoke({"q": "capital?"}, config={"callbacks": [tracer]})

    handle()

    assert len(pipeline.traces()) == 1, "the chain must join the enclosing trace, not start one"
    handler = next(e for e in pipeline.observations() if e["name"] == "handler")
    roots = [e for e in pipeline.observations() if "parent_id" not in e]
    assert roots == [handler], "@observe should be the only root"

    generation = pipeline.one(ObservationType.GENERATION)
    assert generation["trace_id"] == handler["trace_id"]


def test_async_runs_are_traced(pipeline: Pipeline) -> None:
    chain = ChatPromptTemplate.from_template("{q}") | chat_model("Paris.")

    async def main() -> Any:
        return await chain.ainvoke({"q": "capital?"}, config={"callbacks": [LlmMetricsTracer()]})

    result = asyncio.run(main())

    assert result.content == "Paris."
    assert pipeline.of_type(ObservationType.GENERATION)


def test_streaming_records_time_to_first_token(pipeline: Pipeline) -> None:
    """Total latency hides the metric that actually matters for a stream."""
    llm = chat_model("Paris is the capital.")
    chunks = list(llm.stream("capital?", config={"callbacks": [LlmMetricsTracer()]}))

    assert chunks, "the caller must still get the stream"
    generation = pipeline.one(ObservationType.GENERATION)
    ttft = generation["metadata"]["time_to_first_token_ms"]
    assert isinstance(ttft, float)
    assert 0 <= ttft <= generation["latency_ms"] + 1


def test_one_tracer_serves_many_invocations(pipeline: Pipeline) -> None:
    tracer = LlmMetricsTracer()
    for _ in range(3):
        chat_model("hi").invoke("q", config={"callbacks": [tracer]})

    assert len(pipeline.of_type(ObservationType.GENERATION)) == 3
    assert len(pipeline.traces()) == 3, "separate invocations are separate traces"
    assert tracer.open_runs == 0, "runs must be released when they end"


def test_trace_name_and_user_id_are_applied(pipeline: Pipeline) -> None:
    tracer = LlmMetricsTracer(trace_name="support-bot", user_id="u-42", metadata={"env": "test"})
    chat_model("hi").invoke("q", config={"callbacks": [tracer]})

    trace = pipeline.traces()[0]
    assert trace["name"] == "support-bot"
    assert trace["user_id"] == "u-42"
    assert pipeline.one(ObservationType.GENERATION)["metadata"]["env"] == "test"


def test_no_cost_is_ever_computed(pipeline: Pipeline) -> None:
    """Rule 3."""
    import json

    llm = chat_model("hi", usage={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6})
    llm.invoke("q", config={"callbacks": [LlmMetricsTracer()]})

    assert "cost" not in json.dumps(pipeline.flush()).lower()


# --------------------------------------------------------------------------- #
# Bounded memory and orphan runs
# --------------------------------------------------------------------------- #


def test_runs_are_released_when_they_end(pipeline: Pipeline) -> None:
    tracer = LlmMetricsTracer()
    chain = ChatPromptTemplate.from_template("{q}") | chat_model("hi") | StrOutputParser()
    chain.invoke({"q": "x"}, config={"callbacks": [tracer]})

    assert tracer.open_runs == 0, "a completed tree must leave nothing behind"


def test_runs_that_never_end_are_evicted_rather_than_pinned(pipeline: Pipeline) -> None:
    """Rule 4. A crash between callbacks must not leak memory forever."""
    tracer = LlmMetricsTracer(max_runs=10)

    for _ in range(100):
        tracer.on_chain_start({"name": "abandoned"}, {"x": 1}, run_id=uuid4())

    assert tracer.open_runs == 10, "the map grew past its cap"
    assert tracer.dropped_runs == 90


def test_an_end_for_an_unknown_run_is_ignored(pipeline: Pipeline) -> None:
    tracer = LlmMetricsTracer()
    tracer.on_chain_end({"out": 1}, run_id=uuid4())  # never started
    tracer.on_llm_error(RuntimeError("x"), run_id=uuid4())

    assert pipeline.observations() == []


def test_an_unknown_parent_roots_the_run(pipeline: Pipeline) -> None:
    """A handler attached partway down a tree roots itself instead of guessing."""
    tracer = LlmMetricsTracer()
    run_id = uuid4()

    tracer.on_chain_start({"name": "orphan"}, {"x": 1}, run_id=run_id, parent_run_id=uuid4())
    tracer.on_chain_end({"out": 1}, run_id=run_id)

    observation = pipeline.one(ObservationType.SPAN)
    assert observation["name"] == "orphan"
    assert "parent_id" not in observation


def test_the_handler_is_safe_to_share_between_threads(pipeline: Pipeline) -> None:
    tracer = LlmMetricsTracer()
    errors: list[BaseException] = []

    def run(i: int) -> None:
        try:
            chat_model(f"reply-{i}").invoke(f"q-{i}", config={"callbacks": [tracer]})
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors, f"concurrent use raised: {errors[0]!r}"
    assert len(pipeline.of_type(ObservationType.GENERATION)) == 8
    assert tracer.open_runs == 0


# --------------------------------------------------------------------------- #
# Rule 2 — the tracer must never break the chain
# --------------------------------------------------------------------------- #


def test_a_hostile_payload_does_not_break_the_run(pipeline: Pipeline) -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    tracer = LlmMetricsTracer()
    run_id = uuid4()

    tracer.on_chain_start({"name": "chain"}, {"bad": Hostile()}, run_id=run_id)
    tracer.on_chain_end({"also_bad": Hostile()}, run_id=run_id)

    observation = pipeline.one(ObservationType.SPAN)
    assert observation["status"] == ObservationStatus.OK


def test_hooks_swallow_their_own_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangChain suppresses handler errors by default, but that is a setting a
    user can flip — the guard has to be ours."""
    from llm_metrics.integrations import langchain as integration

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("tracer is broken")

    monkeypatch.setattr(integration, "open_span", explode)

    tracer = LlmMetricsTracer()
    tracer.raise_error = True

    # Must not raise despite open_span being broken and raise_error being set.
    tracer.on_chain_start({"name": "chain"}, {"x": 1}, run_id=uuid4())


def test_a_chain_still_runs_when_the_sdk_is_disabled(pipeline: Pipeline) -> None:
    _runtime.configure(sink=pipeline.buffer, enabled=False)
    chain = ChatPromptTemplate.from_template("{q}") | chat_model("Paris.")

    result = chain.invoke({"q": "capital?"}, config={"callbacks": [LlmMetricsTracer()]})

    assert result.content == "Paris."
    assert pipeline.flush() == []


def test_a_broken_sink_does_not_break_the_chain() -> None:
    def explode(_payload: object, _deadline: object = None) -> None:
        raise RuntimeError("sink is down")

    broken = EventBuffer(explode, flush_at=1, flush_interval=300.0, start=False)
    _runtime.configure(sink=broken, enabled=True)
    try:
        chain = ChatPromptTemplate.from_template("{q}") | chat_model("Paris.")
        result = chain.invoke({"q": "capital?"}, config={"callbacks": [LlmMetricsTracer()]})
        assert result.content == "Paris."
        broken.flush_once()  # must not propagate
    finally:
        _runtime.shutdown(timeout=1.0)


def test_user_id_reaches_an_enclosing_trace(pipeline: Pipeline) -> None:
    """A caller who passed user_id meant it, even if @observe owns the trace."""
    tracer = LlmMetricsTracer(trace_name="support-bot", user_id="u-42")

    @observe(name="handler")
    def handle() -> Any:
        return chat_model("hi").invoke("q", config={"callbacks": [tracer]})

    handle()

    trace = pipeline.traces()[0]
    assert trace["user_id"] == "u-42"
    assert trace["name"] == "handler", "renaming someone else's trace would be presumptuous"


def test_an_existing_user_id_is_not_overwritten(pipeline: Pipeline) -> None:
    from llm_metrics import context
    from llm_metrics.models import Trace

    tracer = LlmMetricsTracer(user_id="from-tracer")
    trace = Trace(name="request", user_id="from-caller")

    with context.use_trace(trace):
        chat_model("hi").invoke("q", config={"callbacks": [tracer]})

    assert trace.user_id == "from-caller"
