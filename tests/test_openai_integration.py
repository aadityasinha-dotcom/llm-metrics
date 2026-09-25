"""OpenAI integration.

Two layers. Most tests drive a fake client, because the integration is
duck-typed and a fake keeps the failure modes (odd shapes, raising calls,
abandoned streams) easy to construct. A smaller set drives the *real* ``openai``
client over a mocked HTTP transport, because the whole point of duck typing is
an assumption about what the real client looks like, and an assumption is worth
checking.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator, Sequence
from typing import Any

import httpx
import pytest

from llm_metrics import _runtime
from llm_metrics.buffer import EventBuffer
from llm_metrics.decorator import observe
from llm_metrics.integrations.openai import wrap_openai
from llm_metrics.models import ObservationStatus, ObservationType

openai = pytest.importorskip("openai")


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class Collector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, payload: list[dict[str, Any]], deadline: object = None) -> None:
        self.events.extend(payload)

    @property
    def generations(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == ObservationType.GENERATION]


class Pipeline:
    def __init__(self, sink: Collector, buffer: EventBuffer) -> None:
        self.sink = sink
        self.buffer = buffer

    def flush(self) -> Collector:
        self.buffer.flush_once()
        return self.sink

    def one(self) -> dict[str, Any]:
        generations = self.flush().generations
        assert len(generations) == 1, f"expected one generation, got {len(generations)}"
        return generations[0]


@pytest.fixture
def pipeline() -> Iterator[Pipeline]:
    sink = Collector()
    buffer = EventBuffer(sink, flush_at=10_000, flush_interval=300.0, start=False)
    _runtime.configure(sink=buffer, enabled=True)
    try:
        yield Pipeline(sink, buffer)
    finally:
        _runtime.shutdown(timeout=1.0)


# --- fake client ----------------------------------------------------------- #


class FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content

    def model_dump(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str = "Paris.", model: str = "gpt-4o-2024-08-06") -> None:
        self.model = model
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage(11, 3)


class FakeCompletions:
    def __init__(self, result: Any = None, error: BaseException | None = None) -> None:
        self.result = result if result is not None else FakeResponse()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeAsyncCompletions(FakeCompletions):
    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def fake_client(completions: FakeCompletions | None = None) -> Any:
    completions = completions or FakeCompletions()
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #


def test_a_completion_becomes_a_generation(pipeline: Pipeline) -> None:
    client = wrap_openai(fake_client())

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "capital of France?"}],
        temperature=0.2,
    )

    assert isinstance(response, FakeResponse), "the caller must get the real response back"

    obs = pipeline.one()
    assert obs["name"] == "openai.chat.completions"
    assert obs["model"] == "gpt-4o-2024-08-06", "the model actually used, not the one requested"
    assert obs["prompt_tokens"] == 11
    assert obs["completion_tokens"] == 3
    assert obs["input"] == [{"role": "user", "content": "capital of France?"}]
    assert obs["output"] == {"role": "assistant", "content": "Paris."}
    assert obs["metadata"]["temperature"] == 0.2
    assert isinstance(obs["latency_ms"], float)


def test_no_cost_is_ever_computed(pipeline: Pipeline) -> None:
    """Rule 3: token counts go up, pricing stays on the server."""
    client = wrap_openai(fake_client())
    client.chat.completions.create(model="gpt-4o", messages=[])

    assert "cost" not in json.dumps(pipeline.one()).lower()


def test_tool_schemas_are_reduced_to_names(pipeline: Pipeline) -> None:
    """Full JSON schemas are bulky and static; the names carry the meaning."""
    client = wrap_openai(fake_client())
    client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        tools=[
            {"type": "function", "function": {"name": "search", "parameters": {"huge": "schema"}}},
            {"type": "function", "function": {"name": "lookup", "parameters": {}}},
        ],
    )

    assert pipeline.one()["metadata"]["tools"] == ["search", "lookup"]


def test_an_api_error_is_recorded_and_re_raised(pipeline: Pipeline) -> None:
    boom = RuntimeError("rate limited")
    client = wrap_openai(fake_client(FakeCompletions(error=boom)))

    with pytest.raises(RuntimeError) as caught:
        client.chat.completions.create(model="gpt-4o", messages=[])

    assert caught.value is boom, "the caller must get the original exception"
    obs = pipeline.one()
    assert obs["status"] == ObservationStatus.ERROR
    assert obs["status_message"] == "RuntimeError: rate limited"


def test_completions_nest_under_an_enclosing_trace(pipeline: Pipeline) -> None:
    client = wrap_openai(fake_client())

    @observe(name="answer")
    def answer() -> None:
        client.chat.completions.create(model="gpt-4o", messages=[])

    answer()

    sink = pipeline.flush()
    parent = next(e for e in sink.events if e["name"] == "answer" and e["type"] != "trace")
    assert sink.generations[0]["parent_id"] == parent["id"]
    assert len([e for e in sink.events if e["type"] == "trace"]) == 1


def test_a_bare_completion_starts_its_own_trace(pipeline: Pipeline) -> None:
    client = wrap_openai(fake_client())
    client.chat.completions.create(model="gpt-4o", messages=[])

    sink = pipeline.flush()
    assert len([e for e in sink.events if e["type"] == "trace"]) == 1
    assert "parent_id" not in sink.generations[0]


def test_async_clients_are_wrapped(pipeline: Pipeline) -> None:
    client = wrap_openai(fake_client(FakeAsyncCompletions()))

    async def main() -> Any:
        return await client.chat.completions.create(model="gpt-4o", messages=[])

    response = asyncio.run(main())

    assert isinstance(response, FakeResponse)
    assert pipeline.one()["prompt_tokens"] == 11


def test_wrapping_twice_does_not_double_record(pipeline: Pipeline) -> None:
    client = wrap_openai(fake_client())
    wrap_openai(client)
    wrap_openai(client)

    client.chat.completions.create(model="gpt-4o", messages=[])

    assert len(pipeline.flush().generations) == 1


def test_wrapping_something_unrecognisable_is_a_no_op() -> None:
    """A client shape we do not know comes back uninstrumented, not broken."""
    stranger = object()
    assert wrap_openai(stranger) is stranger


def test_a_response_without_usage_records_no_tokens(pipeline: Pipeline) -> None:
    response = FakeResponse()
    del response.usage
    client = wrap_openai(fake_client(FakeCompletions(result=response)))

    client.chat.completions.create(model="gpt-4o", messages=[])

    obs = pipeline.one()
    assert "prompt_tokens" not in obs
    assert obs["output"] == {"role": "assistant", "content": "Paris."}


def test_a_hostile_response_does_not_break_the_call(pipeline: Pipeline) -> None:
    class Hostile:
        @property
        def model(self) -> str:
            raise RuntimeError("no model for you")

    hostile = Hostile()
    client = wrap_openai(fake_client(FakeCompletions(result=hostile)))

    assert client.chat.completions.create(model="gpt-4o", messages=[]) is hostile
    assert pipeline.one()["status"] == ObservationStatus.OK


# --------------------------------------------------------------------------- #
# Streaming — against the fake
# --------------------------------------------------------------------------- #


class FakeChunk:
    def __init__(self, content: str | None, usage: FakeUsage | None = None) -> None:
        self.model = "gpt-4o-2024-08-06"
        self.usage = usage
        delta = type("Delta", (), {"content": content})()
        self.choices = [type("Choice", (), {"delta": delta})()]


class FakeStream:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[FakeChunk]:
        yield from self.chunks

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


def streaming_client(chunks: Sequence[FakeChunk] | None = None) -> Any:
    chunks = chunks if chunks is not None else [FakeChunk(t) for t in ("Pa", "ris", ".")]
    return wrap_openai(fake_client(FakeCompletions(result=FakeStream(list(chunks)))))


def test_a_stream_is_traced_once_it_is_consumed(pipeline: Pipeline) -> None:
    client = streaming_client()

    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    assert pipeline.flush().generations == [], "nothing recorded before consumption"

    assert [c.choices[0].delta.content for c in stream] == ["Pa", "ris", "."]

    obs = pipeline.one()
    assert obs["output"] == "Paris.", "deltas should be reassembled"
    assert obs["metadata"]["stream_chunks"] == 3
    assert obs["model"] == "gpt-4o-2024-08-06"


def test_stream_usage_is_recorded_when_the_caller_asks_for_it(pipeline: Pipeline) -> None:
    chunks = [FakeChunk("hi"), FakeChunk(None, usage=FakeUsage(7, 2))]
    client = streaming_client(chunks)

    list(
        client.chat.completions.create(
            model="gpt-4o", messages=[], stream=True, stream_options={"include_usage": True}
        )
    )

    obs = pipeline.one()
    assert obs["prompt_tokens"] == 7
    assert obs["completion_tokens"] == 2


def test_request_parameters_are_never_injected(pipeline: Pipeline) -> None:
    """Adding stream_options behind the caller's back appends a chunk with an
    empty choices list, and plenty of real code indexes choices[0] unguarded."""
    completions = FakeCompletions(result=FakeStream([FakeChunk("x")]))
    client = wrap_openai(fake_client(completions))

    list(client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    assert "stream_options" not in completions.calls[0]


def test_an_abandoned_stream_still_records(pipeline: Pipeline) -> None:
    client = streaming_client([FakeChunk(str(i)) for i in range(100)])

    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    for i, _chunk in enumerate(stream):
        if i == 2:
            break
    stream.close()

    obs = pipeline.one()
    assert obs["status"] == ObservationStatus.OK
    assert obs["output"] == "012"


def test_a_stream_used_as_a_context_manager_records_on_exit(pipeline: Pipeline) -> None:
    client = streaming_client()

    with client.chat.completions.create(model="gpt-4o", messages=[], stream=True) as stream:
        assert next(iter(stream)).choices[0].delta.content == "Pa"

    assert pipeline.one()["output"] == "Pa"


def test_a_stream_that_raises_mid_flight_is_recorded(pipeline: Pipeline) -> None:
    class Exploding:
        def __iter__(self) -> Iterator[FakeChunk]:
            yield FakeChunk("ok")
            raise RuntimeError("connection reset")

    client = wrap_openai(fake_client(FakeCompletions(result=Exploding())))

    with pytest.raises(RuntimeError, match="connection reset"):
        list(client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    assert pipeline.one()["status"] == ObservationStatus.ERROR


def test_the_stream_proxy_forwards_unknown_attributes(pipeline: Pipeline) -> None:
    """Callers reach for .response and .close() on what they get back."""
    client = streaming_client()
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    assert hasattr(stream, "chunks"), "proxy did not forward through to the stream"
    stream.close()
    assert stream._stream.closed is True


def test_the_span_is_closed_exactly_once(pipeline: Pipeline) -> None:
    """Draining and then closing must not emit two observations."""
    client = streaming_client()
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    list(stream)
    stream.close()
    stream.close()

    assert len(pipeline.flush().generations) == 1


# --------------------------------------------------------------------------- #
# Against the real openai client, over a mocked transport
# --------------------------------------------------------------------------- #


CHAT_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Paris."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
}


def real_client(handler: Any, *, is_async: bool = False) -> Any:
    kind = openai.AsyncOpenAI if is_async else openai.OpenAI
    http = (httpx.AsyncClient if is_async else httpx.Client)(
        transport=(httpx.MockTransport(handler))
    )
    return wrap_openai(kind(api_key="sk-test", http_client=http))


def test_real_client_chat_completion(pipeline: Pipeline) -> None:
    client = real_client(lambda request: httpx.Response(200, json=CHAT_BODY))

    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "capital of France?"}]
    )

    assert response.choices[0].message.content == "Paris."

    obs = pipeline.one()
    assert obs["model"] == "gpt-4o-2024-08-06"
    assert obs["prompt_tokens"] == 11
    assert obs["completion_tokens"] == 3
    assert obs["output"]["content"] == "Paris."
    assert obs["input"] == [{"role": "user", "content": "capital of France?"}]


def test_real_async_client_is_detected_as_async(pipeline: Pipeline) -> None:
    """openai wraps `create` with functools.wraps, so a plain
    iscoroutinefunction check on the bound method reports False and would pick
    the sync path for AsyncOpenAI."""
    client = real_client(lambda request: httpx.Response(200, json=CHAT_BODY), is_async=True)

    async def main() -> Any:
        return await client.chat.completions.create(model="gpt-4o", messages=[])

    response = asyncio.run(main())

    assert response.choices[0].message.content == "Paris."
    assert pipeline.one()["prompt_tokens"] == 11


def test_real_client_streaming(pipeline: Pipeline) -> None:
    def sse(request: httpx.Request) -> httpx.Response:
        frames = [
            {
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            for piece in ("Pa", "ris", ".")
        ]
        body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = real_client(sse)
    stream = client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
    pieces = [chunk.choices[0].delta.content for chunk in stream]

    assert pieces == ["Pa", "ris", "."]
    obs = pipeline.one()
    assert obs["output"] == "Paris."
    assert obs["model"] == "gpt-4o-mini"


def test_real_client_api_error_propagates(pipeline: Pipeline) -> None:
    client = real_client(
        lambda request: httpx.Response(429, json={"error": {"message": "slow down"}})
    )

    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(model="gpt-4o", messages=[])

    obs = pipeline.one()
    assert obs["status"] == ObservationStatus.ERROR
    assert "RateLimitError" in obs["status_message"]


def test_next_works_without_calling_iter_first(pipeline: Pipeline) -> None:
    """openai's Stream supports next() directly, so the proxy must too.

    Without a lazily-initialised iterator, __getattr__ forwards the lookup of
    the missing attribute to the wrapped stream and raises a confusing
    AttributeError from inside the SDK.
    """
    client = streaming_client()
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    assert next(stream).choices[0].delta.content == "Pa"
    stream.close()

    assert pipeline.one()["output"] == "Pa"


def test_nulls_are_pruned_from_provider_objects(pipeline: Pipeline) -> None:
    """A dumped chat message is mostly nulls; they cost payload and read as noise."""
    client = real_client(lambda request: httpx.Response(200, json=CHAT_BODY))
    client.chat.completions.create(model="gpt-4o", messages=[])

    output = pipeline.one()["output"]
    assert output == {"role": "assistant", "content": "Paris."}
    assert "refusal" not in output


def test_caller_supplied_arguments_are_never_rewritten(pipeline: Pipeline) -> None:
    """Pruning applies to dumps, not to what the caller actually passed."""
    client = wrap_openai(fake_client())
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi", "name": None}]
    )

    assert pipeline.one()["input"] == [{"role": "user", "content": "hi", "name": None}]


# --------------------------------------------------------------------------- #
# Beyond the basics: token detail, how it ended, provider identity
# --------------------------------------------------------------------------- #


class DetailedUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 1000
        self.completion_tokens = 300
        self.prompt_tokens_details = {"cached_tokens": 900, "audio_tokens": 0}
        self.completion_tokens_details = {
            "reasoning_tokens": 250,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 7,
            "rejected_prediction_tokens": 0,
        }


class RichMessage(FakeMessage):
    def __init__(
        self, content: str | None, *, refusal: str | None = None, tools: Sequence[str] = ()
    ):
        super().__init__(content or "")
        self.refusal = refusal
        self.tool_calls = [
            type("Call", (), {"function": type("Fn", (), {"name": n})()})() for n in tools
        ]

    def model_dump(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "refusal": self.refusal,
            "tool_calls": [{"function": {"name": c.function.name}} for c in self.tool_calls]
            or None,
        }


class RichResponse:
    def __init__(
        self,
        *,
        finish_reason: str = "stop",
        refusal: str | None = None,
        tools: Sequence[str] = (),
    ) -> None:
        self.id = "chatcmpl-abc"
        self.model = "gpt-4o-2024-08-06"
        self.system_fingerprint = "fp_123"
        self.service_tier = "default"
        self.usage = DetailedUsage()
        choice = type("Choice", (), {})()
        choice.finish_reason = finish_reason
        choice.message = RichMessage(
            "Paris." if not refusal else None, refusal=refusal, tools=tools
        )
        self.choices = [choice]


def test_cached_and_reasoning_tokens_are_first_class(pipeline: Pipeline) -> None:
    """They are priced differently from the totals they sit inside, so a
    dashboard needs them as columns, not buried in a details blob."""
    client = wrap_openai(fake_client(FakeCompletions(result=RichResponse())))
    client.chat.completions.create(model="o3", messages=[])

    obs = pipeline.one()
    assert obs["prompt_tokens"] == 1000
    assert obs["completion_tokens"] == 300
    assert obs["cached_tokens"] == 900
    assert obs["reasoning_tokens"] == 250
    assert obs["metadata"]["usage"] == {"accepted_prediction_tokens": 7}, "zeros are noise"


def test_finish_reason_and_provider_identity_are_recorded(pipeline: Pipeline) -> None:
    client = wrap_openai(fake_client(FakeCompletions(result=RichResponse(finish_reason="length"))))
    client.chat.completions.create(model="gpt-4o", messages=[])

    meta = pipeline.one()["metadata"]
    assert meta["finish_reason"] == "length"
    assert meta["response_id"] == "chatcmpl-abc"
    assert meta["system_fingerprint"] == "fp_123"
    assert meta["service_tier"] == "default"


def test_a_refusal_is_flagged(pipeline: Pipeline) -> None:
    client = wrap_openai(fake_client(FakeCompletions(result=RichResponse(refusal="I can't."))))
    client.chat.completions.create(model="gpt-4o", messages=[])

    obs = pipeline.one()
    assert obs["metadata"]["refusal"] is True
    assert obs["output"]["refusal"] == "I can't."


def test_tools_the_model_actually_called_are_recorded(pipeline: Pipeline) -> None:
    """``tools`` is what was offered; ``tool_calls`` is what was chosen. The
    gap between the two is tool-selection accuracy."""
    response = RichResponse(finish_reason="tool_calls", tools=["search", "search"])
    client = wrap_openai(fake_client(FakeCompletions(result=response)))
    client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        tools=[{"function": {"name": "search"}}, {"function": {"name": "lookup"}}],
    )

    meta = pipeline.one()["metadata"]
    assert meta["tools"] == ["search", "lookup"]
    assert meta["tool_calls"] == ["search", "search"]
    assert meta["finish_reason"] == "tool_calls"


def test_a_plain_response_carries_no_empty_facts(pipeline: Pipeline) -> None:
    client = wrap_openai(fake_client())
    client.chat.completions.create(model="gpt-4o", messages=[])

    meta = pipeline.one()["metadata"]
    for key in ("finish_reason", "refusal", "tool_calls", "system_fingerprint", "usage"):
        assert key not in meta


def test_responses_api_incomplete_status_is_a_finish_reason(pipeline: Pipeline) -> None:
    response = {
        "id": "resp_1",
        "model": "gpt-4o",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output_text": "partial",
        "output": [{"type": "function_call", "name": "search"}],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 2,
            "input_tokens_details": {"cached_tokens": 4},
        },
    }
    responses = FakeCompletions(result=response)
    client = type("Client", (), {"responses": responses})()
    wrap_openai(client)
    client.responses.create(model="gpt-4o", input="hi")

    obs = pipeline.one()
    assert obs["name"] == "openai.responses"
    assert obs["metadata"]["finish_reason"] == "max_output_tokens"
    assert obs["metadata"]["tool_calls"] == ["search"]
    assert obs["cached_tokens"] == 4
    assert obs["output"] == "partial"


# --------------------------------------------------------------------------- #
# Streaming timing: time to first token, throughput, abandonment
# --------------------------------------------------------------------------- #


class TimedChunk(FakeChunk):
    def __init__(
        self,
        content: str | None,
        *,
        usage: FakeUsage | None = None,
        finish_reason: str | None = None,
        tool: tuple[int, str] | None = None,
    ) -> None:
        super().__init__(content, usage=usage)
        self.id = "chatcmpl-stream"
        self.system_fingerprint = "fp_stream"
        self.choices[0].finish_reason = finish_reason
        if tool is not None:
            index, name = tool
            call = type(
                "Call", (), {"index": index, "function": type("Fn", (), {"name": name})()}
            )()
            self.choices[0].delta.tool_calls = [call]


def test_time_to_first_token_is_measured_from_the_first_content(pipeline: Pipeline) -> None:
    """The first chunk carries only a role; the clock must not stop on it."""

    class SlowStart(FakeStream):
        def __iter__(self) -> Iterator[FakeChunk]:
            yield TimedChunk("")  # role-only preamble
            time.sleep(0.05)
            yield TimedChunk("Pa")
            yield TimedChunk("ris", usage=FakeUsage(7, 2), finish_reason="stop")

    client = wrap_openai(fake_client(FakeCompletions(result=SlowStart([]))))
    list(client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    meta = pipeline.one()["metadata"]
    assert meta["time_to_first_token_ms"] >= 45
    assert meta["stream_completed"] is True
    assert meta["stream_chunks"] == 3
    assert meta["finish_reason"] == "stop"
    assert meta["response_id"] == "chatcmpl-stream"
    assert meta["system_fingerprint"] == "fp_stream"
    assert meta["output_tokens_per_second"] > 0


def test_throughput_needs_usage_and_is_absent_without_it(pipeline: Pipeline) -> None:
    client = streaming_client([TimedChunk("a"), TimedChunk("b")])
    list(client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    meta = pipeline.one()["metadata"]
    assert "time_to_first_token_ms" in meta
    assert "output_tokens_per_second" not in meta


def test_an_abandoned_stream_is_marked_incomplete(pipeline: Pipeline) -> None:
    client = streaming_client([TimedChunk(str(i)) for i in range(10)])
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    next(stream)
    stream.close()

    assert pipeline.one()["metadata"]["stream_completed"] is False


def test_streamed_tool_calls_are_named_once_per_index(pipeline: Pipeline) -> None:
    chunks = [
        TimedChunk(None, tool=(0, "search")),
        TimedChunk(None, tool=(0, "")),  # argument fragments carry no name
        TimedChunk(None, tool=(1, "lookup")),
        TimedChunk(None, finish_reason="tool_calls"),
    ]
    client = streaming_client(chunks)
    list(client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    meta = pipeline.one()["metadata"]
    assert meta["tool_calls"] == ["search", "lookup"]
    assert meta["finish_reason"] == "tool_calls"
    assert "time_to_first_token_ms" in meta, "a tool call is the first token too"


def test_responses_api_stream_reads_the_completed_event(pipeline: Pipeline) -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "Pa"},
        {"type": "response.output_text.delta", "delta": "ris"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_9",
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 9,
                    "output_tokens": 2,
                    "output_tokens_details": {"reasoning_tokens": 1},
                },
            },
        },
    ]
    responses = FakeCompletions(result=FakeStream(events))  # type: ignore[arg-type]
    client = type("Client", (), {"responses": responses})()
    wrap_openai(client)
    list(client.responses.create(model="gpt-4o-mini", input="hi", stream=True))

    obs = pipeline.one()
    assert obs["output"] == "Paris"
    assert obs["model"] == "gpt-4o-mini"
    assert obs["prompt_tokens"] == 9
    assert obs["reasoning_tokens"] == 1
    assert obs["metadata"]["finish_reason"] == "stop"


# --------------------------------------------------------------------------- #
# Headers: request id, rate-limit headroom, attempts — real client only
# --------------------------------------------------------------------------- #

RATE_HEADERS = {
    "x-request-id": "req_42",
    "openai-processing-ms": "123",
    "x-ratelimit-limit-requests": "10000",
    "x-ratelimit-limit-tokens": "2000000",
    "x-ratelimit-remaining-requests": "9998",
    "x-ratelimit-remaining-tokens": "1999500",
    "x-ratelimit-reset-requests": "6ms",
    "x-ratelimit-reset-tokens": "1s",
}


def test_real_client_records_headers(pipeline: Pipeline) -> None:
    client = real_client(lambda request: httpx.Response(200, json=CHAT_BODY, headers=RATE_HEADERS))
    client.chat.completions.create(model="gpt-4o", messages=[])

    meta = pipeline.one()["metadata"]
    assert meta["request_id"] == "req_42"
    assert meta["upstream_processing_ms"] == 123
    assert meta["http_status"] == 200
    assert meta["http_attempts"] == 1
    assert meta["rate_limit"] == {
        "limit_requests": 10000,
        "limit_tokens": 2000000,
        "remaining_requests": 9998,
        "remaining_tokens": 1999500,
        "reset_requests": "6ms",
        "reset_tokens": "1s",
    }


def test_real_client_counts_internal_retries(pipeline: Pipeline) -> None:
    """The openai client retries a 429 by itself. The caller sees one slow
    call; the observation says it was two attempts."""
    seen = 0

    def flaky(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        if seen == 1:
            return httpx.Response(
                429, json={"error": {"message": "slow down"}}, headers={"retry-after-ms": "1"}
            )
        return httpx.Response(200, json=CHAT_BODY, headers={"x-request-id": "req_2nd"})

    http = httpx.Client(transport=httpx.MockTransport(flaky))
    client = wrap_openai(openai.OpenAI(api_key="sk-test", http_client=http, max_retries=1))
    client.chat.completions.create(model="gpt-4o", messages=[])

    meta = pipeline.one()["metadata"]
    assert meta["http_attempts"] == 2
    assert meta["http_status"] == 200
    assert meta["request_id"] == "req_2nd", "the last attempt's headers win"


def test_real_client_error_observation_carries_the_rate_limit_headers(
    pipeline: Pipeline,
) -> None:
    """When you are throttled, the headroom at the moment it happened is the
    most useful thing the observation can say."""
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                429,
                json={"error": {"message": "slow down"}},
                headers={"x-ratelimit-remaining-requests": "0", "x-request-id": "req_429"},
            )
        )
    )
    client = wrap_openai(openai.OpenAI(api_key="sk-test", http_client=http, max_retries=0))

    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(model="gpt-4o", messages=[])

    obs = pipeline.one()
    assert obs["status"] == ObservationStatus.ERROR
    assert obs["metadata"]["http_status"] == 429
    assert obs["metadata"]["rate_limit"] == {"remaining_requests": 0}
    assert obs["metadata"]["request_id"] == "req_429"


def test_real_async_client_records_headers(pipeline: Pipeline) -> None:
    client = real_client(
        lambda request: httpx.Response(200, json=CHAT_BODY, headers=RATE_HEADERS), is_async=True
    )

    async def main() -> Any:
        return await client.chat.completions.create(model="gpt-4o", messages=[])

    asyncio.run(main())

    meta = pipeline.one()["metadata"]
    assert meta["request_id"] == "req_42"
    assert meta["http_attempts"] == 1


def test_real_client_streaming_records_headers_and_timing(pipeline: Pipeline) -> None:
    def sse(request: httpx.Request) -> httpx.Response:
        frames = [
            {
                "id": "chatcmpl-s",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            for piece in ("Pa", "ris")
        ]
        frames.append(
            {
                "id": "chatcmpl-s",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        frames.append(
            {
                "id": "chatcmpl-s",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
        )
        body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream", "x-request-id": "req_stream"},
        )

    client = real_client(sse)
    list(
        client.chat.completions.create(
            model="gpt-4o-mini", messages=[], stream=True, stream_options={"include_usage": True}
        )
    )

    obs = pipeline.one()
    meta = obs["metadata"]
    assert meta["request_id"] == "req_stream"
    assert meta["finish_reason"] == "stop"
    assert meta["stream_completed"] is True
    assert "time_to_first_token_ms" in meta
    assert obs["completion_tokens"] == 2
    assert meta["output_tokens_per_second"] > 0


def test_wrapping_twice_installs_one_hook(pipeline: Pipeline) -> None:
    client = real_client(lambda request: httpx.Response(200, json=CHAT_BODY))
    wrap_openai(client)
    client.chat.completions.create(model="gpt-4o", messages=[])

    assert pipeline.one()["metadata"]["http_attempts"] == 1, "two hooks would count two"
