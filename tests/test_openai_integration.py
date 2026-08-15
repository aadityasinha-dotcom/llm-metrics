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
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from llmobserve import _runtime
from llmobserve.buffer import EventBuffer
from llmobserve.decorator import observe
from llmobserve.integrations.openai import wrap_openai
from llmobserve.models import ObservationStatus, ObservationType

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


def streaming_client(chunks: list[FakeChunk] | None = None) -> Any:
    chunks = chunks if chunks is not None else [FakeChunk(t) for t in ("Pa", "ris", ".")]
    return wrap_openai(fake_client(FakeCompletions(result=FakeStream(chunks))))


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
