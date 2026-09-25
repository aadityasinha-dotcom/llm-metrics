"""Anthropic integration.

Same two layers as the OpenAI tests: a fake client for the shapes and failure
modes, and the real ``anthropic`` client over a mocked transport to check the
duck-typing assumptions hold. The real client is built on ``httpx2``, not
``httpx``, which is itself one of the assumptions worth checking — the header
hook must attach to either.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest

try:
    import anthropic
    import httpx2
except ImportError:  # pragma: no cover - depends on the extra
    anthropic = None  # type: ignore[assignment]
    httpx2 = None  # type: ignore[assignment]

from llm_metrics import _runtime, observe
from llm_metrics.buffer import EventBuffer
from llm_metrics.integrations.anthropic import wrap_anthropic
from llm_metrics.models import ObservationStatus, ObservationType

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


def usage(**fields: int) -> dict[str, int]:
    return {"input_tokens": 11, "output_tokens": 3, **fields}


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def message(
    *blocks: dict[str, Any], stop_reason: str = "end_turn", **usage_fields: int
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": list(blocks) or [text_block("Paris.")],
        "stop_reason": stop_reason,
        "usage": usage(**usage_fields),
    }


class FakeMessages:
    def __init__(self, result: Any = None, error: BaseException | None = None) -> None:
        self.result = result if result is not None else message()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    def stream(self, **kwargs: Any) -> Any:
        return FakeStreamManager(self.result)


class FakeAsyncMessages(FakeMessages):
    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeStreamManager:
    """Shape of ``MessageStreamManager``: a context manager yielding a stream
    that exposes ``current_message_snapshot``."""

    def __init__(self, final: dict[str, Any]) -> None:
        self.final = final

    def __enter__(self) -> Any:
        return type("Stream", (), {"current_message_snapshot": self.final})()

    def __exit__(self, *exc: Any) -> None:
        pass


def fake_client(messages: FakeMessages | None = None) -> Any:
    return type("Client", (), {"messages": messages or FakeMessages()})()


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #


def test_a_message_becomes_a_generation(pipeline: Pipeline) -> None:
    client = wrap_anthropic(fake_client())

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=64,
        system="Be brief.",
        messages=[{"role": "user", "content": "capital of France?"}],
        temperature=0.1,
    )

    assert response["content"][0]["text"] == "Paris."

    obs = pipeline.one()
    assert obs["name"] == "anthropic.messages"
    assert obs["model"] == "claude-sonnet-5"
    assert obs["prompt_tokens"] == 11
    assert obs["completion_tokens"] == 3
    assert obs["input"] == {
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "capital of France?"}],
    }
    assert obs["output"] == "Paris.", "a single text block reads as plain text"
    assert obs["metadata"]["temperature"] == 0.1
    assert obs["metadata"]["max_tokens"] == 64
    assert obs["metadata"]["finish_reason"] == "stop", "end_turn maps onto the shared vocabulary"
    assert obs["metadata"]["response_id"] == "msg_1"


def test_prompt_tokens_include_the_cached_part(pipeline: Pipeline) -> None:
    """Anthropic's input_tokens excludes cache hits; OpenAI's prompt_tokens
    includes them. Normalise, or every cross-provider sum is wrong."""
    result = message(cache_read_input_tokens=500, cache_creation_input_tokens=40)
    client = wrap_anthropic(fake_client(FakeMessages(result=result)))
    client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])

    obs = pipeline.one()
    assert obs["prompt_tokens"] == 11 + 500 + 40
    assert obs["cached_tokens"] == 500
    assert obs["metadata"]["usage"] == {"cache_creation_input_tokens": 40}


def test_no_cost_is_ever_computed(pipeline: Pipeline) -> None:
    client = wrap_anthropic(fake_client())
    client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])

    assert "cost" not in json.dumps(pipeline.one()).lower()


def test_stop_reasons_map_onto_finish_reasons(pipeline: Pipeline) -> None:
    result = message(stop_reason="max_tokens")
    client = wrap_anthropic(fake_client(FakeMessages(result=result)))
    client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])

    assert pipeline.one()["metadata"]["finish_reason"] == "length"


def test_a_refusal_is_flagged(pipeline: Pipeline) -> None:
    result = message(stop_reason="refusal")
    client = wrap_anthropic(fake_client(FakeMessages(result=result)))
    client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])

    meta = pipeline.one()["metadata"]
    assert meta["finish_reason"] == "content_filter"
    assert meta["refusal"] is True


def test_tool_use_blocks_are_recorded_by_name(pipeline: Pipeline) -> None:
    result = message(
        text_block("Let me check."),
        {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "paris"}},
        stop_reason="tool_use",
    )
    client = wrap_anthropic(fake_client(FakeMessages(result=result)))
    client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1,
        messages=[],
        tools=[{"name": "search", "input_schema": {}}, {"name": "lookup", "input_schema": {}}],
        thinking={"type": "enabled", "budget_tokens": 2048},
    )

    obs = pipeline.one()
    assert obs["metadata"]["tools"] == ["search", "lookup"]
    assert obs["metadata"]["tool_calls"] == ["search"]
    assert obs["metadata"]["finish_reason"] == "tool_calls"
    assert obs["metadata"]["thinking"] == "enabled"
    assert obs["metadata"]["thinking_budget_tokens"] == 2048
    assert obs["output"][1]["name"] == "search", "multi-block content stays structured"


def test_an_api_error_is_recorded_and_re_raised(pipeline: Pipeline) -> None:
    boom = RuntimeError("overloaded")
    client = wrap_anthropic(fake_client(FakeMessages(error=boom)))

    with pytest.raises(RuntimeError) as caught:
        client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])

    assert caught.value is boom
    obs = pipeline.one()
    assert obs["status"] == ObservationStatus.ERROR
    assert obs["status_message"] == "RuntimeError: overloaded"


def test_messages_nest_under_an_enclosing_trace(pipeline: Pipeline) -> None:
    client = wrap_anthropic(fake_client())

    @observe(name="answer")
    def answer() -> None:
        client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])

    answer()

    sink = pipeline.flush()
    parent = next(e for e in sink.events if e["name"] == "answer" and e["type"] != "trace")
    assert sink.generations[0]["parent_id"] == parent["id"]


def test_async_clients_are_wrapped(pipeline: Pipeline) -> None:
    client = wrap_anthropic(fake_client(FakeAsyncMessages()))

    async def main() -> Any:
        return await client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])

    response = asyncio.run(main())

    assert response["content"][0]["text"] == "Paris."
    assert pipeline.one()["prompt_tokens"] == 11


def test_wrapping_twice_does_not_double_record(pipeline: Pipeline) -> None:
    client = wrap_anthropic(fake_client())
    wrap_anthropic(client)
    client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])

    assert len(pipeline.flush().generations) == 1


def test_wrapping_something_unrecognisable_is_a_no_op() -> None:
    stranger = object()
    assert wrap_anthropic(stranger) is stranger


# --------------------------------------------------------------------------- #
# Streaming via messages.create(stream=True)
# --------------------------------------------------------------------------- #


def events(*texts: str, stop_reason: str = "end_turn") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_s",
                "model": "claude-sonnet-5",
                "usage": {"input_tokens": 25, "output_tokens": 1, "cache_read_input_tokens": 20},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    out += [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}
        for t in texts
    ]
    out += [
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]
    return out


class FakeStream:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.closed = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.items

    def close(self) -> None:
        self.closed = True


def test_a_stream_is_reassembled_with_usage_and_timing(pipeline: Pipeline) -> None:
    client = wrap_anthropic(fake_client(FakeMessages(result=FakeStream(events("Pa", "ris", ".")))))

    stream = client.messages.create(model="claude-sonnet-5", max_tokens=8, messages=[], stream=True)
    assert pipeline.flush().generations == [], "nothing recorded before consumption"
    kinds = [e["type"] for e in stream]
    assert kinds[0] == "message_start"
    assert kinds[-1] == "message_stop"

    obs = pipeline.one()
    assert obs["output"] == "Paris."
    assert obs["model"] == "claude-sonnet-5"
    assert obs["prompt_tokens"] == 45, "fresh plus cached"
    assert obs["cached_tokens"] == 20
    assert obs["completion_tokens"] == 3, "the message_delta count supersedes message_start"
    meta = obs["metadata"]
    assert meta["finish_reason"] == "stop"
    assert meta["response_id"] == "msg_s"
    assert meta["stream_completed"] is True
    assert "time_to_first_token_ms" in meta
    assert meta["output_tokens_per_second"] > 0


def test_an_abandoned_stream_still_records(pipeline: Pipeline) -> None:
    client = wrap_anthropic(fake_client(FakeMessages(result=FakeStream(events("Pa", "ris")))))
    stream = client.messages.create(model="claude-sonnet-5", max_tokens=8, messages=[], stream=True)
    for i, _event in enumerate(stream):
        if i == 2:
            break
    stream.close()

    obs = pipeline.one()
    assert obs["status"] == ObservationStatus.OK
    assert obs["output"] == "Pa"
    assert obs["metadata"]["stream_completed"] is False


# --------------------------------------------------------------------------- #
# messages.stream() — the context-manager helper
# --------------------------------------------------------------------------- #


def test_the_stream_helper_records_from_the_final_snapshot(pipeline: Pipeline) -> None:
    result = message(stop_reason="end_turn", cache_read_input_tokens=5)
    client = wrap_anthropic(fake_client(FakeMessages(result=result)))

    with client.messages.stream(model="claude-sonnet-5", max_tokens=8, messages=[]) as stream:
        assert stream.current_message_snapshot is result

    obs = pipeline.one()
    assert obs["output"] == "Paris."
    assert obs["prompt_tokens"] == 16
    assert obs["metadata"]["stream"] is True
    assert obs["metadata"]["finish_reason"] == "stop"


def test_the_stream_helper_records_an_error_raised_inside_the_block(pipeline: Pipeline) -> None:
    client = wrap_anthropic(fake_client())

    with pytest.raises(ValueError, match="consumer bug"):  # noqa: SIM117
        with client.messages.stream(model="claude-sonnet-5", max_tokens=8, messages=[]):
            raise ValueError("consumer bug")

    assert pipeline.one()["status"] == ObservationStatus.ERROR


# --------------------------------------------------------------------------- #
# Against the real anthropic client, over a mocked transport
# --------------------------------------------------------------------------- #

needs_anthropic = pytest.mark.skipif(anthropic is None, reason="anthropic not installed")

MESSAGE_BODY = {
    "id": "msg_real",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "Paris."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 11,
        "output_tokens": 3,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 0,
    },
}

RATE_HEADERS = {
    "request-id": "req_anthropic",
    "anthropic-ratelimit-requests-limit": "50",
    "anthropic-ratelimit-requests-remaining": "49",
    "anthropic-ratelimit-requests-reset": "2026-09-25T00:00:00Z",
    "anthropic-ratelimit-tokens-limit": "40000",
    "anthropic-ratelimit-tokens-remaining": "39000",
}


def real_client(handler: Any, *, is_async: bool = False, **kwargs: Any) -> Any:
    transport = httpx2.MockTransport(handler)
    if is_async:
        client: Any = anthropic.AsyncAnthropic(
            api_key="sk-ant-test", http_client=httpx2.AsyncClient(transport=transport), **kwargs
        )
    else:
        client = anthropic.Anthropic(
            api_key="sk-ant-test", http_client=httpx2.Client(transport=transport), **kwargs
        )
    return wrap_anthropic(client)


def sse_body(frames: list[dict[str, Any]]) -> str:
    return "".join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames)


def real_stream_frames() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_real_s",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 25, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Pa"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ris"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


@needs_anthropic
def test_real_client_message(pipeline: Pipeline) -> None:
    client = real_client(
        lambda request: httpx2.Response(200, json=MESSAGE_BODY, headers=RATE_HEADERS)
    )

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=64,
        messages=[{"role": "user", "content": "capital of France?"}],
    )

    assert response.content[0].text == "Paris."

    obs = pipeline.one()
    assert obs["model"] == "claude-sonnet-5"
    assert obs["prompt_tokens"] == 15
    assert obs["cached_tokens"] == 4
    assert obs["completion_tokens"] == 3
    assert obs["output"] == "Paris."
    meta = obs["metadata"]
    assert meta["finish_reason"] == "stop"
    assert meta["request_id"] == "req_anthropic", "the header hook attached to an httpx2 client"
    assert meta["http_attempts"] == 1
    assert meta["rate_limit"]["remaining_requests"] == 49
    assert meta["rate_limit"]["reset_requests"] == "2026-09-25T00:00:00Z"


@needs_anthropic
def test_real_async_client_is_detected_as_async(pipeline: Pipeline) -> None:
    client = real_client(
        lambda request: httpx2.Response(200, json=MESSAGE_BODY, headers=RATE_HEADERS),
        is_async=True,
    )

    async def main() -> Any:
        return await client.messages.create(model="claude-sonnet-5", max_tokens=8, messages=[])

    response = asyncio.run(main())

    assert response.content[0].text == "Paris."
    obs = pipeline.one()
    assert obs["prompt_tokens"] == 15
    assert obs["metadata"]["request_id"] == "req_anthropic"


@needs_anthropic
def test_real_client_streaming(pipeline: Pipeline) -> None:
    client = real_client(
        lambda request: httpx2.Response(
            200,
            content=sse_body(real_stream_frames()),
            headers={"content-type": "text/event-stream", "request-id": "req_s"},
        )
    )

    stream = client.messages.create(model="claude-sonnet-5", max_tokens=8, messages=[], stream=True)
    kinds = [event.type for event in stream]

    assert kinds[0] == "message_start"
    obs = pipeline.one()
    assert obs["output"] == "Paris"
    assert obs["prompt_tokens"] == 25
    assert obs["completion_tokens"] == 2
    assert obs["metadata"]["finish_reason"] == "stop"
    assert obs["metadata"]["request_id"] == "req_s"
    assert "time_to_first_token_ms" in obs["metadata"]


@needs_anthropic
def test_real_client_stream_helper(pipeline: Pipeline) -> None:
    client = real_client(
        lambda request: httpx2.Response(
            200,
            content=sse_body(real_stream_frames()),
            headers={"content-type": "text/event-stream"},
        )
    )

    with client.messages.stream(model="claude-sonnet-5", max_tokens=8, messages=[]) as stream:
        text = "".join(stream.text_stream)

    assert text == "Paris"
    obs = pipeline.one()
    assert obs["output"] == "Paris"
    assert obs["completion_tokens"] == 2
    assert obs["metadata"]["finish_reason"] == "stop"


@needs_anthropic
def test_real_client_api_error_propagates(pipeline: Pipeline) -> None:
    client = real_client(
        lambda request: httpx2.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
            headers={"anthropic-ratelimit-requests-remaining": "0"},
        ),
        max_retries=0,
    )

    with pytest.raises(anthropic.RateLimitError):
        client.messages.create(model="claude-sonnet-5", max_tokens=8, messages=[])

    obs = pipeline.one()
    assert obs["status"] == ObservationStatus.ERROR
    assert "RateLimitError" in obs["status_message"]
    assert obs["metadata"]["http_status"] == 429
    assert obs["metadata"]["rate_limit"] == {"remaining_requests": 0}
