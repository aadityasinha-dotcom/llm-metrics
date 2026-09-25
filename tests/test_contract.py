"""The wire contract: what this SDK sends versus what the server accepts.

The server publishes its request schemas as ``openapi.json`` (``make openapi``
in the ``llm-observe`` repo). A copy is vendored at ``tests/contract/openapi.json``
and refreshed with ``scripts/sync_openapi.py``; ``LLM_METRICS_OPENAPI`` points
the tests at a different file, for checking against a branch of the server.

What is verified here, against a real request captured off the transport:

* the endpoint, the bearer auth scheme, and the ``X-SDK-Version`` header
* the flat ``{"events": [...]}`` envelope
* every event the SDK can produce validates against the server's schema for
  its type, and every key it carries is one the server knows - not merely
  tolerates. The server ignores unknown fields by design, so plain schema
  validation would pass an SDK that had renamed ``prompt_tokens`` to
  ``tokens_in`` and was silently losing every count. The subset check is
  the assertion that matters.

What cannot be verified from a schema, and where it is verified instead:

* The server accepts children before their parent (a batch can carry an
  observation before the trace it belongs to, and a score before either).
  That is behaviour, not shape; the server's own suite asserts it in
  ``test_observations_arriving_before_their_trace_create_a_stub`` and
  ``test_a_score_may_arrive_before_its_trace``.
* Aliases. Pydantic publishes only the first spelling of a field that accepts
  several, so ``parent_id`` (what the SDK sends) appears in the contract as
  ``parent_observation_id``. The mapping is written down in ``ALIASES`` below,
  and a server rename shows up here as a missing property.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

import llm_metrics
from llm_metrics import _runtime, observe, score, update_observation, update_trace
from llm_metrics.buffer import EventBuffer
from llm_metrics.client import INGEST_PATH, IngestClient

jsonschema = pytest.importorskip("jsonschema")

CONTRACT_PATH = Path(__file__).parent / "contract" / "openapi.json"

#: SDK key -> contract property, per event type. The server accepts both
#: spellings; the contract only prints one.
ALIASES: dict[str, dict[str, str]] = {
    "trace": {"start_time": "started_at", "end_time": "ended_at"},
    "observation": {
        "start_time": "started_at",
        "end_time": "ended_at",
        "parent_id": "parent_observation_id",
        "status": "level",
    },
    "score": {"timestamp": "scored_at"},
}

#: The envelope discriminator. Consumed by the server's router to decide which
#: schema an event is parsed with, so it is not a property of those schemas.
DISCRIMINATOR = "type"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    path = Path(os.environ.get("LLM_METRICS_OPENAPI") or CONTRACT_PATH)
    return cast("dict[str, Any]", json.loads(path.read_text()))


@pytest.fixture(scope="module")
def schemas(contract: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", contract["components"]["schemas"])


def _validator(contract: dict[str, Any], name: str) -> Any:
    """A validator for one component schema, with ``$ref`` resolving into the
    whole document, formats (uuid, date-time) checked rather than ignored."""
    root = {"$ref": f"#/components/schemas/{name}", "components": contract["components"]}
    return jsonschema.Draft202012Validator(
        root, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


class Capture:
    """The last request the SDK made, exactly as the wire saw it."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            202, json={"accepted_traces": 0, "accepted_observations": 0, "accepted_scores": 0}
        )

    @property
    def request(self) -> httpx.Request:
        assert len(self.requests) == 1, f"expected one request, saw {len(self.requests)}"
        return self.requests[0]

    @property
    def body(self) -> dict[str, Any]:
        return cast("dict[str, Any]", json.loads(self.request.content))


@pytest.fixture
def wire() -> Iterator[tuple[Capture, EventBuffer]]:
    """The real client and buffer, with only the socket replaced."""
    capture = Capture()
    client = IngestClient(
        api_key="sk-contract", host="https://ingest.example", transport=httpx.MockTransport(capture)
    )
    buffer = EventBuffer(client.send, flush_at=10_000, flush_interval=300.0, start=False)
    _runtime.configure(sink=buffer, enabled=True, capture_input=True, capture_output=True)
    try:
        yield capture, buffer
    finally:
        _runtime.shutdown(timeout=1.0)
        client.close()


def _exercise_everything() -> None:
    """One run that produces every event shape the SDK has.

    A root span with attribution, a nested generation with every token field,
    a failed tool call, a streamed generator (so latency is a float measured
    over the stream), and scores both ambient and after the fact.
    """

    @observe(as_type="generation", name="complete")
    def complete(prompt: str) -> str:
        update_observation(
            model="gpt-4o",
            prompt_name="qa",
            prompt_version=3,
            prompt_tokens=1000,
            completion_tokens=100,
            cached_tokens=900,
            reasoning_tokens=40,
            metadata={"finish_reason": "stop"},
        )
        return "Paris."

    @observe(as_type="tool")
    def failing_tool() -> None:
        raise RuntimeError("boom")

    @observe(as_type="generation")
    def stream() -> Iterator[str]:
        yield "Pa"
        yield "ris"

    @observe()
    def answer(user_id: str, question: str) -> str:
        update_trace(
            user_id=user_id,
            session_id="sess-1",
            tags=["contract", "qa"],
            metadata={"tenant": "acme"},
        )
        with pytest.raises(RuntimeError):
            failing_tool()
        "".join(stream())
        reply = complete(question)
        score("has_answer", True, source="heuristic", comment="cites Paris")
        score("rating", 0.875)
        score("category", "geography")
        return reply

    _runtime.configure(environment="test", release="abc123")
    try:
        answer("user-42", "capital of France?")
    finally:
        _runtime.configure(environment=None, release=None)
    llm_metrics.score("thumbs", -1, trace_id="00000000-0000-4000-8000-000000000001")


def _events(capture: Capture, buffer: EventBuffer) -> list[dict[str, Any]]:
    buffer.flush_once()
    body = capture.body
    assert isinstance(body.get("events"), list)
    return list(body["events"])


def _schema_name(event: dict[str, Any]) -> str:
    kind = event.get(DISCRIMINATOR)
    if kind == "trace":
        return "IngestTrace"
    if kind == "score":
        return "IngestScore"
    return "IngestObservation"


def _translate(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Rename the SDK's spellings to the contract's, and drop the discriminator
    where the schema does not carry it."""
    name = _schema_name(event)
    family = {"IngestTrace": "trace", "IngestScore": "score"}.get(name, "observation")
    aliases = ALIASES[family]
    translated = {aliases.get(key, key): value for key, value in event.items()}
    if family != "observation":
        translated.pop(DISCRIMINATOR, None)
    return name, translated


# --------------------------------------------------------------------------- #
# The operation
# --------------------------------------------------------------------------- #


def test_the_ingest_operation_is_where_the_client_posts(contract: dict[str, Any]) -> None:
    operation = contract["paths"][INGEST_PATH]["post"]
    body = operation["requestBody"]["content"]["application/json"]["schema"]
    assert body == {"$ref": "#/components/schemas/IngestBatch"}
    assert operation["responses"].get("202"), "the SDK treats 202 as delivered"


def test_auth_is_a_bearer_token(
    contract: dict[str, Any], wire: tuple[Capture, EventBuffer]
) -> None:
    """The header the SDK sends must be the scheme the operation declares."""
    capture, buffer = wire
    operation = contract["paths"][INGEST_PATH]["post"]
    schemes = contract["components"]["securitySchemes"]
    required = [name for entry in operation["security"] for name in entry]
    assert required, "the operation declares no security at all"
    assert all(
        schemes[name] == {**schemes[name], "type": "http", "scheme": "bearer"} for name in required
    )

    _exercise_everything()
    _events(capture, buffer)
    assert capture.request.headers["Authorization"] == "Bearer sk-contract"
    assert capture.request.method == "POST"
    assert capture.request.url.path == INGEST_PATH


def test_sdk_version_header_is_declared(
    contract: dict[str, Any], wire: tuple[Capture, EventBuffer]
) -> None:
    capture, buffer = wire
    operation = contract["paths"][INGEST_PATH]["post"]
    headers = {p["name"].lower(): p for p in operation.get("parameters", []) if p["in"] == "header"}
    assert "x-sdk-version" in headers, "the server no longer reads X-SDK-Version"
    assert headers["x-sdk-version"].get("required") is not True

    _exercise_everything()
    _events(capture, buffer)
    assert capture.request.headers["X-SDK-Version"] == llm_metrics.__version__
    assert capture.request.headers["Content-Type"] == "application/json"


# --------------------------------------------------------------------------- #
# The envelope
# --------------------------------------------------------------------------- #


def test_the_flat_envelope_is_documented(
    schemas: dict[str, Any], wire: tuple[Capture, EventBuffer]
) -> None:
    """``events`` is normalised away before validation, so it is not a schema
    property - the description is the only place the contract states it."""
    capture, buffer = wire
    _exercise_everything()
    body = capture.body if _events(capture, buffer) else {}
    assert set(body) == {"events"}, "the SDK sends exactly one top-level key"

    batch = schemas["IngestBatch"]
    assert '"events"' in batch.get("description", ""), "the server stopped documenting `events`"
    assert {"traces", "observations", "scores"} <= set(batch["properties"]), (
        "the normalised form no longer has a home for every event kind the SDK sends"
    )


# --------------------------------------------------------------------------- #
# The events
# --------------------------------------------------------------------------- #


def test_every_event_validates_against_its_schema(
    contract: dict[str, Any], wire: tuple[Capture, EventBuffer]
) -> None:
    capture, buffer = wire
    _exercise_everything()
    events = _events(capture, buffer)

    kinds = {e.get(DISCRIMINATOR) for e in events}
    assert {"trace", "span", "generation", "tool", "score"} <= kinds, "the run lost a shape"

    validators = {
        name: _validator(contract, name)
        for name in ("IngestTrace", "IngestObservation", "IngestScore")
    }
    for event in events:
        name, translated = _translate(event)
        errors = sorted(validators[name].iter_errors(translated), key=lambda e: list(e.path))
        assert not errors, (
            f"{name} rejects {event.get('name', event.get(DISCRIMINATOR))!r}: "
            + "; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors)
        )


def test_every_sdk_key_is_one_the_server_stores(
    schemas: dict[str, Any], wire: tuple[Capture, EventBuffer]
) -> None:
    """Schema validation alone would pass a renamed field: the server ignores
    unknown keys by design. This is the check that a key is not merely
    tolerated but has a column."""
    capture, buffer = wire
    _exercise_everything()

    for event in _events(capture, buffer):
        name, translated = _translate(event)
        known = set(schemas[name]["properties"])
        unknown = set(translated) - known
        assert not unknown, f"{name} would silently drop {sorted(unknown)} from {event}"


def test_alias_targets_exist(schemas: dict[str, Any]) -> None:
    """A server rename of an aliased field must show up here, not as a silent
    drop on the wire."""
    for family, mapping in ALIASES.items():
        name = {"trace": "IngestTrace", "score": "IngestScore"}.get(family, "IngestObservation")
        for sdk_key, contract_key in mapping.items():
            assert contract_key in schemas[name]["properties"], (
                f"{name} no longer has {contract_key!r}, the target of SDK key {sdk_key!r}"
            )


def test_the_fields_the_sdk_relies_on_are_present(schemas: dict[str, Any]) -> None:
    """The columns the dashboard will slice on. Losing one is a product
    regression, not a schema nicety."""
    assert {"user_id", "session_id", "tags", "environment", "release", "metadata"} <= set(
        schemas["IngestTrace"]["properties"]
    )
    assert {
        "model",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "latency_ms",
        "prompt_name",
        "prompt_version",
        "level",
        "status_message",
        "input",
        "output",
    } <= set(schemas["IngestObservation"]["properties"])
    assert {"trace_id", "observation_id", "name", "value", "source", "comment"} <= set(
        schemas["IngestScore"]["properties"]
    )


def test_measurements_may_be_fractional(schemas: dict[str, Any]) -> None:
    """latency_ms comes from perf_counter and is a float. A contract that said
    `integer` would 422 every batch."""
    properties = schemas["IngestObservation"]["properties"]
    for field in ("latency_ms", "prompt_tokens", "completion_tokens", "cached_tokens"):
        types = properties[field].get("type")
        assert "number" in (types if isinstance(types, list) else [types]), field


def test_score_values_of_every_kind_are_accepted(contract: dict[str, Any]) -> None:
    validator = _validator(contract, "IngestScore")
    for value in (True, 0, 0.875, "geography"):
        assert not list(
            validator.iter_errors(
                {"name": "s", "value": value, "trace_id": "00000000-0000-4000-8000-000000000001"}
            )
        ), value


def test_no_cost_crosses_the_wire(
    schemas: dict[str, Any], wire: tuple[Capture, EventBuffer]
) -> None:
    """Rule 3 on both sides: the SDK never sends it, the server never accepts it."""
    capture, buffer = wire
    _exercise_everything()
    _events(capture, buffer)
    assert "cost" not in capture.request.content.decode().lower()
    assert not any("cost" in key for key in schemas["IngestObservation"]["properties"])
