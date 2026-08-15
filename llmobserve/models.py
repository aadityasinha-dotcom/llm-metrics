"""Wire-format dataclasses for the ingest API.

These are the only shapes that go over the network. They must stay in sync with
the server's published ``openapi.json`` — ``tests/test_contract.py`` is the
enforcement mechanism.

Field naming follows the OpenTelemetry GenAI semantic conventions where the
server schema allows it:

===========================  ==================================
this SDK                     OTel GenAI attribute
===========================  ==================================
``Observation.model``        ``gen_ai.request.model``
``Observation.prompt_tokens``      ``gen_ai.usage.input_tokens``
``Observation.completion_tokens``  ``gen_ai.usage.output_tokens``
===========================  ==================================

Note what is deliberately absent: there is no ``cost`` field. Token counts go
up, pricing is applied server-side so it can change without users upgrading the
SDK.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = ["Observation", "ObservationType", "Trace"]


def _new_id() -> str:
    """Client-generated id so nested observations can reference a parent before
    the server has ever seen it."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime rather than guessing the local zone.

    A naive timestamp that silently picks up the host's local offset produces
    traces that are hours off on the dashboard, which is worse than being
    slightly wrong in a documented direction.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _as_utc(value).isoformat()


class ObservationType:
    """Discriminator for what kind of work an observation represents.

    A bare class of string constants rather than an ``enum.Enum``: the value
    travels as a plain string on the wire, and users writing custom
    instrumentation should be able to pass ``"generation"`` directly without
    importing anything.
    """

    SPAN = "span"
    GENERATION = "generation"
    TOOL = "tool"
    RETRIEVAL = "retrieval"


@dataclass
class Trace:
    """One user-facing request — the root of a tree of observations."""

    name: str
    id: str = field(default_factory=_new_id)
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=_utcnow)
    end_time: datetime | None = None

    def __post_init__(self) -> None:
        self.start_time = _as_utc(self.start_time)
        if self.end_time is not None:
            self.end_time = _as_utc(self.end_time)

    def end(self, when: datetime | None = None) -> None:
        """Mark the trace complete. Idempotent-ish: the last call wins."""
        self.end_time = _as_utc(when) if when is not None else _utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the ingest payload.

        Keys whose value is ``None`` are omitted so the server can distinguish
        "not set by this SDK version" from "explicitly null". Keys that are
        always present are always present, so the shape is predictable.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "type": "trace",
            "name": self.name,
            "metadata": self.metadata,
            "start_time": _iso(self.start_time),
        }
        _put(payload, "user_id", self.user_id)
        _put(payload, "end_time", _iso(self.end_time))
        return payload


@dataclass
class Observation:
    """One LLM, tool, or retrieval call. Nests under a trace via ``trace_id``
    and under another observation via ``parent_id``."""

    trace_id: str
    name: str
    id: str = field(default_factory=_new_id)
    parent_id: str | None = None
    type: str = ObservationType.SPAN
    model: str | None = None
    input: Any = None
    output: Any = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=_utcnow)
    end_time: datetime | None = None

    def __post_init__(self) -> None:
        self.start_time = _as_utc(self.start_time)
        if self.end_time is not None:
            self.end_time = _as_utc(self.end_time)

    def end(self, when: datetime | None = None) -> None:
        """Mark the observation complete and derive ``latency_ms`` from the
        wall clock if the caller has not measured it more precisely."""
        self.end_time = _as_utc(when) if when is not None else _utcnow()
        if self.latency_ms is None:
            delta = self.end_time - self.start_time
            self.latency_ms = delta.total_seconds() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the ingest payload. See :meth:`Trace.to_dict`."""
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "trace_id": self.trace_id,
            "name": self.name,
            "metadata": self.metadata,
            "start_time": _iso(self.start_time),
        }
        _put(payload, "parent_id", self.parent_id)
        _put(payload, "model", self.model)
        _put(payload, "input", self.input)
        _put(payload, "output", self.output)
        _put(payload, "prompt_tokens", self.prompt_tokens)
        _put(payload, "completion_tokens", self.completion_tokens)
        _put(payload, "latency_ms", self.latency_ms)
        _put(payload, "end_time", _iso(self.end_time))
        return payload


def _put(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value
