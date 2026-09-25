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
``Observation.cached_tokens``      ``gen_ai.usage.cache_read.input_tokens``
``Observation.reasoning_tokens``   (no OTel attribute yet)
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

__all__ = ["Observation", "ObservationStatus", "ObservationType", "Score", "ScoreSource", "Trace"]


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


class ObservationStatus:
    """Outcome of a call, mapping to the OTel span status codes."""

    OK = "ok"
    ERROR = "error"


@dataclass
class Trace:
    """One user-facing request — the root of a tree of observations."""

    name: str
    id: str = field(default_factory=_new_id)
    user_id: str | None = None
    #: Groups the traces of one conversation or job, so a dashboard can show
    #: cost and quality per session rather than per isolated request.
    session_id: str | None = None
    #: Free-form labels for slicing: feature, tenant, experiment arm.
    tags: list[str] = field(default_factory=list)
    #: Where this ran. Filled from ``LLM_METRICS_ENVIRONMENT`` /
    #: ``LLM_METRICS_RELEASE`` when the SDK creates the trace, so a regression
    #: can be pinned to a deploy without the user tagging every call.
    environment: str | None = None
    release: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=_utcnow)
    end_time: datetime | None = None
    #: Local only, never serialised. ``False`` means the sampler dropped this
    #: trace; every observation under it is dropped too, so the tree stays
    #: whole on the server rather than arriving with holes in it.
    sampled: bool = field(default=True, compare=False, repr=False)

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
        _put(payload, "session_id", self.session_id)
        if self.tags:
            payload["tags"] = list(self.tags)
        _put(payload, "environment", self.environment)
        _put(payload, "release", self.release)
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
    #: Subsets of the two counts above, kept separate because providers price
    #: them differently: a cached prompt token costs a fraction of a fresh one,
    #: and reasoning tokens are billed as output the caller never sees.
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_ms: float | None = None
    #: Which prompt template produced this call, so quality and cost can be
    #: compared across versions. Set with :func:`llm_metrics.update_observation`.
    prompt_name: str | None = None
    prompt_version: str | None = None
    status: str = ObservationStatus.OK
    status_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=_utcnow)
    end_time: datetime | None = None

    def __post_init__(self) -> None:
        self.start_time = _as_utc(self.start_time)
        if self.end_time is not None:
            self.end_time = _as_utc(self.end_time)

    def fail(self, exc: BaseException) -> None:
        """Record that the wrapped call raised.

        Stores the type and message, never the traceback: a traceback carries
        source lines and local context that a user did not consent to ship to
        an observability backend.
        """
        self.status = ObservationStatus.ERROR
        self.status_message = f"{type(exc).__name__}: {exc}"[:500]

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
            "status": self.status,
            "metadata": self.metadata,
            "start_time": _iso(self.start_time),
        }
        _put(payload, "parent_id", self.parent_id)
        _put(payload, "status_message", self.status_message)
        _put(payload, "model", self.model)
        _put(payload, "input", self.input)
        _put(payload, "output", self.output)
        _put(payload, "prompt_tokens", self.prompt_tokens)
        _put(payload, "completion_tokens", self.completion_tokens)
        _put(payload, "cached_tokens", self.cached_tokens)
        _put(payload, "reasoning_tokens", self.reasoning_tokens)
        _put(payload, "latency_ms", self.latency_ms)
        _put(payload, "prompt_name", self.prompt_name)
        _put(payload, "prompt_version", self.prompt_version)
        _put(payload, "end_time", _iso(self.end_time))
        return payload


class ScoreSource:
    """Who produced a score. Matches the server's ``scores.source`` enum."""

    HUMAN = "human"
    LLM_JUDGE = "llm_judge"
    HEURISTIC = "heuristic"


@dataclass
class Score:
    """A judgement about a trace or an observation.

    Scores close the loop that token ledgers cannot: a thumbs-down from a user,
    a heuristic check that the answer cited a document, a judge model's rating.
    Exactly one of ``trace_id`` / ``observation_id`` is normally set; both is
    allowed and means "this observation, in this trace".
    """

    name: str
    value: float | int | bool | str
    id: str = field(default_factory=_new_id)
    trace_id: str | None = None
    observation_id: str | None = None
    comment: str | None = None
    source: str = ScoreSource.HUMAN
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        self.timestamp = _as_utc(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": "score",
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": _iso(self.timestamp),
        }
        _put(payload, "trace_id", self.trace_id)
        _put(payload, "observation_id", self.observation_id)
        _put(payload, "comment", self.comment)
        return payload


def _put(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value
