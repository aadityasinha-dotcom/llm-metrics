"""The default pipeline: one buffer, one client, created on first use.

``@observe`` needs somewhere to put events. Rather than making every user wire
a buffer to a client, this module owns a lazily-built default pair and hands
the decorator an :func:`emit` that never raises.

Construction is lazy on purpose. Building the pipeline at import time would
start a background thread as a side effect of ``import llm_metrics``, which is
rude in a library and actively wrong in anything that forks after import.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from llm_metrics.buffer import (
    DEFAULT_FLUSH_AT,
    DEFAULT_FLUSH_INTERVAL,
    DEFAULT_MAX_SIZE,
    DEFAULT_SHUTDOWN_TIMEOUT,
    EventBuffer,
)
from llm_metrics.client import ENV_API_KEY, IngestClient

__all__ = [
    "Settings",
    "Stats",
    "configure",
    "current_settings",
    "emit",
    "flush",
    "is_enabled",
    "shutdown",
    "stats",
]

ENV_ENABLED = "LLM_METRICS_ENABLED"
ENV_ENVIRONMENT = "LLM_METRICS_ENVIRONMENT"
ENV_RELEASE = "LLM_METRICS_RELEASE"
ENV_SAMPLE_RATE = "LLM_METRICS_SAMPLE_RATE"

#: Transforms a captured input or output before it is stored on an event.
#: Returning ``None`` drops the value. If it raises, the value is dropped too:
#: shipping something a redactor choked on is the one outcome it must not have.
Redactor = Callable[[Any], Any]

_FALSEY = frozenset({"0", "false", "no", "off"})

#: Distinguishes "caller did not mention this" from "caller passed None".
#: Typed loosely so callers keep normal parameter types at the call site.
_UNSET: Any = object()


@dataclass(frozen=True)
class Settings:
    """Knobs the decorator reads on every call. Cheap to copy, never mutated."""

    enabled: bool = True
    capture_input: bool = True
    capture_output: bool = True
    #: Per-value ceiling on captured input/output. Prompts are large and
    #: responses larger; without a cap one runaway payload can fill the buffer.
    max_value_chars: int = 2_000
    #: Stamped on every trace the SDK creates. Filters on the dashboard, and
    #: the only way to tell "the model got worse" from "we deployed".
    environment: str | None = None
    release: str | None = None
    #: Fraction of root traces kept, decided once per trace so a kept trace is
    #: complete and a dropped one leaves nothing behind. 1.0 keeps everything.
    sample_rate: float = 1.0
    #: Applied to every captured input and output before it is stored.
    redact: Redactor | None = None


def _env_enabled() -> bool:
    raw = os.environ.get(ENV_ENABLED)
    return True if raw is None else raw.strip().lower() not in _FALSEY


def _env_sample_rate() -> float:
    raw = os.environ.get(ENV_SAMPLE_RATE)
    if raw is None:
        return 1.0
    try:
        return _clamp_rate(float(raw))
    except ValueError:
        return 1.0


def _clamp_rate(rate: float) -> float:
    return min(1.0, max(0.0, rate))


def _env_str(name: str) -> str | None:
    raw = os.environ.get(name)
    return raw.strip() or None if raw is not None else None


# Guards the globals below. Never held while flushing — it is taken only to
# swap references, never around network I/O.
_lock = threading.Lock()
_settings = Settings(
    enabled=_env_enabled(),
    environment=_env_str(ENV_ENVIRONMENT),
    release=_env_str(ENV_RELEASE),
    sample_rate=_env_sample_rate(),
)
_buffer: EventBuffer | None = None
_client: IngestClient | None = None
_overridden = False  # a test or embedder supplied its own sink


def current_settings() -> Settings:
    return _settings


def is_enabled() -> bool:
    """Cheap gate for the decorator's hot path."""
    return _settings.enabled


def configure(
    api_key: str = _UNSET,
    host: str = _UNSET,
    *,
    enabled: bool | None = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
    max_value_chars: int | None = None,
    environment: str | None = _UNSET,
    release: str | None = _UNSET,
    sample_rate: float | None = None,
    redact: Redactor | None = _UNSET,
    max_size: int = _UNSET,
    flush_at: int = _UNSET,
    flush_interval: float = _UNSET,
    shutdown_timeout: float = _UNSET,
    sink: EventBuffer | None = None,
) -> None:
    """Set up the default pipeline explicitly.

    Optional — leaving it out and setting ``$LLM_METRICS_API_KEY`` works just as
    well.

    Only arguments that actually affect transport rebuild the pipeline. A call
    that just flips ``capture_input`` leaves the running buffer and its queued
    events alone: silently discarding buffered traces because someone adjusted
    a capture flag would be a nasty surprise. When the pipeline *is* replaced,
    the outgoing one is flushed first.

    Args:
        environment: Stamped on every trace, e.g. ``"prod"``. ``None`` clears it.
        release: Stamped on every trace, e.g. a git SHA. ``None`` clears it.
        sample_rate: Fraction of root traces to keep, ``0.0`` to ``1.0``. Decided
            once per trace; children follow their root.
        redact: Called on every captured input and output. Return the value to
            store, or ``None`` to drop it. ``None`` here removes the redactor.
        sink: Use this buffer instead of building one. For tests and for
            embedders that want to own the transport. Passing back the buffer
            that is already installed keeps it running.
    """
    global _buffer, _client, _settings, _overridden

    updates: dict[str, Any] = {}
    if enabled is not None:
        updates["enabled"] = enabled
    if capture_input is not None:
        updates["capture_input"] = capture_input
    if capture_output is not None:
        updates["capture_output"] = capture_output
    if max_value_chars is not None:
        updates["max_value_chars"] = max_value_chars
    if environment is not _UNSET:
        updates["environment"] = environment
    if release is not _UNSET:
        updates["release"] = release
    if sample_rate is not None:
        updates["sample_rate"] = _clamp_rate(float(sample_rate))
    if redact is not _UNSET:
        updates["redact"] = redact

    transport = (api_key, host, max_size, flush_at, flush_interval, shutdown_timeout)
    rebuild = sink is not None or any(value is not _UNSET for value in transport)

    retire_buffer, retire_client = None, None
    with _lock:
        _settings = replace(_settings, **updates)

        # Settings-only calls build nothing. _sink() creates the default
        # pipeline on first emit anyway, so building here would only start a
        # thread — and warn about a missing API key — for a caller who was
        # adjusting a capture flag, possibly while disabling the SDK entirely.
        if rebuild:
            retire_buffer, retire_client = _buffer, _client

            if sink is not None:
                if sink is retire_buffer:
                    # Re-installing the running buffer. Retiring it here would
                    # shut down the very sink we are about to point at.
                    retire_buffer, retire_client = None, None
                _buffer, _client, _overridden = sink, None, True
            else:
                _client = IngestClient(
                    api_key=_or_none(api_key),
                    host=_or_none(host),
                )
                _buffer = EventBuffer(
                    _client.send,
                    max_size=_or_default(max_size, DEFAULT_MAX_SIZE),
                    flush_at=_or_default(flush_at, DEFAULT_FLUSH_AT),
                    flush_interval=_or_default(flush_interval, DEFAULT_FLUSH_INTERVAL),
                    shutdown_timeout=_or_default(shutdown_timeout, DEFAULT_SHUTDOWN_TIMEOUT),
                )
                _overridden = False

    # Outside the lock: shutting the old pipeline down can block on the network.
    _retire(retire_buffer, retire_client)


def _or_none(value: Any) -> Any:
    return None if value is _UNSET else value


def _or_default(value: Any, default: Any) -> Any:
    return default if value is _UNSET else value


def _sink() -> EventBuffer | None:
    """The buffer to emit into, building the default one on first use."""
    buffer = _buffer
    if buffer is not None:
        return buffer

    with _lock:
        if _buffer is not None:
            return _buffer
        if not _settings.enabled:
            return None
        _install_default_locked()
        return _buffer


def _install_default_locked() -> None:
    global _buffer, _client
    client = IngestClient()
    _client = client
    _buffer = EventBuffer(client.send)


def emit(event: Any) -> None:
    """Queue one Trace or Observation. Never raises, never blocks on I/O."""
    try:
        if not _settings.enabled:
            return
        buffer = _sink()
        if buffer is not None:
            buffer.add(event)
    except Exception:  # noqa: BLE001 - rule 2: never crash the host app
        pass


def flush() -> None:
    """Drain buffered events on the calling thread.

    Blocking, and deliberately not used anywhere inside the SDK. For scripts
    and tests that want delivery before moving on.
    """
    buffer = _buffer
    if buffer is not None:
        with contextlib.suppress(Exception):  # rule 2: never crash the host app
            buffer.flush_once()


def shutdown(timeout: float | None = None) -> bool:
    """Flush and tear down the default pipeline.

    Rarely needed — the buffer's own atexit hook covers process exit. Useful
    when an embedder wants delivery to finish at a known point, and in tests.

    Returns:
        ``True`` if the flush thread finished within the deadline.
    """
    global _buffer, _client, _overridden
    with _lock:
        buffer, client = _buffer, _client
        _buffer, _client, _overridden = None, None, False
    return _retire(buffer, client, timeout)


def _retire(
    buffer: EventBuffer | None,
    client: IngestClient | None,
    timeout: float | None = None,
) -> bool:
    delivered = True
    if buffer is not None:
        try:
            delivered = buffer.shutdown(timeout)
        except Exception:  # noqa: BLE001 - rule 2
            delivered = False
    if client is not None:
        with contextlib.suppress(Exception):  # rule 2
            client.close()
    return delivered


@dataclass(frozen=True)
class Stats:
    """A point-in-time view of the pipeline's health.

    The SDK fails silently by design, so this is how a user finds out whether
    it is actually delivering. Every field is a counter since the pipeline was
    built; ``None`` for the client fields means a custom sink owns transport.
    """

    enabled: bool
    #: Events accepted by :func:`emit`.
    queued: int = 0
    #: Events handed to the transport.
    flushed: int = 0
    #: Events discarded because the buffer was full.
    dropped_on_overflow: int = 0
    #: Batches the transport gave up on.
    failed_batches: int = 0
    sent_batches: int | None = None
    sent_events: int | None = None
    #: Events the transport dropped after exhausting retries or on a 4xx.
    dropped_by_transport: int | None = None
    retries: int | None = None
    last_error: str | None = None

    @property
    def healthy(self) -> bool:
        """No event has been lost since the pipeline started."""
        return not (self.dropped_on_overflow or self.failed_batches or self.dropped_by_transport)


def stats() -> Stats:
    """Counters from the running pipeline. Never raises, never blocks.

    Before the first event there is no pipeline, and the counters are zero.
    """
    try:
        buffer, client = _buffer, _client
        view = Stats(enabled=_settings.enabled)
        if buffer is not None:
            b = buffer.stats
            view = replace(
                view,
                queued=b.queued,
                flushed=b.flushed,
                dropped_on_overflow=b.dropped,
                failed_batches=b.failed_batches,
            )
        if client is not None:
            c = client.stats
            view = replace(
                view,
                sent_batches=c.sent_batches,
                sent_events=c.sent_events,
                dropped_by_transport=c.dropped_events,
                retries=c.retries,
                last_error=c.last_error,
            )
        return view
    except Exception:  # noqa: BLE001 - rule 2
        return Stats(enabled=_settings.enabled)


def _api_key_present() -> bool:
    """Only used for diagnostics — the client owns the real decision."""
    return bool(os.environ.get(ENV_API_KEY))
