"""Design rule 2: the SDK never crashes the host app.

Every failure an ingest API can produce — refused connections, timeouts, 5xx,
a rejected API key, a garbage response body, an unserialisable payload — has to
come out as a dropped event and a bumped counter. Nothing here may raise into
caller code, and nothing may kill the flush thread.

The transport is faked with ``httpx.MockTransport`` so these tests exercise the
real ``IngestClient`` retry logic without a network or an extra test dependency.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import httpx
import pytest

from llm_metrics._version import __version__
from llm_metrics.buffer import Deadline, EventBuffer
from llm_metrics.client import ENV_API_KEY, ENV_HOST, IngestClient
from llm_metrics.models import Observation

Handler = Callable[[httpx.Request], httpx.Response]


class Responder:
    """A handler that always answers with the given status, recording requests."""

    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return httpx.Response(self.status, headers=self.headers, json={"ok": self.status < 400})


def make_client(
    handler: Handler,
    *,
    max_attempts: int = 3,
    # Retries are instant unless a test says otherwise — no test should pay for
    # real backoff sleeps.
    backoff_base: float = 0.0,
    backoff_max: float = 0.0,
    timeout: float = 5.0,
) -> IngestClient:
    return IngestClient(
        api_key="test-key",
        host="https://ingest.example",
        transport=httpx.MockTransport(handler),
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        timeout=timeout,
    )


def payload(n: int = 3) -> list[dict[str, object]]:
    return [Observation(trace_id="t", name=f"c{i}").to_dict() for i in range(n)]


# --------------------------------------------------------------------------- #
# Transport-level failures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
)
def test_network_failures_are_swallowed_and_retried(exc: Exception) -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise exc

    client = make_client(handler, max_attempts=3)
    client.send(payload())  # must not raise

    assert len(attempts) == 3, "transport errors should be retried to the attempt limit"
    assert client.stats.sent_batches == 0
    assert client.stats.dropped_events == 3
    assert client.stats.retries == 2


def test_unreachable_host_does_not_raise_without_a_mock_transport() -> None:
    """The real socket path, against a port nothing is listening on."""
    client = IngestClient(
        api_key="test-key",
        host="http://127.0.0.1:1",
        max_attempts=2,
        timeout=0.5,
        backoff_base=0.0,
        backoff_max=0.0,
    )
    try:
        client.send(payload())  # must not raise
        assert client.stats.dropped_events == 3
        assert client.stats.last_error is not None
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# HTTP status handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [500, 502, 503, 504, 429, 408, 425])
def test_retryable_statuses_are_retried_then_dropped(status: int) -> None:
    handler = Responder(status)
    client = make_client(handler, max_attempts=3)

    client.send(payload())

    assert len(handler.calls) == 3
    assert client.stats.dropped_events == 3
    assert client.stats.permanent_failures == 0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422])
def test_client_errors_are_dropped_without_retrying(status: int) -> None:
    """A bad key or a malformed payload will fail identically on retry.

    Retrying them is not merely useless — it spends the shutdown budget that
    other, deliverable batches need.
    """
    handler = Responder(status)
    client = make_client(handler, max_attempts=3)

    client.send(payload())

    assert len(handler.calls) == 1, f"HTTP {status} must not be retried"
    assert client.stats.permanent_failures == 1
    assert client.stats.dropped_events == 3


def test_success_stops_retrying_and_counts_events() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503 if len(seen) < 3 else 202, json={})

    client = make_client(handler, max_attempts=5)
    client.send(payload(4))

    assert len(seen) == 3, "should stop as soon as one attempt succeeds"
    assert client.stats.sent_batches == 1
    assert client.stats.sent_events == 4
    assert client.stats.dropped_events == 0


def test_429_retry_after_is_honoured() -> None:
    handler = Responder(429, headers={"Retry-After": "0.05"})
    client = IngestClient(
        api_key="test-key",
        host="https://ingest.example",
        transport=httpx.MockTransport(handler),
        max_attempts=2,
        backoff_base=10.0,  # would be a 10s sleep if Retry-After were ignored
        backoff_max=30.0,
    )

    started = time.monotonic()
    client.send(payload())
    elapsed = time.monotonic() - started

    assert len(handler.calls) == 2
    assert elapsed < 1.0, f"Retry-After was ignored; slept {elapsed:.2f}s"


def test_garbage_retry_after_falls_back_to_backoff() -> None:
    """An HTTP-date or nonsense value must not blow up the retry path."""
    handler = Responder(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
    client = make_client(handler, max_attempts=2)

    client.send(payload())  # must not raise

    assert len(handler.calls) == 2
    assert client.stats.dropped_events == 3


def test_garbage_response_body_on_success_is_not_parsed() -> None:
    """We do not read the response body, so it cannot break us."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x00\x01 not json at all")

    client = make_client(handler)
    client.send(payload())

    assert client.stats.sent_batches == 1
    assert client.stats.dropped_events == 0


# --------------------------------------------------------------------------- #
# Payload and configuration problems
# --------------------------------------------------------------------------- #


def test_unserialisable_metadata_degrades_to_a_string() -> None:
    """User metadata is arbitrary. One odd value must not cost the batch."""
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(202, json={})

    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    client = make_client(handler)
    event = Observation(trace_id="t", name="c", metadata={"obj": Opaque()}).to_dict()

    client.send([event])

    assert client.stats.sent_batches == 1
    assert b"Opaque" in bodies[0]


def test_metadata_that_cannot_even_be_stringified_is_dropped_quietly() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("nope")

        __repr__ = __str__

    handler = Responder(202)
    client = make_client(handler)
    event = Observation(trace_id="t", name="c", metadata={"bad": Hostile()}).to_dict()

    client.send([event])  # must not raise

    assert client.stats.sent_batches == 1
    assert b"unserialisable" in handler.calls[0].content


def test_missing_api_key_disables_the_client_without_opening_a_socket(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    handler = Responder(202)
    client = IngestClient(host="https://ingest.example", transport=httpx.MockTransport(handler))

    client.send(payload())

    assert client.enabled is False
    assert handler.calls == [], "a keyless client must not make requests"
    assert client.stats.dropped_events == 3
    # Config problems are the one thing we are loud about — silence here would
    # leave a user staring at an empty dashboard with no explanation.
    assert ENV_API_KEY in capsys.readouterr().err


def test_auth_rejection_is_reported_once_not_per_batch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    handler = Responder(401)
    client = make_client(handler)

    for _ in range(5):
        client.send(payload())

    assert client.stats.permanent_failures == 5
    assert capsys.readouterr().err.count("rejected the API key") == 1


def test_empty_batch_is_a_no_op() -> None:
    handler = Responder(202)
    client = make_client(handler)
    client.send([])
    assert handler.calls == []


# --------------------------------------------------------------------------- #
# Configuration resolution
# --------------------------------------------------------------------------- #


def test_env_vars_supply_config_and_explicit_args_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_API_KEY, "from-env")
    monkeypatch.setenv(ENV_HOST, "https://env.example/")

    from_env = IngestClient()
    assert from_env.api_key == "from-env"
    assert from_env.url == "https://env.example/v1/ingest", "trailing slash must be normalised"

    explicit = IngestClient(api_key="explicit", host="https://arg.example")
    assert explicit.api_key == "explicit"
    assert explicit.url == "https://arg.example/v1/ingest"


def test_every_request_carries_the_sdk_version_and_auth_header() -> None:
    handler = Responder(202)
    client = make_client(handler)

    client.send(payload())

    request = handler.calls[0]
    assert request.headers["X-SDK-Version"] == __version__
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.url.path == "/v1/ingest"
    assert request.method == "POST"


# --------------------------------------------------------------------------- #
# Deadlines — the interaction with the buffer's shutdown budget
# --------------------------------------------------------------------------- #


def test_retries_stop_at_the_deadline() -> None:
    """Backoff must not overrun the budget it was given.

    Without this, three attempts of exponential backoff eat a whole 5s atexit
    budget on one batch and the remaining batches are never even attempted.
    """
    handler = Responder(503)
    client = IngestClient(
        api_key="test-key",
        host="https://ingest.example",
        transport=httpx.MockTransport(handler),
        max_attempts=10,
        backoff_base=1.0,
        backoff_max=8.0,
    )

    deadline = Deadline()
    deadline.arm(0.3)
    started = time.monotonic()
    client.send(payload(), deadline=deadline)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"retries ran {elapsed:.2f}s past a 0.3s deadline"
    assert len(handler.calls) < 10, "should have given up before the attempt limit"
    assert client.stats.dropped_events == 3


def test_an_already_expired_deadline_skips_the_request_entirely() -> None:
    handler = Responder(202)
    client = make_client(handler)

    expired = Deadline()
    expired.arm(-1.0)
    client.send(payload(), deadline=expired)

    assert handler.calls == []
    assert client.stats.dropped_events == 3


def test_normal_flushes_carry_an_unarmed_deadline() -> None:
    """Nothing is waiting on the background thread, so it gets no budget."""
    seen: list[Deadline | None] = []

    def flush(_payload: list[dict[str, object]], deadline: Deadline | None = None) -> None:
        seen.append(deadline)

    buffer = EventBuffer(flush, flush_at=1, flush_interval=0.05, shutdown_timeout=2.0)
    try:
        buffer.add({"i": 0})
        time.sleep(0.3)
        assert len(seen) == 1
        assert seen[0] is not None
        assert seen[0].armed is False, "normal operation must not impose a budget"
        assert seen[0].remaining() is None
    finally:
        buffer.shutdown(timeout=1.0)


def test_the_shutdown_drain_carries_a_deadline() -> None:
    """Regression: the drain that delivers the final batches is the one that
    needs the budget.

    ``flush_at`` and ``flush_interval`` are both out of reach, so the only
    drain that can run is the shutdown one — no race with a normal flush.
    """
    seen: list[Deadline | None] = []

    def flush(_payload: list[dict[str, object]], deadline: Deadline | None = None) -> None:
        seen.append(deadline)

    buffer = EventBuffer(flush, flush_at=1_000, flush_interval=300.0, shutdown_timeout=2.0)
    buffer.add({"i": 0})
    assert seen == [], "nothing should have flushed yet"

    assert buffer.shutdown(timeout=2.0) is True

    assert len(seen) == 1
    deadline = seen[0]
    assert deadline is not None
    assert deadline.armed is True, "the shutdown drain must carry an armed deadline"
    remaining = deadline.remaining()
    assert remaining is not None
    # 90% of the join timeout, so the transport lands inside the join backstop.
    assert 0 < remaining <= 1.8


# --------------------------------------------------------------------------- #
# End to end: a dead ingest API behind a live buffer
# --------------------------------------------------------------------------- #


def test_host_app_is_unaffected_by_a_completely_dead_ingest_api() -> None:
    """The whole point, assembled: buffer + client + an API that only fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler, max_attempts=2)
    buffer = EventBuffer(
        client.send,
        max_size=500,
        flush_at=50,
        flush_interval=0.05,
        shutdown_timeout=1.0,
    )

    try:
        started = time.perf_counter()
        for i in range(2_000):
            buffer.add(Observation(trace_id="t", name=f"c{i}"))
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0, f"a dead ingest API slowed the caller to {elapsed:.3f}s"
        assert buffer.stats.queued == 2_000

        buffer.shutdown(timeout=1.0)

        assert client.stats.sent_batches == 0
        assert client.stats.dropped_events > 0
        assert buffer.stats.failed_batches == 0, (
            "client.send swallows its own errors, so no batch should escape to "
            "the buffer's catch-all"
        )
    finally:
        buffer.shutdown(timeout=0.5)
        client.close()


def test_flush_thread_survives_a_transport_that_always_raises() -> None:
    """A flush target that raises must not kill the thread — the next batch
    still gets attempted."""
    attempts = []

    def hostile(_payload: list[dict[str, object]], _deadline: Deadline | None = None) -> None:
        attempts.append(1)
        raise RuntimeError("boom")

    buffer = EventBuffer(hostile, flush_at=1, flush_interval=0.05, shutdown_timeout=1.0)
    try:
        for i in range(5):
            buffer.add({"i": i})
            time.sleep(0.08)

        assert len(attempts) >= 5, "flush thread died after the first exception"
        assert buffer.stats.failed_batches >= 5
    finally:
        buffer.shutdown(timeout=1.0)


def test_a_deadline_armed_mid_send_bounds_the_retries_already_under_way() -> None:
    """Regression: the flush thread is usually mid-batch when shutdown lands.

    Before the deadline became a live object, a send that started before
    ``shutdown()`` kept its original unlimited budget and retried straight past
    the join timeout — so the batches it was blocking were never attempted.
    """
    handler = Responder(503)
    client = IngestClient(
        api_key="test-key",
        host="https://ingest.example",
        transport=httpx.MockTransport(handler),
        max_attempts=10,
        backoff_base=30.0,  # one un-interrupted sleep would outlast the test
        backoff_max=60.0,
    )
    deadline = Deadline()

    # Arm it from another thread once the send is already retrying.
    threading.Timer(0.2, lambda: deadline.arm(0.1)).start()

    started = time.monotonic()
    client.send(payload(), deadline=deadline)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"send ran {elapsed:.2f}s — arming the deadline mid-flight had no effect"
    assert client.stats.dropped_events == 3


def test_arming_a_deadline_cuts_short_a_sleep_already_in_progress() -> None:
    """The unit-level version of the same property."""
    deadline = Deadline()
    threading.Timer(0.1, lambda: deadline.arm(0.0)).start()

    started = time.monotonic()
    deadline.sleep(30.0)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"sleep({30.0}) ignored the arming and ran {elapsed:.2f}s"


def test_shutdown_bounds_a_flush_thread_that_is_already_retrying() -> None:
    """End to end: the scenario that exposed the bug.

    Enough events to trip ``flush_at`` immediately, so the flush thread is deep
    in a retry sequence before ``shutdown()`` is ever called.
    """
    handler = Responder(503)
    client = IngestClient(
        api_key="test-key",
        host="https://ingest.example",
        transport=httpx.MockTransport(handler),
        max_attempts=10,
        backoff_base=1.0,
        backoff_max=8.0,
    )
    buffer = EventBuffer(
        client.send,
        max_size=1_000,
        flush_at=100,
        flush_interval=30.0,
        shutdown_timeout=1.0,
    )
    try:
        for i in range(500):
            buffer.add(Observation(trace_id="t", name=f"c{i}"))
        time.sleep(0.3)  # let the flush thread get into its retry loop
        assert handler.calls, "the flush thread should already be sending"

        started = time.monotonic()
        buffer.shutdown(timeout=1.0)
        elapsed = time.monotonic() - started

        assert elapsed <= 1.2, f"shutdown took {elapsed:.2f}s against a 1.0s budget"
        thread = buffer._thread
        assert thread is not None
        assert not thread.is_alive(), (
            "the flush thread outlived its budget — an in-flight send ignored the armed deadline"
        )
    finally:
        client.close()
