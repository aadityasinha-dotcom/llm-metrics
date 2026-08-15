"""Design rule 1: the SDK never blocks the caller.

The property under test is not "adding events is fast on a good day". It is
"adding events is fast *while the network is pathologically slow*", because
that is the situation in which an observability SDK actually takes down a
production service.

Every test here makes the flush target slow or permanently wedged on purpose,
then asserts the producing thread is unaffected.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llmobserve.buffer import Deadline, EventBuffer
from llmobserve.models import Observation

REPO_ROOT = Path(__file__).resolve().parents[1]

# 10k events is roughly a busy service's worth of traffic inside one flush
# interval. Thresholds are deliberately loose — they are 1-2 orders of
# magnitude away from a real regression, so they catch "we added a blocking
# call" without failing on a loaded CI box.
EVENT_COUNT = 10_000
MAX_TOTAL_SECONDS = 1.0
MAX_SINGLE_ADD_SECONDS = 0.05


class Recorder:
    """A flush target that just remembers what it was handed."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.deadlines: list[Deadline | None] = []

    def __call__(self, payload: list[dict[str, object]], deadline: Deadline | None = None) -> None:
        self.events.extend(payload)
        self.deadlines.append(deadline)

    def field(self, key: str) -> list[object]:
        return [e[key] for e in self.events]


class SlowFlush:
    """A flush target that sleeps, to stand in for a slow ingest API."""

    def __init__(self, delay: float = 0.25) -> None:
        self.delay = delay
        self.batches: list[int] = []
        self.calls = 0
        self.entered = threading.Event()

    def __call__(self, payload: list[dict[str, object]], deadline: Deadline | None = None) -> None:
        self.entered.set()
        self.calls += 1
        self.batches.append(len(payload))
        time.sleep(self.delay)


def make_event(i: int) -> Observation:
    return Observation(trace_id="t-1", name=f"call-{i}", metadata={"i": i})


@pytest.mark.slow
def test_adding_10k_events_does_not_block_when_flush_is_slow() -> None:
    """The headline guarantee.

    The flush thread is made ~25s slow in aggregate; the producer must still
    return in well under a second.
    """
    flush = SlowFlush(delay=0.25)
    buffer = EventBuffer(
        flush,
        max_size=1_000,
        flush_at=100,
        flush_interval=0.05,
        shutdown_timeout=0.1,
    )
    try:
        events = [make_event(i) for i in range(EVENT_COUNT)]  # build cost excluded

        worst = 0.0
        started = time.perf_counter()
        for event in events:
            call_start = time.perf_counter()
            buffer.add(event)
            worst = max(worst, time.perf_counter() - call_start)
        elapsed = time.perf_counter() - started

        # The property under test comes first, so a regression reports the
        # timing directly instead of tripping a guard below.
        assert elapsed < MAX_TOTAL_SECONDS, (
            f"{EVENT_COUNT} add() calls took {elapsed:.3f}s "
            f"(limit {MAX_TOTAL_SECONDS}s) — something in the hot path blocks"
        )
        assert worst < MAX_SINGLE_ADD_SECONDS, (
            f"slowest single add() took {worst * 1000:.1f}ms "
            f"(limit {MAX_SINGLE_ADD_SECONDS * 1000:.0f}ms)"
        )

        # Anti-vacuity: prove the slow path was real. The flush target ran, and
        # the queue overflowed — so the producer genuinely outran the flush
        # thread and the pressure was absorbed by dropping events rather than
        # by making the caller wait.
        assert flush.entered.wait(timeout=2.0), "flush thread never ran"
        assert buffer.stats.dropped > 0, (
            "the queue never overflowed, so the producer was never ahead of the "
            "flush thread — this test would also pass with a blocking add()"
        )
    finally:
        buffer.shutdown(timeout=0.1)


@pytest.mark.slow
def test_memory_stays_bounded_while_flush_is_slow() -> None:
    """Rule 4: the queue must not grow to absorb the backlog."""
    flush = SlowFlush(delay=0.25)
    buffer = EventBuffer(
        flush,
        max_size=500,
        flush_at=100,
        flush_interval=0.05,
        shutdown_timeout=0.1,
    )
    try:
        for i in range(EVENT_COUNT):
            buffer.add(make_event(i))

        assert len(buffer) <= 500
        assert buffer.stats.queued == EVENT_COUNT
        assert buffer.stats.dropped > 0, (
            "producer outran a 0.25s-per-batch flush without dropping anything"
        )
    finally:
        buffer.shutdown(timeout=0.1)


def test_overflow_drops_oldest_not_newest() -> None:
    """Under overload the newest events are the ones worth keeping."""
    sent = Recorder()
    buffer = EventBuffer(sent, max_size=3, flush_at=100, start=False)

    for i in range(10):
        buffer.add({"i": i})

    buffer.flush_once()

    assert sent.field("i") == [7, 8, 9]
    assert buffer.stats.dropped == 7


def test_concurrent_producers_are_not_serialised_behind_the_flush() -> None:
    """Eight threads adding at once, with a slow flush, and no lost writes."""
    flush = SlowFlush(delay=0.2)
    buffer = EventBuffer(
        flush,
        max_size=100_000,
        flush_at=100,
        flush_interval=0.05,
        shutdown_timeout=0.5,
    )
    per_thread = 1_000
    threads = 8
    errors: list[BaseException] = []
    durations: list[float] = []
    lock = threading.Lock()

    def produce(worker: int) -> None:
        try:
            start = time.perf_counter()
            for i in range(per_thread):
                buffer.add(make_event(worker * per_thread + i))
            taken = time.perf_counter() - start
        except BaseException as exc:  # noqa: BLE001 - surfaced via assert below
            with lock:
                errors.append(exc)
            return
        with lock:
            durations.append(taken)

    workers = [threading.Thread(target=produce, args=(w,)) for w in range(threads)]
    started = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=10)
    elapsed = time.perf_counter() - started

    try:
        assert not errors, f"add() raised on a producer thread: {errors[0]!r}"
        assert all(not w.is_alive() for w in workers), "a producer thread hung"
        assert buffer.stats.queued == threads * per_thread
        assert buffer.stats.dropped == 0, "max_size was large enough; nothing should drop"
        assert elapsed < MAX_TOTAL_SECONDS, (
            f"{threads} producers took {elapsed:.3f}s — contention or blocking"
        )
        assert max(durations) < MAX_TOTAL_SECONDS
    finally:
        buffer.shutdown(timeout=0.5)


def test_flush_failures_never_reach_the_caller() -> None:
    """Rule 2: a broken ingest API drops events, it does not raise."""

    def explode(_payload: list[dict[str, object]], _deadline: Deadline | None = None) -> None:
        raise RuntimeError("ingest is down")

    buffer = EventBuffer(explode, max_size=100, flush_at=10, start=False)
    for i in range(30):
        buffer.add({"i": i})

    buffer.flush_once()  # must not propagate

    assert buffer.stats.failed_batches == 3
    assert buffer.stats.flushed == 0
    assert len(buffer) == 0


def test_add_after_shutdown_is_a_no_op_not_an_error() -> None:
    sent = Recorder()
    buffer = EventBuffer(sent, flush_at=1, flush_interval=0.05)
    buffer.add({"i": 0})
    assert buffer.shutdown(timeout=2.0) is True

    buffer.add({"i": 1})  # late event, e.g. from another atexit hook

    assert sent.events == [{"i": 0}]
    assert buffer.stats.dropped == 1


# --------------------------------------------------------------------------- #
# Shutdown: delivery vs. hanging on exit
# --------------------------------------------------------------------------- #


def test_shutdown_flushes_buffered_events() -> None:
    """The reason atexit exists at all: a short script must still deliver."""
    sent = Recorder()
    buffer = EventBuffer(sent, flush_at=1_000, flush_interval=30.0)

    for i in range(5):
        buffer.add({"i": i})

    assert sent.events == [], "nothing should have flushed yet"
    assert buffer.shutdown(timeout=2.0) is True
    assert sent.field("i") == [0, 1, 2, 3, 4]


def test_shutdown_gives_up_on_a_wedged_flush_instead_of_hanging() -> None:
    """The other half of the tradeoff: delivery is best-effort, exit is not.

    The flush target blocks forever. ``shutdown`` must return ``False`` at its
    deadline rather than waiting on it — the daemon thread is abandoned and
    the interpreter is free to exit.
    """
    release = threading.Event()

    def wedged(_payload: list[dict[str, object]], _deadline: Deadline | None = None) -> None:
        release.wait()  # never set until teardown

    buffer = EventBuffer(wedged, flush_at=1, flush_interval=0.05)
    buffer.add({"i": 0})

    started = time.perf_counter()
    delivered = buffer.shutdown(timeout=0.3)
    elapsed = time.perf_counter() - started

    try:
        assert delivered is False, "a wedged flush must be reported as undelivered"
        assert 0.3 <= elapsed < 1.0, (
            f"shutdown took {elapsed:.3f}s — the deadline is not being honoured"
        )
    finally:
        release.set()


# --------------------------------------------------------------------------- #
# Real interpreter exit. atexit + daemon-thread interaction cannot be faked
# in-process, so these run a child interpreter and inspect what it delivered.
# --------------------------------------------------------------------------- #


def run_child(body: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


ATEXIT_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from llmobserve.buffer import Deadline, EventBuffer

def flush(payload, deadline=None):
    print("FLUSHED", len(payload), flush=True)

# flush_at and flush_interval are both far out of reach: the only thing that
# can deliver these events is the atexit hook.
buf = EventBuffer(flush, flush_at=1000, flush_interval=300.0)
for i in range(7):
    buf.add({{"i": i}})
print("EXITING", flush=True)
"""


def test_atexit_flushes_events_a_short_script_would_otherwise_lose() -> None:
    result = run_child(ATEXIT_SCRIPT.format(root=str(REPO_ROOT)))

    assert result.returncode == 0, result.stderr
    assert "EXITING" in result.stdout
    assert "FLUSHED 7" in result.stdout, (
        f"atexit did not deliver the buffered events; stdout={result.stdout!r}"
    )


WEDGED_EXIT_SCRIPT = """
import sys, threading
sys.path.insert(0, {root!r})
from llmobserve.buffer import Deadline, EventBuffer

def flush(payload, deadline=None):
    threading.Event().wait()  # blocks forever

buf = EventBuffer(flush, flush_at=1, flush_interval=0.05, shutdown_timeout=1.0)
buf.add({{"i": 0}})
print("EXITING", flush=True)
"""


def test_interpreter_exit_is_bounded_when_the_ingest_api_hangs() -> None:
    """A hung ingest API may delay exit by ``shutdown_timeout`` — no more.

    This is the failure mode that makes naive atexit flushing dangerous: without
    a deadline this child would never exit and would hang the user's CI job.
    """
    started = time.perf_counter()
    result = run_child(WEDGED_EXIT_SCRIPT.format(root=str(REPO_ROOT)), timeout=20.0)
    elapsed = time.perf_counter() - started

    assert result.returncode == 0, result.stderr
    assert "EXITING" in result.stdout
    assert elapsed < 10.0, (
        f"child took {elapsed:.1f}s to exit with a wedged flush target — "
        "the shutdown deadline is not bounding interpreter exit"
    )
