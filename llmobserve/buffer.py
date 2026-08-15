"""Bounded, thread-safe event buffer with a background flush thread.

This module exists to satisfy design rules 1 and 4: the caller is never blocked,
and memory is bounded. Everything here is written around one invariant:

    **No lock is ever held while the flush target runs.**

``add()`` does three cheap things under a lock — append to a ``deque``, bump a
counter, maybe set an ``Event`` — and never touches the network. The flush
thread copies a batch out under the same lock, releases it, and only then calls
out to the transport. A flush target that takes ten seconds therefore cannot
make ``add()`` take longer than a few microseconds.

Overflow is handled by ``deque(maxlen=...)``, which discards from the left on
append. Dropping the *oldest* event is the deliberate choice: under sustained
overload the newest events describe what is happening right now, and a
producer that outruns the network should degrade by losing history, not by
growing until the host process is OOM-killed.

See ``EventBuffer.shutdown`` for the atexit / hang tradeoff.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Protocol, Union, runtime_checkable

__all__ = ["BufferStats", "Deadline", "EventBuffer", "SupportsToDict"]

DEFAULT_MAX_SIZE = 10_000
DEFAULT_FLUSH_AT = 100
DEFAULT_FLUSH_INTERVAL = 5.0
DEFAULT_SHUTDOWN_TIMEOUT = 5.0


@runtime_checkable
class SupportsToDict(Protocol):
    """Anything the buffer can serialise — :class:`~llmobserve.models.Trace`,
    :class:`~llmobserve.models.Observation`, or a user-supplied equivalent."""

    def to_dict(self) -> dict[str, Any]: ...


Event = Union[SupportsToDict, "dict[str, Any]"]


class Deadline:
    """A time budget that can be armed *after* it has been handed out.

    A plain ``float`` deadline captured when a send begins is not enough. The
    flush thread is usually already mid-batch when shutdown arrives, and that
    in-flight send would keep retrying with the unbounded budget it was given
    at start — sailing straight past the join timeout while the events it was
    supposed to make room for are never even attempted.

    So the buffer hands out *this object*, once, and arms it when shutdown
    begins. A target that re-reads :meth:`remaining` between attempts, and
    sleeps via :meth:`sleep`, notices the budget the moment it appears.
    """

    __slots__ = ("_armed", "_at")

    def __init__(self) -> None:
        self._at: float | None = None
        self._armed = threading.Event()

    @property
    def armed(self) -> bool:
        return self._armed.is_set()

    def arm(self, seconds: float) -> None:
        """Start the clock. Wakes anything sleeping in :meth:`sleep`."""
        self._at = time.monotonic() + seconds
        self._armed.set()

    def remaining(self) -> float | None:
        """Seconds left, or ``None`` while unarmed (meaning: no limit)."""
        at = self._at
        return None if at is None else at - time.monotonic()

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0

    def sleep(self, seconds: float) -> None:
        """Sleep, but never past the budget and never through its arming.

        While unarmed this blocks on the arming event, so a backoff sleep that
        started before shutdown is cut short the instant shutdown lands rather
        than burning the whole budget it was supposed to respect.
        """
        if not self._armed.is_set():
            self._armed.wait(seconds)
            return
        remaining = self.remaining()
        if remaining is not None:
            seconds = min(seconds, remaining)
        if seconds > 0:
            time.sleep(seconds)


#: Flush target: ``(payload, deadline) -> None``.
#:
#: ``deadline`` is live — unarmed during normal operation, armed by
#: :meth:`EventBuffer.shutdown`. Targets should consult it between retries. The
#: target is expected to swallow its own errors; the buffer catches anything
#: that escapes anyway.
FlushFn = Callable[["list[dict[str, Any]]", Deadline], None]


class BufferStats:
    """Counters for introspection and tests. Plain attributes, read without a
    lock — they are advisory, not transactional."""

    __slots__ = ("dropped", "failed_batches", "flushed", "queued")

    def __init__(self) -> None:
        self.queued = 0
        self.flushed = 0
        self.dropped = 0
        self.failed_batches = 0

    def __repr__(self) -> str:
        return (
            f"BufferStats(queued={self.queued}, flushed={self.flushed}, "
            f"dropped={self.dropped}, failed_batches={self.failed_batches})"
        )


class EventBuffer:
    """Collect events from any thread and hand them to ``flush_fn`` in batches.

    Args:
        flush_fn: Called on the flush thread as ``flush_fn(payload, deadline)``.
            It may block, raise, or take arbitrarily long — none of that
            reaches the caller of :meth:`add`. Exceptions are swallowed and
            counted. ``deadline`` is unarmed except during the shutdown drain.
        max_size: Hard cap on buffered events. Appending past it discards the
            oldest event.
        flush_at: Wake the flush thread as soon as this many events are queued.
        flush_interval: Wake the flush thread at least this often, in seconds,
            so low-traffic apps still deliver promptly.
        shutdown_timeout: Ceiling, in seconds, on how long interpreter exit may
            be delayed by a final flush.
        start: Start the flush thread immediately. Tests set this to ``False``
            to drive :meth:`flush_once` by hand.
    """

    def __init__(
        self,
        flush_fn: FlushFn,
        *,
        max_size: int = DEFAULT_MAX_SIZE,
        flush_at: int = DEFAULT_FLUSH_AT,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        start: bool = True,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if flush_at < 1:
            raise ValueError("flush_at must be >= 1")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be > 0")

        self._flush_fn = flush_fn
        self._max_size = max_size
        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._shutdown_timeout = shutdown_timeout

        # maxlen gives us "drop oldest on overflow" for free, in C, with no
        # branch in the hot path.
        self._queue: deque[Event] = deque(maxlen=max_size)
        self._lock = threading.Lock()

        # _wake: "there is work, or we are shutting down" — lets add() cut the
        # interval short. _shutdown: one-way latch, never cleared.
        self._wake = threading.Event()
        self._shutdown = threading.Event()

        self._thread: threading.Thread | None = None
        self._atexit_hook: Callable[[], None] | None = None
        self._fork_registered = False

        # Handed to every flush_fn call, armed once by shutdown(). Shared
        # rather than per-batch precisely so an in-flight send sees the arming.
        self._deadline = Deadline()

        self.stats = BufferStats()

        if start:
            self.start()

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the flush thread. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._shutdown.is_set():
                raise RuntimeError("EventBuffer has been shut down")
            thread = threading.Thread(
                target=self._run,
                name="llmobserve-flush",
                daemon=True,
            )
            self._thread = thread

        # Registered outside the lock: atexit.register takes its own lock, and
        # nesting two locks in different orders elsewhere is how deadlocks start.
        if self._atexit_hook is None:
            hook = self._on_interpreter_exit
            atexit.register(hook)
            self._atexit_hook = hook

        self._register_fork_handler()
        thread.start()

    def _register_fork_handler(self) -> None:
        """Reset in forked children.

        A child inherits a *copy* of the buffer but none of the parent's
        threads. Without this the child would re-send every event the parent
        had queued at fork time (duplicates on the dashboard) and would never
        start a flush thread of its own — which is exactly the
        gunicorn/uvicorn ``--preload`` deployment.
        """
        if self._fork_registered or not hasattr(os, "register_at_fork"):
            return
        os.register_at_fork(after_in_child=self._reset_after_fork)
        self._fork_registered = True

    def _reset_after_fork(self) -> None:  # pragma: no cover - requires fork
        # The inherited lock may have been held by a thread that does not exist
        # in this process, so replace it rather than trying to acquire it.
        self._lock = threading.Lock()
        self._queue = deque(maxlen=self._max_size)
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._deadline = Deadline()
        self._thread = None
        self.stats = BufferStats()
        self.start()

    # ------------------------------------------------------------------ producer

    def add(self, event: Event) -> None:
        """Queue an event. Never blocks on I/O, never raises.

        This is the hot path — it runs inside the user's request handler. The
        lock is held for a couple of ``deque`` operations and nothing else.
        """
        try:
            with self._lock:
                if self._shutdown.is_set():
                    self.stats.dropped += 1
                    return
                overflowed = len(self._queue) == self._max_size
                self._queue.append(event)
                self.stats.queued += 1
                if overflowed:
                    self.stats.dropped += 1
                should_wake = len(self._queue) >= self._flush_at
            if should_wake:
                self._wake.set()
        except Exception:  # noqa: BLE001 - rule 2: never crash the host app
            pass

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    # ------------------------------------------------------------------ consumer

    def _run(self) -> None:
        """Flush thread main loop.

        The shutdown check happens *before* the drain, not after. shutdown()
        sets ``_wake`` as well as ``_shutdown``, so the very wake-up that ends
        the wait is usually the one that has to deliver the final batches — if
        this drained first and checked afterwards, the last drain would run
        with no deadline and the trailing one would find an empty queue.
        """
        while True:
            # Returns early when add() trips flush_at, otherwise on interval.
            self._wake.wait(self._flush_interval)
            self._wake.clear()

            shutting_down = self._shutdown.is_set()
            self._drain()
            if shutting_down:
                return

    def _drain(self) -> None:
        """Send every queued event, in batches of ``flush_at``.

        Loops until the queue is empty rather than sending one batch per wake,
        so a burst of 5000 events does not take 250 seconds to clear at a
        5-second interval.
        """
        while True:
            if self._deadline.expired():
                # Out of time. Whatever is still queued is abandoned; the join()
                # in shutdown() is about to give up on us anyway, and reporting
                # that honestly beats blocking the interpreter.
                return
            batch = self._take_batch()
            if not batch:
                return
            self._send(batch)

    def _take_batch(self) -> list[Event]:
        with self._lock:
            if not self._queue:
                return []
            n = min(self._flush_at, len(self._queue))
            return [self._queue.popleft() for _ in range(n)]

    def _send(self, batch: list[Event]) -> None:
        """Serialise and hand off. Called with no lock held."""
        try:
            payload = [e if isinstance(e, dict) else e.to_dict() for e in batch]
        except Exception:  # noqa: BLE001 - a bad event must not kill the thread
            self.stats.failed_batches += 1
            return
        try:
            self._flush_fn(payload, self._deadline)
            self.stats.flushed += len(payload)
        except Exception:  # noqa: BLE001 - rule 2: drop events, keep running
            self.stats.failed_batches += 1

    def flush_once(self) -> None:
        """Drain the buffer on the *calling* thread.

        Only for tests and for callers who have explicitly opted into blocking.
        The SDK's own code paths never call this.
        """
        self._drain()

    # ------------------------------------------------------------------ shutdown

    def shutdown(self, timeout: float | None = None) -> bool:
        """Stop accepting events, flush what is queued, and join the thread.

        The tradeoff, stated plainly: the flush thread is a daemon, so without
        this method every buffered event is lost when the interpreter exits —
        fatal for short scripts, which is most first-run experiences. With it,
        exit waits for delivery. To keep that wait bounded, the actual flushing
        stays on the daemon thread and this method only *joins* with a
        deadline. A ``join`` timeout is a real ceiling: if the thread is wedged
        in a socket read we abandon it and let the interpreter kill it. Flushing
        inline here instead would leave no way to walk away.

        Returns:
            ``True`` if the flush thread finished within the deadline, ``False``
            if it was abandoned with events still queued.
        """
        if timeout is None:
            timeout = self._shutdown_timeout

        # Unregister first, and unconditionally. If we only did it on success,
        # a shutdown() that timed out would leave the hook armed and the
        # interpreter would pay the same timeout a second time on exit.
        hook, self._atexit_hook = self._atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)

        # Arm *before* signalling. The deadline is shared and live, so this
        # also bounds a send that is already in flight. Aim to finish at 90% of
        # the join timeout: the transport gets a budget it can land inside, and
        # the join stays a hard backstop rather than a race.
        self._deadline.arm(timeout * 0.9)
        self._shutdown.set()
        self._wake.set()

        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        if thread is threading.current_thread():
            # flush_fn called shutdown() on itself; joining would deadlock.
            return False

        thread.join(timeout)
        return not thread.is_alive()

    def _on_interpreter_exit(self) -> None:
        """atexit hook. Runs on the main thread, before daemon threads die."""
        try:
            delivered = self.shutdown()
        except Exception:  # noqa: BLE001 - never turn exit into a traceback
            return
        if not delivered and os.environ.get("LLMOBSERVE_DEBUG"):
            remaining = len(self._queue)
            print(
                f"llmobserve: abandoned {remaining} event(s) after "
                f"{self._shutdown_timeout}s shutdown timeout",
                file=sys.stderr,
            )

    # -------------------------------------------------------------- context mgr

    def __enter__(self) -> EventBuffer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
