"""Ambient trace state, held in :mod:`contextvars`.

Nesting works by keeping two things current: the :class:`~llm_metrics.models.Trace`
a call belongs to, and the id of the observation it sits under. A new
observation reads both, records itself as the new parent for the duration of
its body, and puts the old parent back on the way out. That is the whole
mechanism — the tree shape falls out of the stack discipline.

Why contextvars and not thread-locals
-------------------------------------
``asyncio`` copies the context when a task is created, which gives exactly the
semantics tracing wants and a thread-local cannot express:

* a child task sees its parent's trace, so nesting works across ``await``
* a child's changes do **not** leak back to the parent when it finishes
* concurrent sibling tasks cannot see each other's observations, so
  ``asyncio.gather`` of five LLM calls produces five siblings rather than an
  accidental chain

The thread trap
---------------
The flip side, and it is sharp: **a new thread inherits nothing.** Neither
``threading.Thread`` nor ``ThreadPoolExecutor.submit`` copies the context, so
work handed to a pool starts with no trace and its observations become orphan
roots. This is the common shape for fanning out LLM calls, so it is worth
knowing about before it produces a confusing dashboard.

Two ways to carry the context across that boundary::

    # stdlib, when you control the call site
    ctx = contextvars.copy_context()
    executor.submit(ctx.run, do_work, arg)

    # this module, when you do not — e.g. across a queue
    snap = context.snapshot()
    ...
    with context.adopt(snap):
        do_work(arg)
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from llm_metrics.models import Observation, Trace

__all__ = [
    "ContextSnapshot",
    "adopt",
    "current_observation",
    "current_parent_id",
    "current_trace",
    "current_trace_id",
    "snapshot",
    "use_observation",
    "use_trace",
]

# Module level, created once. A ContextVar built per call would be a fresh
# variable each time and would never see anything set by an earlier one.
_TRACE: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "llm_metrics_trace", default=None
)
_PARENT_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_metrics_parent_id", default=None
)
# The live object behind _PARENT_ID, so code inside a traced call can annotate
# the observation it is running under. Kept separate rather than replacing the
# id: integrations that nest by explicit id (LangChain) never have the object.
_OBSERVATION: contextvars.ContextVar[Observation | None] = contextvars.ContextVar(
    "llm_metrics_observation", default=None
)


# ------------------------------------------------------------------ readers


def current_trace() -> Trace | None:
    """The trace this call belongs to, or ``None`` outside any trace.

    Returns the live object, so a caller holding it can still call
    :meth:`~llm_metrics.models.Trace.end`.
    """
    return _TRACE.get()


def current_trace_id() -> str | None:
    trace = _TRACE.get()
    return trace.id if trace is not None else None


def current_parent_id() -> str | None:
    """The observation to nest under, or ``None`` at the root of a trace."""
    return _PARENT_ID.get()


def current_observation() -> Observation | None:
    """The observation the caller is running inside, or ``None``.

    Returns the live object, so annotations made through it land on the event
    that is eventually emitted.
    """
    return _OBSERVATION.get()


# ------------------------------------------------------------------ scopes


@contextmanager
def use_trace(trace: Trace) -> Iterator[Trace]:
    """Make ``trace`` current for the duration of the block.

    The parent id is cleared as well: a new trace starts a fresh spine, so
    whatever observation happened to be open outside must not adopt the first
    observation inside.
    """
    trace_token = _TRACE.set(trace)
    parent_token = _PARENT_ID.set(None)
    observation_token = _OBSERVATION.set(None)
    try:
        yield trace
    finally:
        # Reset in reverse order of set. Tokens are only valid in the context
        # that produced them, which is why these are context managers rather
        # than a set/reset pair a caller could accidentally split across an
        # await or a thread.
        _OBSERVATION.reset(observation_token)
        _PARENT_ID.reset(parent_token)
        _TRACE.reset(trace_token)


@contextmanager
def use_observation(observation: Observation) -> Iterator[Observation]:
    """Nest everything in the block under ``observation``.

    Restores the previous parent on the way out, including when the body
    raises — an observation that fails still has to close its scope, or every
    later sibling in the trace would hang off it.
    """
    token = _PARENT_ID.set(observation.id)
    observation_token = _OBSERVATION.set(observation)
    try:
        yield observation
    finally:
        _OBSERVATION.reset(observation_token)
        _PARENT_ID.reset(token)


# ----------------------------------------------------- crossing boundaries


@dataclass(frozen=True)
class ContextSnapshot:
    """A copy of the ambient state, for carrying into a thread or a worker."""

    trace: Trace | None = None
    parent_id: str | None = None
    observation: Observation | None = None

    @property
    def empty(self) -> bool:
        return self.trace is None and self.parent_id is None


def snapshot() -> ContextSnapshot:
    """Capture the current trace and parent so another thread can adopt them."""
    return ContextSnapshot(
        trace=_TRACE.get(), parent_id=_PARENT_ID.get(), observation=_OBSERVATION.get()
    )


@contextmanager
def adopt(snap: ContextSnapshot) -> Iterator[None]:
    """Install a snapshot for the duration of the block.

    Use in a worker thread, which starts with no context of its own. Adopting
    an empty snapshot is also the clean way to detach — useful in tests, and
    for background work that should not be attributed to whatever request
    happened to start it.
    """
    trace_token = _TRACE.set(snap.trace)
    parent_token = _PARENT_ID.set(snap.parent_id)
    observation_token = _OBSERVATION.set(snap.observation)
    try:
        yield
    finally:
        _OBSERVATION.reset(observation_token)
        _PARENT_ID.reset(parent_token)
        _TRACE.reset(trace_token)
