"""Trace nesting via contextvars.

The property under test is the shape of the resulting tree: every observation
must hang off the one that was open when it started, and closing an
observation must put the previous parent back — including when the body
raises, and including when siblings run concurrently.

Async and threads are covered separately because their propagation rules
differ, and the difference is the thing most likely to produce a wrong tree.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_metrics import context
from llm_metrics.context import (
    ContextSnapshot,
    adopt,
    current_parent_id,
    current_trace,
    current_trace_id,
    snapshot,
    use_observation,
    use_trace,
)
from llm_metrics.models import Observation, Trace


@pytest.fixture(autouse=True)
def _detached():
    """Every test starts with no ambient trace, whatever ran before it."""
    with adopt(ContextSnapshot()):
        yield


def obs(name: str) -> Observation:
    return Observation(trace_id=current_trace_id() or "orphan", name=name)


# --------------------------------------------------------------------------- #
# Basics
# --------------------------------------------------------------------------- #


def test_nothing_is_current_outside_a_trace() -> None:
    assert current_trace() is None
    assert current_trace_id() is None
    assert current_parent_id() is None


def test_use_trace_sets_and_restores() -> None:
    trace = Trace(name="request")

    with use_trace(trace) as active:
        assert active is trace
        assert current_trace() is trace
        assert current_trace_id() == trace.id

    assert current_trace() is None, "the trace must not outlive its block"


def test_current_trace_returns_the_live_object() -> None:
    """The decorator needs to end() the trace it found."""
    trace = Trace(name="request")
    with use_trace(trace):
        found = current_trace()
        assert found is not None
        found.end()
    assert trace.end_time is not None


def test_a_new_trace_starts_a_fresh_spine() -> None:
    """An open observation outside must not adopt the first one inside."""
    outer = Trace(name="outer")
    with use_trace(outer), use_observation(obs("outer-span")):
        assert current_parent_id() is not None

        with use_trace(Trace(name="inner")):
            assert current_parent_id() is None, (
                "a nested trace inherited a parent from the enclosing trace"
            )


# --------------------------------------------------------------------------- #
# Tree shape
# --------------------------------------------------------------------------- #


def test_observations_nest_into_a_parent_chain() -> None:
    trace = Trace(name="request")
    with use_trace(trace):
        assert current_parent_id() is None, "the first observation is a root"

        a = obs("a")
        with use_observation(a):
            assert current_parent_id() == a.id

            b = Observation(trace_id=trace.id, name="b", parent_id=current_parent_id())
            with use_observation(b):
                assert current_parent_id() == b.id

                c = Observation(trace_id=trace.id, name="c", parent_id=current_parent_id())
                assert c.parent_id == b.id

            assert current_parent_id() == a.id, "closing b must restore a"
        assert current_parent_id() is None, "closing a must restore the root"

        assert b.parent_id == a.id
        assert a.parent_id is None


def test_siblings_share_a_parent_rather_than_chaining() -> None:
    trace = Trace(name="request")
    with use_trace(trace):
        parent = obs("parent")
        with use_observation(parent):
            children = []
            for i in range(3):
                child = Observation(
                    trace_id=trace.id, name=f"child-{i}", parent_id=current_parent_id()
                )
                with use_observation(child):
                    pass
                children.append(child)

        assert [c.parent_id for c in children] == [parent.id] * 3


def test_an_observation_that_raises_still_closes_its_scope() -> None:
    """Otherwise every later sibling hangs off the failed call."""
    trace = Trace(name="request")
    with use_trace(trace):
        parent = obs("parent")
        with use_observation(parent):
            failing = Observation(trace_id=trace.id, name="boom", parent_id=current_parent_id())

            with pytest.raises(RuntimeError, match="LLM call failed"), use_observation(failing):
                raise RuntimeError("the LLM call failed")

            assert current_parent_id() == parent.id, "a raising body leaked its scope"


def test_a_raising_trace_body_still_restores() -> None:
    with pytest.raises(ValueError, match="handler failed"), use_trace(Trace(name="request")):
        raise ValueError("handler failed")

    assert current_trace() is None


# --------------------------------------------------------------------------- #
# asyncio — where contextvars earn their keep
# --------------------------------------------------------------------------- #


def test_async_children_nest_under_the_awaiting_parent() -> None:
    trace = Trace(name="request")
    seen: dict[str, str | None] = {}

    async def child() -> None:
        seen["child_trace"] = current_trace_id()
        seen["child_parent"] = current_parent_id()

    async def main() -> None:
        with use_trace(trace):
            parent = obs("parent")
            with use_observation(parent):
                await child()
            seen["after"] = current_parent_id()

    asyncio.run(main())

    assert seen["child_trace"] == trace.id
    assert seen["child_parent"] is not None
    assert seen["after"] is None


def test_concurrent_tasks_are_siblings_not_a_chain() -> None:
    """The reason this module is not built on thread-locals.

    ``gather`` of N calls must produce N siblings. A shared mutable parent
    would let whichever task started last capture the others.
    """
    trace = Trace(name="request")
    parents: list[str | None] = []

    async def leaf(i: int) -> None:
        own = Observation(trace_id=trace.id, name=f"leaf-{i}", parent_id=current_parent_id())
        with use_observation(own):
            await asyncio.sleep(0.01 * (3 - i))  # finish out of order on purpose
        parents.append(own.parent_id)

    async def main() -> None:
        with use_trace(trace):
            root = obs("root")
            with use_observation(root):
                await asyncio.gather(*(leaf(i) for i in range(3)))
                assert current_parent_id() == root.id, "a task leaked into its parent"
            parents.append(root.id)

    asyncio.run(main())

    root_id = parents.pop()
    assert parents == [root_id] * 3, f"tasks did not stay siblings: {parents}"


def test_a_task_cannot_see_a_sibling_task_context() -> None:
    trace = Trace(name="request")
    observed: list[str | None] = []

    async def setter() -> None:
        with use_observation(obs("setter")):
            await asyncio.sleep(0.02)

    async def peeker() -> None:
        await asyncio.sleep(0.01)  # while setter is inside its scope
        observed.append(current_parent_id())

    async def main() -> None:
        with use_trace(trace):
            await asyncio.gather(setter(), peeker())

    asyncio.run(main())

    assert observed == [None], "one task saw another task's open observation"


# --------------------------------------------------------------------------- #
# Threads — where the trap is
# --------------------------------------------------------------------------- #


def test_a_plain_thread_inherits_nothing() -> None:
    """Documents the trap rather than pretending it does not exist.

    A worker thread starts with an empty context, so its observations become
    orphan roots unless the context is carried across explicitly.
    """
    trace = Trace(name="request")
    seen: list[str | None] = []

    with use_trace(trace), use_observation(obs("parent")):
        thread = threading.Thread(target=lambda: seen.append(current_trace_id()))
        thread.start()
        thread.join()

    assert seen == [None], (
        "threads now inherit context — the docs and the adopt() helper need updating"
    )


def test_thread_pool_workers_inherit_nothing_either() -> None:
    trace = Trace(name="request")

    with use_trace(trace), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(current_trace_id).result() is None


def test_copy_context_carries_the_trace_into_a_worker() -> None:
    """The stdlib fix, for when you control the call site."""
    trace = Trace(name="request")

    with use_trace(trace):
        parent = obs("parent")
        with use_observation(parent), ThreadPoolExecutor(max_workers=1) as pool:
            ctx = contextvars.copy_context()
            assert pool.submit(ctx.run, current_trace_id).result() == trace.id
            assert pool.submit(ctx.run, current_parent_id).result() == parent.id


def test_snapshot_and_adopt_carry_the_trace_across_a_thread() -> None:
    """This module's fix, for when you do not control the call site."""
    trace = Trace(name="request")
    result: dict[str, str | None] = {}

    def worker(snap: ContextSnapshot) -> None:
        assert current_trace_id() is None, "worker should start detached"
        with adopt(snap):
            result["trace"] = current_trace_id()
            result["parent"] = current_parent_id()
        assert current_trace_id() is None, "adopt must not outlive its block"

    with use_trace(trace):
        parent = obs("parent")
        with use_observation(parent):
            snap = snapshot()

        thread = threading.Thread(target=worker, args=(snap,))
        thread.start()
        thread.join()

    assert result == {"trace": trace.id, "parent": parent.id}


def test_adopting_an_empty_snapshot_detaches() -> None:
    """Background work should not be attributed to whatever request began it."""
    with use_trace(Trace(name="request")), use_observation(obs("parent")):
        with adopt(ContextSnapshot()):
            assert current_trace() is None
            assert current_parent_id() is None
        assert current_trace() is not None, "detaching must be scoped"


def test_snapshot_reports_emptiness() -> None:
    assert snapshot().empty is True
    with use_trace(Trace(name="request")):
        assert snapshot().empty is False


def test_context_vars_are_module_level_singletons() -> None:
    """A ContextVar built per call would silently never see anything.

    Guards against someone "tidying" the module-level vars into a factory.
    """
    assert context._TRACE is context._TRACE
    trace = Trace(name="request")
    with use_trace(trace):
        assert context._TRACE.get() is trace
