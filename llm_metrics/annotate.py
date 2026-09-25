"""Attach who, where, and how-good to traces from inside the code that runs them.

    @observe()
    def answer(user, question):
        update_trace(user_id=user.id, session_id=user.session, tags=["qa"])
        ...

    score("thumbs", value=True, comment="helpful")

Token ledgers know which API key spent what. They cannot say which *feature*,
*tenant*, or *prompt version* spent it, or whether the answer was any good.
These three functions supply exactly that, and they are the difference between
a bill and an observability tool.

All of them read the ambient context, so they work anywhere inside an
``@observe`` call — including from a LangChain tool, if the tool itself is
decorated. None of them raises and none of them blocks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from llm_metrics import _runtime, context
from llm_metrics.models import Score, ScoreSource

__all__ = ["score", "update_observation", "update_trace"]

#: "Argument not given" sentinel, so ``None`` can mean "clear this".
_UNSET: Any = object()


def update_trace(
    *,
    name: str | None = None,
    user_id: str | None = _UNSET,
    session_id: str | None = _UNSET,
    tags: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Annotate the trace the caller is running inside.

    Returns ``False`` if there is no trace open, which is the normal result
    outside any traced call and never an error.

    Args:
        name: Rename the trace. Useful when the root function is generic and
            the meaningful name is only known once the request is parsed.
        user_id: The end user this request served. ``None`` clears it.
        session_id: Groups the traces of one conversation. ``None`` clears it.
        tags: Added to the trace's tags. Duplicates are ignored.
        metadata: Merged into the trace's metadata; later keys win.
    """
    try:
        trace = context.current_trace()
        if trace is None:
            return False
        if name:
            trace.name = name
        if user_id is not _UNSET:
            trace.user_id = user_id
        if session_id is not _UNSET:
            trace.session_id = session_id
        if tags:
            for tag in tags:
                tag = str(tag)
                if tag not in trace.tags:
                    trace.tags.append(tag)
        if metadata:
            trace.metadata.update(metadata)
        return True
    except Exception:  # noqa: BLE001 - rule 2
        return False


def update_observation(
    *,
    name: str | None = None,
    model: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Annotate the observation the caller is running inside.

    The main use is hand-rolled instrumentation: a function decorated
    ``@observe(as_type="generation")`` that calls a provider the SDK has no
    wrapper for can still report the model and token counts. Returns ``False``
    when nothing is open.

    Args:
        prompt_name: Which prompt template this call used, e.g. ``"qa"``.
        prompt_version: Its version, so quality can be compared across
            versions. Stored as a string.
    """
    try:
        observation = context.current_observation()
        if observation is None:
            return False
        if name:
            observation.name = name
        if model is not None:
            observation.model = model
        if prompt_name is not None:
            observation.prompt_name = prompt_name
        if prompt_version is not None:
            observation.prompt_version = str(prompt_version)
        if prompt_tokens is not None:
            observation.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            observation.completion_tokens = completion_tokens
        if cached_tokens is not None:
            observation.cached_tokens = cached_tokens
        if reasoning_tokens is not None:
            observation.reasoning_tokens = reasoning_tokens
        if metadata:
            observation.metadata.update(metadata)
        return True
    except Exception:  # noqa: BLE001 - rule 2
        return False


def score(
    name: str,
    value: float | int | bool | str,
    *,
    trace_id: str | None = None,
    observation_id: str | None = None,
    comment: str | None = None,
    source: str = ScoreSource.HUMAN,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    """Record a judgement about a trace or observation.

    With no ids given, the score attaches to the ambient trace and, if the
    caller is inside one, the ambient observation. Pass ``trace_id`` to score
    a request after the fact — say, when the user clicks thumbs-down a minute
    later — using the id you stored from :func:`context.current_trace_id`.

    Args:
        name: What is being measured: ``"thumbs"``, ``"faithfulness"``, ...
        value: A number, a boolean, or a short category label.
        source: One of :class:`~llm_metrics.models.ScoreSource`.

    Returns:
        The score's id, or ``None`` if there was nothing to attach it to or
        the SDK is disabled.
    """
    try:
        if not _runtime.is_enabled():
            return None
        if trace_id is None and observation_id is None:
            trace = context.current_trace()
            if trace is None:
                return None
            if not trace.sampled:
                return None  # the events it would point at were never sent
            trace_id = trace.id
            observation = context.current_observation()
            observation_id = observation.id if observation is not None else None
        event = Score(
            name=name,
            value=value,
            trace_id=trace_id,
            observation_id=observation_id,
            comment=comment,
            source=source,
            metadata=dict(metadata) if metadata else {},
        )
        _runtime.emit(event)
        return event.id
    except Exception:  # noqa: BLE001 - rule 2
        return None
