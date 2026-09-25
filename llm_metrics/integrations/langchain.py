"""Trace LangChain runs through its callback system.

    from llm_metrics.integrations.langchain import LlmMetricsTracer

    tracer = LlmMetricsTracer()
    chain.invoke({"question": "..."}, config={"callbacks": [tracer]})

Chains become spans, LLM and chat-model calls become generations, tools become
tool observations, and retrievers become retrieval observations. If a trace is
already open — the chain runs inside an ``@observe`` function, say — the whole
LangChain tree nests under it.

Nesting comes from LangChain, not from contextvars
--------------------------------------------------
This is the one integration that does **not** read the ambient parent for
nesting. LangChain hands every callback an explicit ``run_id`` and
``parent_run_id``, and it may invoke handlers from a thread other than the one
that started the run — where contextvars are empty by design. So this handler
keeps its own ``run_id -> span`` map and builds the tree from LangChain's ids.
The ambient context is consulted exactly once, for the root run, to decide
whether to join an enclosing trace or start a new one.

Bounded, like everything else
-----------------------------
That map is state with a lifetime the SDK does not control: a run that never
reports an end — a crash between callbacks, a framework bug, a handler attached
to a tree it only partly sees — would sit in it forever. It is capped, and the
oldest entries are evicted when the cap is reached, for the same reason the
event buffer is bounded (rule 4).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar
from uuid import UUID

from llm_metrics import context
from llm_metrics.decorator import _Span, finish_span, open_span, summarise
from llm_metrics.models import ObservationType, Trace

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - depends on the extra
    raise ImportError(
        "The llm-metrics LangChain integration needs langchain-core. "
        "Install it with: pip install 'llm-metrics[langchain]'"
    ) from exc

__all__ = ["LlmMetricsTracer"]

#: Ceiling on in-flight runs held in the map. Generous for any real tree —
#: a deep agent loop is dozens of runs, not thousands.
DEFAULT_MAX_RUNS = 2_000

F = TypeVar("F", bound=Callable[..., Any])


class _Run:
    """One open LangChain run."""

    __slots__ = ("first_token_at", "span", "started", "trace")

    def __init__(self, span: _Span, trace: Trace | None) -> None:
        self.span = span
        #: The trace this subtree belongs to, inherited by child runs.
        self.trace = trace
        self.started = time.perf_counter()
        self.first_token_at: float | None = None


def _never_raises(method: F) -> F:
    """Rule 2, applied to every hook.

    LangChain swallows handler exceptions by default, but "by default" is a
    setting a user can flip, and a tracer that can abort someone's chain is not
    worth having.
    """

    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except Exception:  # noqa: BLE001 - rule 2: never crash the host app
            return None

    guarded.__name__ = method.__name__
    guarded.__qualname__ = method.__qualname__
    guarded.__doc__ = method.__doc__
    return guarded  # type: ignore[return-value]


class LlmMetricsTracer(BaseCallbackHandler):
    """LangChain callback handler that records runs as llm-metrics traces.

    Args:
        trace_name: Name for traces this handler creates. Defaults to the name
            of the root run.
        user_id: Attached to traces this handler creates.
        metadata: Static metadata attached to every observation.
        max_runs: Cap on simultaneously-open runs held in memory.

    One instance is safe to reuse across many invocations and to share between
    threads.
    """

    #: LangChain checks this to decide whether to re-raise handler errors.
    #: False regardless, but the guard on each hook is the real protection.
    raise_error: bool = False

    def __init__(
        self,
        *,
        trace_name: str | None = None,
        user_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        max_runs: int = DEFAULT_MAX_RUNS,
    ) -> None:
        super().__init__()
        self.trace_name = trace_name
        self.user_id = user_id
        self.metadata = dict(metadata) if metadata else {}
        self.max_runs = max(1, max_runs)

        # Callbacks can arrive on any thread, so the map is locked. The lock is
        # only ever held across dict operations, never across an emit.
        self._lock = threading.Lock()
        self._runs: OrderedDict[UUID, _Run] = OrderedDict()
        self.dropped_runs = 0

    # ------------------------------------------------------------- bookkeeping

    def _open(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        *,
        name: str,
        as_type: str,
        input_value: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        parent = self._get(parent_run_id) if parent_run_id is not None else None

        combined: dict[str, Any] = dict(self.metadata)
        if metadata:
            combined.update(metadata)

        if parent is not None:
            span = open_span(
                name,
                as_type,
                input_value=summarise(input_value),
                metadata=combined,
                trace=parent.trace,
                parent_id=parent.span.observation.id,
            )
            trace = parent.trace
        else:
            # Root of this tree. The only point where the ambient context is
            # consulted: join an enclosing @observe trace, or start our own.
            # A parent_run_id we have never seen lands here too — a handler
            # attached partway down a tree roots itself rather than guessing.
            span = open_span(name, as_type, input_value=summarise(input_value), metadata=combined)
            if span is None:
                return
            trace = span.trace if span.trace is not None else context.current_trace()
            if span.trace is not None and self.trace_name:
                # Renaming applies only to a trace we created. Relabelling an
                # enclosing @observe trace would be rewriting someone else's.
                span.trace.name = self.trace_name
            if self.user_id and trace is not None and trace.user_id is None:
                # user_id is filled in either way: a caller who passed one meant
                # it, and silently dropping it because @observe happened to own
                # the trace would be a confusing gap on the dashboard. An id
                # already set wins — first writer keeps it.
                trace.user_id = self.user_id

        if span is None:
            return
        self._put(run_id, _Run(span, trace))

    def _close(self, run_id: UUID, output: Any = None, exc: BaseException | None = None) -> None:
        run = self._pop(run_id)
        if run is None:
            return  # never opened, or already evicted
        finish_span(run.span, output=summarise(output), exc=exc, summarise_output=False)

    def _put(self, run_id: UUID, run: _Run) -> None:
        with self._lock:
            self._runs[run_id] = run
            while len(self._runs) > self.max_runs:
                # Rule 4: a run that never reports an end must not pin memory.
                self._runs.popitem(last=False)
                self.dropped_runs += 1

    def _get(self, run_id: UUID) -> _Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def _pop(self, run_id: UUID) -> _Run | None:
        with self._lock:
            return self._runs.pop(run_id, None)

    @property
    def open_runs(self) -> int:
        with self._lock:
            return len(self._runs)

    # -------------------------------------------------------------- chain hooks

    @_never_raises
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._open(
            run_id,
            parent_run_id,
            name=_name(serialized, kwargs, "chain"),
            as_type=ObservationType.SPAN,
            input_value=inputs,
            metadata=_tags(kwargs),
        )

    @_never_raises
    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, output=outputs)

    @_never_raises
    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, exc=error)

    # ---------------------------------------------------------------- llm hooks

    @_never_raises
    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_generation(serialized, prompts, run_id, parent_run_id, kwargs)

    @_never_raises
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_generation(serialized, _plain_messages(messages), run_id, parent_run_id, kwargs)

    def _start_generation(
        self,
        serialized: dict[str, Any] | None,
        input_value: Any,
        run_id: UUID,
        parent_run_id: UUID | None,
        kwargs: Mapping[str, Any],
    ) -> None:
        params = _invocation_params(kwargs)
        self._open(
            run_id,
            parent_run_id,
            name=_name(serialized, kwargs, "llm"),
            as_type=ObservationType.GENERATION,
            input_value=input_value,
            metadata={**_tags(kwargs), **params},
        )
        run = self._get(run_id)
        if run is not None:
            model = params.get("model") or params.get("model_name")
            if model:
                run.span.observation.model = str(model)

    @_never_raises
    def on_llm_new_token(self, token: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Records time to first token, the metric that matters for streaming.

        Total latency hides it: a stream that starts instantly and one that
        stalls for four seconds finish at the same time and feel nothing alike.
        """
        run = self._get(run_id)
        if run is not None and run.first_token_at is None:
            run.first_token_at = time.perf_counter()

    @_never_raises
    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._get(run_id)
        if run is None:
            return
        observation = run.span.observation

        _apply_usage(observation, response)
        if not observation.model:
            observation.model = _llm_model(response)
        reason = _finish_reason(response)
        if reason:
            observation.metadata["finish_reason"] = reason
        if run.first_token_at is not None:
            observation.metadata["time_to_first_token_ms"] = round(
                (run.first_token_at - run.started) * 1000.0, 3
            )
            generating = time.perf_counter() - run.first_token_at
            tokens = observation.completion_tokens
            if tokens and generating > 0:
                observation.metadata["output_tokens_per_second"] = round(tokens / generating, 2)

        self._close(run_id, output=_llm_output(response))

    @_never_raises
    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, exc=error)

    # --------------------------------------------------------------- tool hooks

    @_never_raises
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._open(
            run_id,
            parent_run_id,
            name=_name(serialized, kwargs, "tool"),
            as_type=ObservationType.TOOL,
            input_value=inputs if inputs is not None else input_str,
            metadata=_tags(kwargs),
        )

    @_never_raises
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, output=output)

    @_never_raises
    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, exc=error)

    # ---------------------------------------------------------- retriever hooks

    @_never_raises
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._open(
            run_id,
            parent_run_id,
            name=_name(serialized, kwargs, "retriever"),
            as_type=ObservationType.RETRIEVAL,
            input_value=query,
            metadata=_tags(kwargs),
        )

    @_never_raises
    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        run = self._get(run_id)
        if run is not None:
            metadata = run.span.observation.metadata
            metadata["documents"] = len(documents)
            # How much context this retrieval will push into the prompt, and
            # how confident the store was — the two numbers that say whether
            # a RAG step is pulling its weight.
            metadata["retrieved_chars"] = sum(
                len(c) for c in (_get(d, "page_content") for d in documents) if isinstance(c, str)
            )
            scores = [_score(d) for d in documents]
            if any(s is not None for s in scores):
                metadata["retrieval_scores"] = scores
        self._close(run_id, output=[_document(d) for d in documents])

    @_never_raises
    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, exc=error)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _name(serialized: Any, kwargs: Mapping[str, Any], default: str) -> str:
    """The most specific name LangChain offers for this run."""
    if isinstance(serialized, Mapping):
        name = serialized.get("name")
        if name:
            return str(name)
        ident = serialized.get("id")
        if isinstance(ident, (list, tuple)) and ident:
            return str(ident[-1])  # fully-qualified path; the class is the tail
    name = kwargs.get("name")
    return str(name) if name else default


def _tags(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    tags = kwargs.get("tags")
    if tags:
        metadata["tags"] = list(tags)
    supplied = kwargs.get("metadata")
    if isinstance(supplied, Mapping) and supplied:
        metadata.update(supplied)
    return metadata


#: Worth keeping from invocation_params. The rest is provider plumbing.
_TRACKED_PARAMS = ("model", "model_name", "temperature", "top_p", "max_tokens", "stop")


def _invocation_params(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    params = kwargs.get("invocation_params")
    if not isinstance(params, Mapping):
        return {}
    return {key: params[key] for key in _TRACKED_PARAMS if params.get(key) is not None}


def _plain_messages(messages: Any) -> Any:
    """``list[list[BaseMessage]]`` down to plain role/content dicts."""
    try:
        batches = [[_plain_message(message) for message in batch] for batch in messages]
    except TypeError:
        return messages
    if len(batches) == 1:
        return batches[0]  # the overwhelmingly common single-prompt case
    return batches


def _plain_message(message: Any) -> Any:
    role = _get(message, "type") or _get(message, "role")
    content = _get(message, "content")
    if role is None and content is None:
        return message
    plain: dict[str, Any] = {"role": role, "content": content}
    calls = _get(message, "tool_calls")
    if calls:
        plain["tool_calls"] = calls
    return plain


def _apply_usage(observation: Any, response: Any) -> None:
    """Token counts, from whichever place this LangChain version put them.

    ``llm_output["token_usage"]`` is the older provider-echo path and is often
    ``None``; modern chat models carry ``usage_metadata`` on the message, and
    that is also the only place the cached and reasoning subsets appear.
    """
    for generation in _flat_generations(response):
        usage = _get(_get(generation, "message"), "usage_metadata")
        if usage:
            observation.prompt_tokens = _as_int(_get(usage, "input_tokens"))
            observation.completion_tokens = _as_int(_get(usage, "output_tokens"))
            observation.cached_tokens = _as_int(
                _get(_get(usage, "input_token_details"), "cache_read")
            )
            observation.reasoning_tokens = _as_int(
                _get(_get(usage, "output_token_details"), "reasoning")
            )
            return

    llm_output = _get(response, "llm_output")
    usage = _get(llm_output, "token_usage") or _get(llm_output, "usage")
    observation.prompt_tokens = _as_int(_get(usage, "prompt_tokens"))
    observation.completion_tokens = _as_int(_get(usage, "completion_tokens"))


def _finish_reason(response: Any) -> str | None:
    """Why generation stopped, from ``generation_info`` or the message."""
    for generation in _flat_generations(response):
        info = _get(generation, "generation_info")
        reason = _get(info, "finish_reason")
        if not reason:
            reason = _get(_get(_get(generation, "message"), "response_metadata"), "finish_reason")
        if reason:
            return str(reason)
    return None


def _score(doc: Any) -> float | None:
    """A retriever's relevance score, when the store attached one."""
    metadata = _get(doc, "metadata")
    for key in ("score", "relevance_score", "similarity"):
        value = _get(metadata, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _llm_model(response: Any) -> str | None:
    llm_output = _get(response, "llm_output")
    model = _get(llm_output, "model_name") or _get(llm_output, "model")
    if model:
        return str(model)
    for generation in _flat_generations(response):
        metadata = _get(_get(generation, "message"), "response_metadata")
        model = _get(metadata, "model_name") or _get(metadata, "model")
        if model:
            return str(model)
    return None


def _llm_output(response: Any) -> Any:
    texts = [_get(generation, "text") for generation in _flat_generations(response)]
    texts = [t for t in texts if isinstance(t, str)]
    if not texts:
        return None
    return summarise(texts[0] if len(texts) == 1 else texts)


def _flat_generations(response: Any) -> list[Any]:
    try:
        return [generation for batch in _get(response, "generations") or [] for generation in batch]
    except TypeError:
        return []


def _document(doc: Any) -> Any:
    content = _get(doc, "page_content")
    if content is None:
        return summarise(doc)
    return {"page_content": summarise(content), "metadata": _get(doc, "metadata") or {}}


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
