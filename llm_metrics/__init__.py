"""llm_metrics — trace LLM calls and ship them to the llm-metrics ingest API.

    import llm_metrics

    @llm_metrics.observe(as_type="generation")
    def complete(prompt: str) -> str:
        ...

Set ``$LLM_METRICS_API_KEY`` and that is the whole setup; :func:`configure` is
there for when you want to be explicit.

The public API surface is intentionally small: anything not listed in
``__all__`` is an implementation detail and may change without a major version
bump.
"""

from llm_metrics._runtime import Stats, configure, flush, shutdown, stats
from llm_metrics._version import __version__
from llm_metrics.annotate import score, update_observation, update_trace
from llm_metrics.decorator import observe
from llm_metrics.models import (
    Observation,
    ObservationStatus,
    ObservationType,
    Score,
    ScoreSource,
    Trace,
)

__all__ = [
    "Observation",
    "ObservationStatus",
    "ObservationType",
    "Score",
    "ScoreSource",
    "Stats",
    "Trace",
    "__version__",
    "configure",
    "flush",
    "observe",
    "score",
    "shutdown",
    "stats",
    "update_observation",
    "update_trace",
]
