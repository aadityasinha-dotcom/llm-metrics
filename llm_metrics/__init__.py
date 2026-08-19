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

from llm_metrics._runtime import configure, flush, shutdown
from llm_metrics._version import __version__
from llm_metrics.decorator import observe
from llm_metrics.models import Observation, ObservationStatus, ObservationType, Trace

__all__ = [
    "Observation",
    "ObservationStatus",
    "ObservationType",
    "Trace",
    "__version__",
    "configure",
    "flush",
    "observe",
    "shutdown",
]
