"""llmobserve — trace LLM calls and ship them to the llm-observe ingest API.

    import llmobserve

    @llmobserve.observe(as_type="generation")
    def complete(prompt: str) -> str:
        ...

Set ``$LLMOBSERVE_API_KEY`` and that is the whole setup; :func:`configure` is
there for when you want to be explicit.

The public API surface is intentionally small: anything not listed in
``__all__`` is an implementation detail and may change without a major version
bump.
"""

from llmobserve._runtime import configure, flush, shutdown
from llmobserve._version import __version__
from llmobserve.decorator import observe
from llmobserve.models import Observation, ObservationStatus, ObservationType, Trace

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
