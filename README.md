# llm-metrics

[![CI](https://github.com/aadityasinha-dotcom/llm-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/aadityasinha-dotcom/llm-metrics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue)](https://github.com/aadityasinha-dotcom/llm-metrics)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Python SDK for the llm-metrics platform. Distribution name `llm-metrics`,
import name `llm_metrics`.

> **Not released yet**, so `pip install llm-metrics` does not work. Nothing else
> owns that name — it is simply unclaimed until the first upload. Install from
> source:

```bash
git clone https://github.com/aadityasinha-dotcom/llm-metrics
cd llm-metrics
pip install -e .                      # core
pip install -e ".[openai]"            # + OpenAI wrapper
pip install -e ".[anthropic]"         # + Anthropic wrapper
pip install -e ".[langchain]"         # + LangChain callback handler
```

The SDK itself runs on Python 3.9+ with `httpx` as its only dependency. The
integration extras need 3.10+, because current `openai` and `langchain-core`
both dropped 3.9 — CI reflects that split.

## Usage

```python
import llm_metrics
from llm_metrics import observe


@observe(as_type="tool")
def search(query: str) -> list[str]: ...


@observe(as_type="generation", name="gpt-4o")
def complete(messages: list[dict]) -> dict: ...


@observe()
def answer(question: str) -> str:  # becomes the root of the trace
    return complete(search(question))["content"]
```

Set `$LLM_METRICS_API_KEY` and that is the whole setup. Nested calls join the
enclosing trace automatically; a call with no trace open starts one.

`@observe` works bare or called, on sync and async functions, and on generators
and async generators — a streamed completion is timed over the whole stream
rather than over the microseconds it took to build the generator.

## Guarantees

- **Never blocks the caller.** All network I/O runs on a background daemon
  thread. Enqueueing an event is a `deque.append` under a microsecond-held lock.
- **Never crashes the host app.** Transport failures drop events and keep going.
- **Bounded memory.** The buffer has a hard cap; on overflow the oldest events
  are discarded rather than growing the queue.
- **No client-side cost math.** Token counts go up; pricing is applied server-side.

## Configuration

| | |
|---|---|
| `LLM_METRICS_API_KEY`     | required; without it the SDK is inert and says so once on stderr |
| `LLM_METRICS_HOST`        | defaults to the cloud endpoint |
| `LLM_METRICS_DEBUG`       | set to log dropped batches to stderr |
| `LLM_METRICS_ENABLED`     | set to `0` to make `@observe` a near no-op (~0.2 us/call) |
| `LLM_METRICS_ENVIRONMENT` | stamped on every trace, e.g. `prod` |
| `LLM_METRICS_RELEASE`     | stamped on every trace, e.g. a git SHA |
| `LLM_METRICS_SAMPLE_RATE` | fraction of traces to keep, `0.0`–`1.0`; default `1.0` |

Explicit arguments beat environment variables, which beat defaults.
`llm_metrics.configure(...)` sets the same things in code; calls that only touch
capture flags leave the running buffer and its queued events alone.

Arguments and return values are captured by default and truncated at 2000
characters per value. Turn it off per-decorator with
`@observe(capture_input=False)` or globally via `configure()`.

Overhead is ~50 us per call with capture on — 0.01% of a 500 ms LLM call.

### Sampling

```python
llm_metrics.configure(sample_rate=0.1)
```

Decided once per root trace. A kept trace arrives whole and a dropped one
leaves nothing behind — never a parent with missing children. Sampled-out calls
still run and still time themselves; they just never reach the buffer.

### Redaction

```python
def scrub(value):
    return value.replace("secret", "[redacted]") if isinstance(value, str) else value


llm_metrics.configure(redact=scrub)
```

Runs over every captured input and output, from `@observe` and from every
integration, before the value is stored. A redactor that raises drops the value
rather than shipping it unscrubbed.

## Attribution

A token ledger knows which API key spent what. It cannot say which feature,
tenant, or prompt version spent it. From inside any traced call:

```python
from llm_metrics import update_trace, update_observation


@observe()
def answer(user, question):
    update_trace(user_id=user.id, session_id=user.session, tags=["qa", "beta"])
    ...


@observe(as_type="generation")
def call_some_provider(prompt):
    update_observation(
        model="mystery-1",
        prompt_name="qa",
        prompt_version=3,
        prompt_tokens=120,
        completion_tokens=40,
        cached_tokens=100,
    )
```

`update_trace` reaches the root trace from any depth. `update_observation`
annotates the innermost open call, which is how a hand-rolled generation for a
provider without a wrapper reports its model and token counts. Both are no-ops
outside a trace and never raise.

`environment` and `release` are stamped on every trace from `configure()` or
the environment variables, so a regression can be pinned to a deploy without
tagging every call.

## Scores

```python
from llm_metrics import score, context

score("faithful", True, source="heuristic", comment="cites the doc")  # inside a call

trace_id = context.current_trace_id()  # keep it; score later
score("thumbs", -1, trace_id=trace_id)
```

A score attaches to the ambient trace and observation, or to explicit ids for
feedback that arrives after the fact. Sources are `human`, `llm_judge` and
`heuristic`.

## Health

```python
>>> llm_metrics.stats()
Stats(enabled=True, queued=412, flushed=400, dropped_on_overflow=0, failed_batches=0,
      sent_batches=4, sent_events=400, dropped_by_transport=0, retries=1, last_error=None)
>>> llm_metrics.stats().healthy
True
```

The SDK fails silently by design; this is how to check it is actually
delivering.

## OpenAI

```python
from openai import OpenAI
from llm_metrics.integrations.openai import wrap_openai

client = wrap_openai(OpenAI())
client.chat.completions.create(model="gpt-4o", messages=[...])
```

Every completion becomes a `generation` observation with the model, messages,
response, token counts, and latency — nesting under an enclosing `@observe`
trace if there is one. Sync and async clients, streaming and not; `chat.completions`,
`responses` and `embeddings`.

Beyond the basics, each generation records:

| | |
|---|---|
| `cached_tokens`, `reasoning_tokens` | first-class fields; priced differently from the totals they sit inside |
| `metadata.finish_reason` | `stop`, `length`, `tool_calls`, `content_filter` — a rising `length` rate is silent truncation |
| `metadata.refusal`, `metadata.tool_calls` | whether the model refused, and which tools it actually called (vs. `tools`, which were offered) |
| `metadata.response_id`, `system_fingerprint`, `service_tier` | the fingerprint changes when the backend model is silently rolled |
| `metadata.request_id`, `rate_limit`, `upstream_processing_ms`, `http_attempts`, `http_status` | from the response headers: the id support asks for, remaining requests/tokens in the window, the provider's own processing time, and how many attempts the `openai` client made before you saw a result |

Streamed responses add `time_to_first_token_ms`, `output_tokens_per_second`,
`stream_chunks` and `stream_completed` (`false` when the caller closed the
stream early).

Token counts on a *streamed* response require
`stream_options={"include_usage": True}` on your call. The integration will not
add it for you: it appends a final chunk with an empty `choices` list, and code
doing `chunk.choices[0]` unguarded would start raising the moment it was wrapped.

## Anthropic

```python
from anthropic import Anthropic
from llm_metrics.integrations.anthropic import wrap_anthropic

client = wrap_anthropic(Anthropic())
client.messages.create(model="claude-sonnet-5", max_tokens=1024, messages=[...])
```

Same fields as the OpenAI wrapper, so the two line up on one dashboard.
`Anthropic` and `AsyncAnthropic`; `messages.create` with and without
`stream=True`; and the `messages.stream()` helper (which reports everything but
time to first token).

Token accounting is normalised: Anthropic's `input_tokens` *excludes* cache
hits and OpenAI's `prompt_tokens` *includes* them, so this wrapper reports
`prompt_tokens` as the whole prompt (fresh + cache reads + cache writes),
`cached_tokens` as the cache-read subset, and cache writes under
`metadata.usage.cache_creation_input_tokens`. Stop reasons are mapped onto the
same `finish_reason` vocabulary (`end_turn` → `stop`, `max_tokens` → `length`,
`tool_use` → `tool_calls`, `refusal` → `content_filter`).

## LangChain

```python
from llm_metrics.integrations.langchain import LlmMetricsTracer

tracer = LlmMetricsTracer()
chain.invoke({"question": "..."}, config={"callbacks": [tracer]})
```

Chains become spans, LLM calls become generations, tools and retrievers get
their own types. Generations carry `cached_tokens` and `reasoning_tokens` when
the model reports them, plus `finish_reason`; streamed ones also record
`time_to_first_token_ms` and `output_tokens_per_second`. Retrievals record
`documents`, `retrieved_chars` (how much context is about to enter the prompt)
and `retrieval_scores` when the store attached any. One tracer instance is safe
to reuse across invocations and share between threads.

Unlike the rest of the SDK, nesting here comes from LangChain's own
`run_id`/`parent_run_id` tree rather than from contextvars — LangChain may
invoke callbacks from a thread where the ambient context is empty. The ambient
context is consulted once, for the root run, so a chain inside an `@observe`
function joins that trace instead of starting a new one.

## Tracing across threads

`asyncio` tasks inherit the ambient trace and stay siblings under `gather`, so
nesting works with no extra effort. **Threads inherit nothing** — a
`ThreadPoolExecutor` worker starts with no trace and its observations become
orphan roots. Carry the context across explicitly:

```python
ctx = contextvars.copy_context()  # stdlib, at the call site
executor.submit(ctx.run, do_work, arg)

snap = context.snapshot()  # or, across a queue
with context.adopt(snap):
    do_work(arg)
```

## Status

Every module in the `CLAUDE.md` build order is in place: `models.py`,
`buffer.py`, `client.py`, `context.py`, `decorator.py`, `annotate.py`, and the
OpenAI, Anthropic and LangChain integrations.

Not yet done: `tests/test_contract.py`, which needs the API's published
`openapi.json`. Until it exists, the request envelope (`{"events": [...]}`),
the `Authorization: Bearer` header, and the assumption that ingest upserts
(children can arrive in a batch before their parent) are unverified.
