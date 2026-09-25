# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Attribution.** `update_trace(user_id=, session_id=, tags=, name=, metadata=)`
  and `update_observation(model=, prompt_name=, prompt_version=, *_tokens=,
  metadata=)` annotate the ambient trace or observation from inside any traced
  call. `Trace` gains `session_id`, `tags`, `environment` and `release`;
  `Observation` gains `prompt_name` and `prompt_version`. `context.snapshot()`
  now carries the live observation as well as its id.
- **Deploy stamping.** `environment` and `release` on `configure()` and as
  `LLM_METRICS_ENVIRONMENT` / `LLM_METRICS_RELEASE`, written onto every trace
  the SDK creates.
- **Scores.** `score(name, value, *, trace_id=, observation_id=, comment=,
  source=)` emits a `Score` event against the ambient trace and observation or
  explicit ids. Sources: `human`, `llm_judge`, `heuristic`.
- **Token detail.** `Observation.cached_tokens` and `reasoning_tokens` as
  first-class fields, filled by every integration that can. Audio and
  prediction counts go under `metadata["usage"]`.
- **Sampling.** `sample_rate` on `configure()` and `LLM_METRICS_SAMPLE_RATE`.
  Decided once per root trace and inherited by every child, so a kept trace is
  whole. Sampled-out calls still run and still time themselves.
- **Redaction.** `redact=` on `configure()`: a callable run over every captured
  input and output, from the decorator and every integration. Returning `None`
  drops the value; raising drops it too.
- **`stats()`.** A `Stats` snapshot of the pipeline: queued, flushed, dropped
  on overflow, failed batches, and the transport's sent/dropped/retry counters,
  with a `healthy` property. The first public signal for an SDK that otherwise
  fails silently.
- **OpenAI wrapper** now records `finish_reason`, `refusal`, the tools the
  model actually called (`tool_calls`, vs. `tools` offered), `response_id`,
  `system_fingerprint` and `service_tier`; on streams, `time_to_first_token_ms`,
  `output_tokens_per_second`, and `stream_completed`. Responses API streams
  are read from the `response.completed` event. `max_output_tokens` and
  `service_tier` join the tracked request parameters.
- **Response headers.** A response hook on the provider client's underlying
  `httpx`/`httpx2` client records `request_id`, `rate_limit` (limit, remaining,
  reset for requests and tokens), `upstream_processing_ms`, `http_status`, and
  `http_attempts` — the number of HTTP attempts the provider client made
  internally, so its silent retries become visible. Headers only; bodies and
  streams are untouched. Fires for error responses too, so a 429 observation
  carries the headroom at the moment it happened.
- **`wrap_anthropic`** — `integrations/anthropic.py`, with the `[anthropic]`
  extra. `Anthropic` and `AsyncAnthropic`; `messages.create` plain and with
  `stream=True`; the `messages.stream()` helper. Prompt tokens are normalised
  to the OpenAI convention (cache hits included in the total, reported
  separately as `cached_tokens`), cache writes go under `metadata["usage"]`,
  and stop reasons map onto the shared `finish_reason` vocabulary. Never
  imports `anthropic`.
- **LangChain tracer** reads `cached_tokens` and `reasoning_tokens` from
  `usage_metadata`, records `finish_reason` and `output_tokens_per_second`,
  and on retrievals records `retrieved_chars` and `retrieval_scores`.
- `integrations/_stream.py` — the stream proxies shared by the provider
  wrappers, with the first-token and throughput timing in one place.
- **`tests/test_contract.py`** — validates a request captured off the real
  transport against the server's `openapi.json`, vendored at
  `tests/contract/openapi.json` and refreshed by `scripts/sync_openapi.py`
  (`make sync-contract`). Checks the endpoint, the bearer scheme, the
  `X-SDK-Version` header, the flat envelope, that every event validates
  against its schema, and that every key the SDK sends is one the server
  stores - the check that catches a rename the server would otherwise ignore.
  `jsonschema` joins the dev extras.

### Changed

- `Trace` and `Observation` payloads carry the new optional keys above. The
  server accepts unknown fields, so older servers ignore them.

## [0.1.0] — unreleased

First cut of the SDK. Not published yet; see **Known gaps** below.

### Added

- **`@observe`** — records one observation per call, with the parent taken from
  the ambient trace, latency from `perf_counter`, arguments bound against a
  signature cached at decoration time, and status when the call raises. Works
  bare or called, on sync and async functions, and on generators and async
  generators. A streamed completion is timed over the whole stream rather than
  over the microseconds it took to build the generator.
- **`EventBuffer`** — bounded, thread-safe queue with a daemon flush thread.
  Flushes on 100 events or 5 seconds. Drops oldest on overflow. No lock is ever
  held while the flush target runs, so a slow transport cannot reach the caller:
  10,000 `add()` calls take ~21 ms while the flush target sleeps 0.25 s per batch.
- **`IngestClient`** — batched `POST /v1/ingest` with full-jitter exponential
  backoff. Retries transport errors, 408, 425, 429 and 5xx; drops every other
  4xx on the first attempt. Honours `Retry-After` in its seconds form. One
  pooled `httpx.Client`, created on the flush thread.
- **Bounded shutdown** — a daemon flush thread plus an `atexit` hook that
  signals and joins with a deadline, so a short script still delivers and a hung
  ingest API cannot hang process exit. The deadline is a live object, not a
  snapshot, so it also bounds a send already in flight.
- **`context`** — trace nesting via `contextvars`. `asyncio` tasks nest and stay
  siblings under `gather`; threads inherit nothing, so `snapshot()`/`adopt()`
  carry the context across that boundary.
- **`wrap_openai`** — instruments `chat.completions`, `responses`, and
  `embeddings` on both `OpenAI` and `AsyncOpenAI`, streaming included. Never
  imports `openai`; everything is duck-typed.
- **`LlmMetricsTracer`** — LangChain callback handler. Chains become spans, LLM
  calls generations, tools and retrievers their own types. Nesting comes from
  LangChain's `run_id`/`parent_run_id` rather than contextvars, since callbacks
  can arrive on any thread. Streamed generations record
  `time_to_first_token_ms`. The in-flight run map is bounded.
- **`configure`, `flush`, `shutdown`** — optional explicit setup. Settings-only
  calls leave a running buffer and its queued events alone.
- Configuration via `LLM_METRICS_API_KEY`, `LLM_METRICS_HOST`, `LLM_METRICS_DEBUG`,
  and `LLM_METRICS_ENABLED`.
- CI across Python 3.9–3.13, plus `make ci` for the same checks locally.

### Design guarantees

- **Never blocks the caller.** ~50 µs of overhead per call with capture on,
  ~0.2 µs with `LLM_METRICS_ENABLED=0`.
- **Never crashes the host app.** A broken SDK degrades to an uninstrumented
  call; the result or exception reaches the caller unchanged.
- **No client-side cost.** Token counts are sent; pricing stays server-side.
- **Bounded memory** in the event buffer and in the LangChain run map.

### Known gaps

- **Not on PyPI yet.** The distribution name is `llm-metrics` and the import
  name is `llm_metrics`; both were unclaimed when checked. The package was
  renamed from `llmobserve` because that name and `llmobserve-sdk` belong to an
  unrelated, actively published LLM-observability product, which collided on the
  import name too — its wheel also ships a top-level `llmobserve/` package.
  `.github/workflows/publish.yml` is wired for trusted publishing but has never
  run.
- The `os.register_at_fork` paths in `buffer.py` and `client.py` are not covered
  by tests.

[Unreleased]: https://github.com/aadityasinha-dotcom/llm-metrics/compare/main...HEAD
