# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] — unreleased

First cut of the SDK. Not published: the request payload has not yet been
validated against the ingest API's `openapi.json`, so the envelope shape, the
auth header, and the assumption that ingest upserts are all unverified. See
**Known gaps** below.

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
- **`LlmObserveTracer`** — LangChain callback handler. Chains become spans, LLM
  calls generations, tools and retrievers their own types. Nesting comes from
  LangChain's `run_id`/`parent_run_id` rather than contextvars, since callbacks
  can arrive on any thread. Streamed generations record
  `time_to_first_token_ms`. The in-flight run map is bounded.
- **`configure`, `flush`, `shutdown`** — optional explicit setup. Settings-only
  calls leave a running buffer and its queued events alone.
- Configuration via `LLMOBSERVE_API_KEY`, `LLMOBSERVE_HOST`, `LLMOBSERVE_DEBUG`,
  and `LLMOBSERVE_ENABLED`.
- CI across Python 3.9–3.13, plus `make ci` for the same checks locally.

### Design guarantees

- **Never blocks the caller.** ~50 µs of overhead per call with capture on,
  ~0.2 µs with `LLMOBSERVE_ENABLED=0`.
- **Never crashes the host app.** A broken SDK degrades to an uninstrumented
  call; the result or exception reaches the caller unchanged.
- **No client-side cost.** Token counts are sent; pricing stays server-side.
- **Bounded memory** in the event buffer and in the LangChain run map.

### Known gaps

- `tests/test_contract.py` does not exist yet — it needs the API's published
  `openapi.json`. Until then these are assumptions, not facts:
  - the request envelope is `{"events": [...]}`
  - auth is `Authorization: Bearer <key>` rather than `X-API-Key`
  - ingest upserts, since children finish before parents and a batch can carry
    an observation before the trace it belongs to
- No redaction hook. Arguments and return values are captured by default;
  `capture_input=False` is the only control and it is all-or-nothing.
- No sampling.
- No public way to set `user_id` or trace metadata from inside a traced call.
- No health/diagnostics accessor. The SDK fails silently by design, so today the
  only signal is `LLMOBSERVE_DEBUG=1`.
- The `os.register_at_fork` paths in `buffer.py` and `client.py` are not covered
  by tests.

[Unreleased]: https://github.com/aadityasinha-dotcom/llmobserve-python/compare/main...HEAD
