# llm-metrics

Python SDK for the llm-metrics platform. Users `pip install llm-metrics`, add an API
key, and decorate functions to send LLM traces to a self-hosted or cloud ingest API.

This repo contains ONLY the SDK. The FastAPI backend, eval worker, and Next.js
dashboard live in a separate repo (`llm-observe`). Do not add server code here.

## Non-negotiable design rules

These are the properties the SDK exists to guarantee. Never violate them for
convenience.

1. **Never block the caller.** All network I/O happens on a background flush
   thread. The `@observe` decorator must add negligible latency to the wrapped
   function. Any change that introduces a synchronous HTTP call in the hot path
   is a bug.
2. **Never crash the host app.** If the ingest API is unreachable, times out, or
   returns 5xx, drop events and continue. Catch broadly around all SDK internals.
   An observability tool that takes down production is worse than no tool.
3. **Never compute cost client-side.** Send token counts only. Pricing lives on
   the server so it can be updated without users upgrading the SDK.
4. **Bounded memory.** The event buffer has a max size. When full, drop oldest
   events rather than growing without limit.

## Architecture

```
user code -> @observe decorator -> in-memory buffer -> background thread
          -> batched POST /v1/ingest (HTTPS + API key)
```

- `client.py`   — HTTP transport, retry with exponential backoff, auth header
- `buffer.py`   — bounded queue, flush on N events or T seconds, daemon thread
- `decorator.py`— `@observe()`, sync + async support
- `context.py`  — `contextvars` holding trace_id and parent_span_id for nesting
- `models.py`   — Trace, Observation, Score dataclasses
- `annotate.py` — `update_trace`, `update_observation`, `score`: attribution from inside a call
- `integrations/` — openai and anthropic wrappers, langchain callback handler;
  `_stream.py` (shared stream proxies + first-token timing) and `_transport.py`
  (response-header hook: request id, rate limits, attempts)

## Data model (must match server schema)

- **Trace** — one user-facing request. id, name, user_id, session_id, tags,
  environment, release, metadata, timestamps
- **Observation** — one LLM/tool call. Nestable via `parent_id`. model, input,
  output, prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens,
  latency_ms, prompt_name, prompt_version
- **Score** — a judgement about a trace or observation. name, value, source
- Field naming follows OpenTelemetry GenAI semantic conventions where applicable
- `prompt_tokens` is always the *whole* prompt, cached part included;
  `cached_tokens` is the subset. Anthropic reports it the other way round and
  the wrapper normalises — do not undo that
- Provider-specific facts (`finish_reason`, `tool_calls`, `rate_limit`,
  `time_to_first_token_ms`, ...) live in `metadata`, on a shared vocabulary
  across wrappers. New wrappers must use the same keys

## Contract with the server

The SDK and API version independently. `tests/test_contract.py` validates request
payloads against the API's published `openapi.json`, vendored at
`tests/contract/openapi.json` and refreshed with `make sync-contract`. If that
test fails after a change, the payload shape drifted — fix the SDK or coordinate
a server change. The alias table in the test (`parent_id` → `parent_observation_id`,
`status` → `level`, ...) is part of the contract: the server accepts both
spellings but publishes only one.

Every request sends `X-SDK-Version`. The ingest endpoint is versioned (`/v1/`).

## Conventions

- Python 3.9+ (CI matrix runs 3.9 through 3.13)
- Zero required runtime dependencies beyond `httpx`. Integrations use extras:
  `pip install llm-metrics[openai]`
- Type hints everywhere; `mypy --strict` passes
- `ruff` for lint and format
- Public API surface is only what's exported in `__init__.py` — keep it small

## Commands

```bash
make install    # dev install with extras
make test       # pytest
make lint       # ruff + mypy
make build      # build wheel
make ci         # everything CI runs, minus the 3.9-3.13 matrix
```

`make ci` is the pre-push check. It cannot cover the version matrix — that
needs interpreters a single checkout does not have, so CI runs `test` on
3.9-3.13. The provider extras are only exercised on 3.10+, because current
`openai` and `langchain-core` both dropped 3.9.

## Build order

Working through the SDK in this sequence:

1. `models.py` — dataclasses first, everything depends on the shape
2. `buffer.py` — bounded queue + background flush thread
3. `client.py` — transport with retry
4. `context.py` — contextvars for trace nesting
5. `decorator.py` — `@observe`, sync then async
6. `integrations/openai.py`
7. `integrations/langchain.py`
8. `annotate.py`, `integrations/anthropic.py`, `_stream.py`, `_transport.py`

Tests that matter most: `test_never_blocks.py` and `test_fails_silently.py`.
Write them alongside the code they cover, not at the end.

## Out of scope

Do not add: database access, eval/scoring logic, cost calculation, dashboard
code, or anything requiring server credentials. Those belong in `llm-observe`.
