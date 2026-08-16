# llmobserve-python

Python SDK for the llm-observe platform. Users `pip install llmobserve`, add an API
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
- `models.py`   — Trace, Observation dataclasses
- `integrations/` — openai wrapper, langchain callback handler

## Data model (must match server schema)

- **Trace** — one user-facing request. id, name, user_id, metadata, timestamps
- **Observation** — one LLM/tool call. Nestable via `parent_id`. model, input,
  output, prompt_tokens, completion_tokens, latency_ms
- Field naming follows OpenTelemetry GenAI semantic conventions where applicable

## Contract with the server

The SDK and API version independently. `tests/test_contract.py` validates request
payloads against the API's published `openapi.json`. If that test fails after a
change, the payload shape drifted — fix the SDK or coordinate a server change.

Every request sends `X-SDK-Version`. The ingest endpoint is versioned (`/v1/`).

## Conventions

- Python 3.9+ (CI matrix runs 3.9 through 3.13)
- Zero required runtime dependencies beyond `httpx`. Integrations use extras:
  `pip install llmobserve[openai]`
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

Tests that matter most: `test_never_blocks.py` and `test_fails_silently.py`.
Write them alongside the code they cover, not at the end.

## Out of scope

Do not add: database access, eval/scoring logic, cost calculation, dashboard
code, or anything requiring server credentials. Those belong in `llm-observe`.
