# llmobserve

Python SDK for the [llm-observe](https://github.com/llm-observe) platform.

```bash
pip install llmobserve
pip install "llmobserve[openai]"      # OpenAI wrapper
pip install "llmobserve[langchain]"   # LangChain callback handler
```

The SDK itself runs on Python 3.9+ with `httpx` as its only dependency. The
integration extras need 3.10+, because current `openai` and `langchain-core`
both dropped 3.9 — CI reflects that split.

## Usage

```python
import llmobserve
from llmobserve import observe


@observe(as_type="tool")
def search(query: str) -> list[str]: ...


@observe(as_type="generation", name="gpt-4o")
def complete(messages: list[dict]) -> dict: ...


@observe()
def answer(question: str) -> str:  # becomes the root of the trace
    return complete(search(question))["content"]
```

Set `$LLMOBSERVE_API_KEY` and that is the whole setup. Nested calls join the
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
| `LLMOBSERVE_API_KEY` | required; without it the SDK is inert and says so once on stderr |
| `LLMOBSERVE_HOST`    | defaults to the cloud endpoint |
| `LLMOBSERVE_DEBUG`   | set to log dropped batches to stderr |
| `LLMOBSERVE_ENABLED` | set to `0` to make `@observe` a near no-op (~0.2 us/call) |

Explicit arguments beat environment variables, which beat defaults.
`llmobserve.configure(...)` sets the same things in code; calls that only touch
capture flags leave the running buffer and its queued events alone.

Arguments and return values are captured by default and truncated at 2000
characters per value. Turn it off per-decorator with
`@observe(capture_input=False)` or globally via `configure()`.

Overhead is ~50 us per call with capture on — 0.01% of a 500 ms LLM call.

## OpenAI

```python
from openai import OpenAI
from llmobserve.integrations.openai import wrap_openai

client = wrap_openai(OpenAI())
client.chat.completions.create(model="gpt-4o", messages=[...])
```

Every completion becomes a `generation` observation with the model, messages,
response, token counts, and latency — nesting under an enclosing `@observe`
trace if there is one. Sync and async clients, streaming and not.

Token counts on a *streamed* response require
`stream_options={"include_usage": True}` on your call. The integration will not
add it for you: it appends a final chunk with an empty `choices` list, and code
doing `chunk.choices[0]` unguarded would start raising the moment it was wrapped.

## LangChain

```python
from llmobserve.integrations.langchain import LlmObserveTracer

tracer = LlmObserveTracer()
chain.invoke({"question": "..."}, config={"callbacks": [tracer]})
```

Chains become spans, LLM calls become generations, tools and retrievers get
their own types. Streamed generations also record `time_to_first_token_ms`.
One tracer instance is safe to reuse across invocations and share between
threads.

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
`buffer.py`, `client.py`, `context.py`, `decorator.py`, and both integrations.

Not yet done: `tests/test_contract.py`, which needs the API's published
`openapi.json`. Until it exists, the request envelope (`{"events": [...]}`),
the `Authorization: Bearer` header, and the assumption that ingest upserts
(children can arrive in a batch before their parent) are unverified.
