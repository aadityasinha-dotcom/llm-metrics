# llmobserve

Python SDK for the [llm-observe](https://github.com/llm-observe) platform.

```bash
pip install llmobserve
pip install "llmobserve[openai]"      # OpenAI wrapper
pip install "llmobserve[langchain]"   # LangChain callback handler
```

## Usage

```python
import llmobserve
from llmobserve import observe

@observe(as_type="tool")
def search(query: str) -> list[str]:
    ...

@observe(as_type="generation", name="gpt-4o")
def complete(messages: list[dict]) -> dict:
    ...

@observe()
def answer(question: str) -> str:            # becomes the root of the trace
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

`models.py`, `buffer.py`, `client.py`, `context.py`, and `decorator.py`. The
OpenAI and LangChain integrations are still to come — see the build order in
`CLAUDE.md`.
