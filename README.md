# llmobserve

Python SDK for the [llm-observe](https://github.com/llm-observe) platform.

```bash
pip install llmobserve
pip install "llmobserve[openai]"      # OpenAI wrapper
pip install "llmobserve[langchain]"   # LangChain callback handler
```

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

Explicit arguments to `IngestClient` beat environment variables, which beat defaults.

## Tracing across threads

`asyncio` tasks inherit the ambient trace and stay siblings under `gather`, so
nesting works with no extra effort. **Threads inherit nothing** — a
`ThreadPoolExecutor` worker starts with no trace and its observations become
orphan roots. Carry the context across explicitly:

```python
ctx = contextvars.copy_context()          # stdlib, at the call site
executor.submit(ctx.run, do_work, arg)

snap = context.snapshot()                 # or, across a queue
with context.adopt(snap):
    do_work(arg)
```

## Status

`models.py`, `buffer.py`, `client.py`, and `context.py`. The `@observe`
decorator and the integrations are still to come — see the build order in
`CLAUDE.md`.
