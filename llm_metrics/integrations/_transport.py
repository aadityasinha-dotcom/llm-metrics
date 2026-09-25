"""Read what the provider says in its response *headers*.

The response object a provider SDK returns is the parsed body. The headers
around it carry things the body never will:

* the provider's request id, the one thing their support team asks for
* rate-limit headroom — how many requests and tokens remain in the window,
  and when it resets — so a 429 can be seen coming instead of found in a log
* the provider's own processing time, which separates "the model was slow"
  from "the network was slow"
* how many HTTP attempts the SDK made, because provider clients retry 429s
  and 5xxs internally and the caller sees one slow call where three happened

Provider SDKs are built on ``httpx`` — or on its ``httpx2`` fork, which the
``anthropic`` package switched to — and both accept response event hooks.
Installing one on the provider client's underlying HTTP client makes every
attempt visible without touching the request path. Nothing here depends on
which of the two it is: the client is duck-typed, and the hook only reads
headers, never the body, so streams are unaffected.

Correlating a hook firing with the span it belongs to uses a contextvar set
around the provider call. Hooks run synchronously inside that call, on the
same thread or task, so the contextvar is exactly right for both sync and
async clients.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["HeaderSpec", "active", "install"]

_MARKER = "__llm_metrics_hook__"

_ACTIVE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "llm_metrics_active_observation", default=None
)


@dataclass(frozen=True)
class HeaderSpec:
    """Which headers a provider uses for which facts."""

    request_id: tuple[str, ...] = ()
    processing_ms: tuple[str, ...] = ()
    #: ``metadata["rate_limit"][key] <- header``. Numeric values are parsed;
    #: reset values are kept as the provider's own string, since the formats
    #: (``"6m0s"``, an ISO timestamp) are not worth normalising client-side.
    rate_limit: Mapping[str, str] = field(default_factory=dict)


@contextlib.contextmanager
def active(observation: Any) -> Iterator[None]:
    """Route header data from the hook into ``observation`` for the block."""
    token = _ACTIVE.set(observation)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def install(client: Any, spec: HeaderSpec) -> bool:
    """Attach a response hook to the provider client's HTTP client.

    Idempotent. Returns ``False``, and changes nothing, when the client does
    not expose an ``httpx``-style client the way the known SDKs do — the
    wrapper still works, it just records no header data. Never raises.
    """
    try:
        http = getattr(client, "_client", None)
        if http is None:
            return False
        current = getattr(http, "event_hooks", None)
        if not isinstance(current, Mapping):
            return False
        hooks = dict(current)
        responses = list(hooks.get("response", ()))
        if any(getattr(h, _MARKER, False) for h in responses):
            return True

        # An async client's send() is a coroutine function and its hooks must
        # be too; httpx awaits them. The class name is not checked because
        # httpx and httpx2 spell it the same but are different types.
        is_async = inspect.iscoroutinefunction(getattr(http, "send", None))
        hook = _async_hook(spec) if is_async else _sync_hook(spec)
        setattr(hook, _MARKER, True)
        responses.append(hook)
        hooks["response"] = responses
        http.event_hooks = hooks
        return True
    except Exception:  # noqa: BLE001 - rule 2
        return False


def _sync_hook(spec: HeaderSpec) -> Any:
    def hook(response: Any) -> None:
        _record(response, spec)

    return hook


def _async_hook(spec: HeaderSpec) -> Any:
    async def hook(response: Any) -> None:
        _record(response, spec)

    return hook


def _record(response: Any, spec: HeaderSpec) -> None:
    try:
        observation = _ACTIVE.get()
        if observation is None:
            return
        metadata = observation.metadata
        # Every attempt fires the hook. The last one's headers win, since
        # that is the response the caller actually got.
        metadata["http_attempts"] = int(metadata.get("http_attempts", 0)) + 1
        metadata["http_status"] = response.status_code

        headers = response.headers
        for name in spec.request_id:
            value = headers.get(name)
            if value:
                metadata["request_id"] = value
                break
        for name in spec.processing_ms:
            number = _number(headers.get(name))
            if number is not None:
                metadata["upstream_processing_ms"] = number
                break

        limits: dict[str, Any] = {}
        for key, name in spec.rate_limit.items():
            raw = headers.get(name)
            if raw is None:
                continue
            number = _number(raw)
            limits[key] = number if number is not None else raw
        if limits:
            metadata["rate_limit"] = limits
    except Exception:  # noqa: BLE001 - rule 2
        pass


def _number(raw: str | None) -> int | float | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return None


def is_async_callable(fn: Any) -> bool:
    """``iscoroutinefunction`` that sees through ``functools.wraps``.

    Provider SDKs decorate ``create`` with ``functools.wraps``, so the bound
    method is not itself a coroutine function even on the async client.
    """
    return inspect.iscoroutinefunction(inspect.unwrap(fn))
