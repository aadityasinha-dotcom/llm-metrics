"""HTTP transport for the ingest API.

This is the ``flush_fn`` the :class:`~llm_metrics.buffer.EventBuffer` calls. It
runs exclusively on the buffer's flush thread, never on the caller's, so it is
allowed to block — but only for as long as its deadline permits.

Retry policy
------------
Retryable: network/transport errors, 408, 425, 429, and 5xx. These are
conditions that plausibly resolve on their own.

Not retryable: every other 4xx. A 401 from a bad API key will still be a 401
in four seconds, so retrying it does nothing except burn the shutdown budget
that other batches need.

Deadlines
---------
:meth:`IngestClient.send` takes a live :class:`~llm_metrics.buffer.Deadline`.
It is unarmed during normal operation — the flush thread is a background
thread and nobody is waiting on it. :meth:`~llm_metrics.buffer.EventBuffer.shutdown`
arms it, and the client abandons retries (and shortens its per-request timeout)
to respect the remaining budget.

The deadline is re-read between attempts and slept against rather than captured
once, because the flush thread is usually already mid-batch when shutdown
arrives. A budget the in-flight send cannot see is no budget at all.

This matters more than it looks. Three attempts with 0.5s exponential backoff
plus a 10s request timeout is a worst case of ~30s for a *single* batch. Run
that unbounded inside a 5s atexit budget and the final flush delivers nothing
at all — the exact failure the atexit hook exists to prevent.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
from typing import Any

import httpx

from llm_metrics._version import __version__
from llm_metrics.buffer import Deadline

__all__ = ["ClientStats", "IngestClient"]

DEFAULT_HOST = "https://api-eta-eight-10.vercel.app"
INGEST_PATH = "/v1/ingest"

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_MAX = 8.0

ENV_API_KEY = "LLM_METRICS_API_KEY"
ENV_HOST = "LLM_METRICS_HOST"
ENV_DEBUG = "LLM_METRICS_DEBUG"

#: Statuses worth trying again. Everything else in the 4xx range is a client
#: bug or a bad key, and will fail identically on retry.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Statuses we call out explicitly once, because they are silent misconfigurations
#: a user would otherwise spend an afternoon on.
AUTH_STATUS = frozenset({401, 403})


class ClientStats:
    """Advisory counters. Mutated only on the flush thread."""

    __slots__ = (
        "dropped_events",
        "last_error",
        "permanent_failures",
        "retries",
        "sent_batches",
        "sent_events",
    )

    def __init__(self) -> None:
        self.sent_batches = 0
        self.sent_events = 0
        self.dropped_events = 0
        self.retries = 0
        self.permanent_failures = 0
        self.last_error: str | None = None

    def __repr__(self) -> str:
        return (
            f"ClientStats(sent_batches={self.sent_batches}, "
            f"sent_events={self.sent_events}, dropped_events={self.dropped_events}, "
            f"retries={self.retries}, permanent_failures={self.permanent_failures}, "
            f"last_error={self.last_error!r})"
        )


class IngestClient:
    """POSTs batches to ``{host}/v1/ingest``.

    Args:
        api_key: Falls back to ``$LLM_METRICS_API_KEY``. Without one the client
            is inert: it drops every batch and never opens a socket.
        host: Falls back to ``$LLM_METRICS_HOST``, then to the cloud endpoint.
        timeout: Per-request timeout in seconds.
        max_attempts: Total attempts per batch, including the first.
        backoff_base: First retry delay. Doubles each attempt, capped at
            ``backoff_max``, with full jitter applied.
        transport: Injected for tests (``httpx.MockTransport``). When given,
            the client uses it instead of opening real connections.
    """

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # Explicit argument beats environment beats default.
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        raw_host = host if host is not None else os.environ.get(ENV_HOST) or DEFAULT_HOST
        self.host = raw_host.rstrip("/")
        self.url = f"{self.host}{INGEST_PATH}"

        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

        self._transport = transport
        self._http: httpx.Client | None = None
        self._warned: set[str] = set()
        self._closed = False

        self.stats = ClientStats()

        if not self.api_key:
            self._warn_once(
                "no_key",
                f"llm-metrics: no API key (set ${ENV_API_KEY}); traces will be dropped",
            )

        self._register_fork_handler()

    # --------------------------------------------------------------- properties

    @property
    def enabled(self) -> bool:
        """False when there is no API key or the client has been closed."""
        return bool(self.api_key) and not self._closed

    # ------------------------------------------------------------------- sending

    def send(self, payload: list[dict[str, Any]], deadline: Deadline | None = None) -> None:
        """Deliver one batch. Never raises.

        The signature matches ``EventBuffer``'s flush target, so this method is
        passed directly as ``flush_fn``.

        Args:
            payload: Serialised events.
            deadline: Live budget. Consulted before every attempt and slept
                against, so arming it mid-send takes effect immediately.
                ``None`` means unlimited.
        """
        try:
            self._send(payload, deadline)
        except Exception as exc:  # noqa: BLE001 - rule 2: never crash the host app
            # Belt and braces. _send already handles everything it expects; this
            # catches the bugs we did not anticipate.
            self.stats.dropped_events += len(payload)
            self.stats.last_error = f"{type(exc).__name__}: {exc}"

    def _send(self, payload: list[dict[str, Any]], deadline: Deadline | None) -> None:
        if not payload:
            return
        if not self.enabled:
            self.stats.dropped_events += len(payload)
            return

        body = self._encode(payload)
        if body is None:
            self.stats.dropped_events += len(payload)
            return

        for attempt in range(self.max_attempts):
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                self._fail(payload, "deadline exceeded before attempt")
                return

            outcome, detail = self._attempt(body, remaining)

            if outcome == "ok":
                self.stats.sent_batches += 1
                self.stats.sent_events += len(payload)
                return
            if outcome == "permanent":
                self._fail(payload, detail, permanent=True)
                return

            # Retryable. Is there time and budget for another go?
            if attempt == self.max_attempts - 1:
                self._fail(payload, detail)
                return

            delay = self._backoff(attempt, detail)
            remaining = self._remaining(deadline)
            if remaining is not None:
                if remaining <= 0:
                    self._fail(payload, "deadline exceeded before retry")
                    return
                # Never sleep past the deadline — a shortened sleep still leaves
                # a chance of delivery, sleeping through it guarantees none.
                delay = min(delay, remaining)
            self.stats.retries += 1
            if deadline is None:
                time.sleep(delay)
            else:
                # Wakes early if the deadline is armed while we are sleeping.
                deadline.sleep(delay)

    def _attempt(self, body: bytes, remaining: float | None) -> tuple[str, str]:
        """One HTTP round trip.

        Returns:
            ``("ok" | "retry" | "permanent", detail)``. ``detail`` carries the
            ``Retry-After`` value for 429s so :meth:`_backoff` can honour it.
        """
        timeout = self.timeout if remaining is None else min(self.timeout, remaining)
        try:
            response = self._client().post(self.url, content=body, timeout=timeout)
        except httpx.TransportError as exc:
            # Connect errors, read timeouts, DNS failures, TLS problems.
            return "retry", f"{type(exc).__name__}: {exc}"
        except httpx.HTTPError as exc:
            # Malformed URL, too many redirects — not going to fix itself.
            return "permanent", f"{type(exc).__name__}: {exc}"

        status = response.status_code
        if 200 <= status < 300:
            return "ok", ""
        if status in AUTH_STATUS:
            self._warn_once(
                f"auth_{status}",
                f"llm-metrics: ingest API rejected the API key (HTTP {status}); "
                "traces will be dropped",
            )
            return "permanent", f"HTTP {status}"
        if status in RETRYABLE_STATUS:
            return "retry", f"HTTP {status}|{response.headers.get('Retry-After', '')}"
        if 500 <= status < 600:
            return "retry", f"HTTP {status}|"
        return "permanent", f"HTTP {status}"

    def _backoff(self, attempt: int, detail: str) -> float:
        """Exponential backoff with full jitter, or the server's Retry-After.

        Full jitter (uniform over ``[0, delay]``) rather than plain exponential:
        when an ingest API comes back from an outage, every SDK instance that
        was retrying in lockstep would otherwise hit it at the same instant.
        """
        retry_after = self._parse_retry_after(detail)
        if retry_after is not None:
            return min(retry_after, self.backoff_max)
        delay = min(self.backoff_max, self.backoff_base * (2**attempt))
        return random.uniform(0, delay)

    @staticmethod
    def _parse_retry_after(detail: str) -> float | None:
        """Read the seconds form of ``Retry-After``.

        The HTTP-date form is ignored deliberately: honouring it means trusting
        clock sync between us and the server, and falling back to our own
        backoff is strictly safer than sleeping for a wrong duration.
        """
        _, _, raw = detail.partition("|")
        if not raw:
            return None
        try:
            seconds = float(raw.strip())
        except ValueError:
            return None
        return seconds if seconds >= 0 else None

    def _fail(self, payload: list[dict[str, Any]], detail: str, *, permanent: bool = False) -> None:
        self.stats.dropped_events += len(payload)
        self.stats.last_error = detail
        if permanent:
            self.stats.permanent_failures += 1
        self._debug(f"llm-metrics: dropped {len(payload)} event(s): {detail}")

    @staticmethod
    def _remaining(deadline: Deadline | None) -> float | None:
        return None if deadline is None else deadline.remaining()

    # ------------------------------------------------------------- serialisation

    def _encode(self, payload: list[dict[str, Any]]) -> bytes | None:
        """Serialise the envelope, degrading gracefully on odd values.

        ``default=_stringify`` matters because ``metadata`` holds whatever the
        user put there. A stray ``datetime`` or model object should cost that
        one field its structure, not cost the whole batch.
        """
        try:
            return json.dumps({"events": payload}, default=_stringify).encode("utf-8")
        except Exception as exc:  # noqa: BLE001 - rule 2
            self.stats.last_error = f"encode failed: {type(exc).__name__}: {exc}"
            return None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-SDK-Version": __version__,
            "User-Agent": f"llm-metrics-python/{__version__}",
        }

    # -------------------------------------------------------------- http client

    def _client(self) -> httpx.Client:
        """Lazily build the pooled client, on the flush thread that uses it."""
        if self._http is None:
            self._http = httpx.Client(
                headers=self._headers(),
                timeout=self.timeout,
                transport=self._transport,
            )
        return self._http

    def close(self) -> None:
        """Release the connection pool. Safe to call more than once."""
        self._closed = True
        http, self._http = self._http, None
        if http is not None:
            # Rule 2: a pool that fails to close is not worth an exception on
            # the way out of a user's process.
            with contextlib.suppress(Exception):
                http.close()

    def _register_fork_handler(self) -> None:
        if not hasattr(os, "register_at_fork"):
            return
        os.register_at_fork(after_in_child=self._reset_after_fork)

    def _reset_after_fork(self) -> None:  # pragma: no cover - requires fork
        # Drop the inherited client without closing it. The child's socket file
        # descriptors point at the *parent's* live connections; closing them
        # would send a FIN on a connection the parent is still using. Leaking
        # them until the child exits is the lesser evil.
        self._http = None
        self.stats = ClientStats()

    # -------------------------------------------------------------- diagnostics

    def _warn_once(self, key: str, message: str) -> None:
        """Configuration problems go to stderr — once — even without debug.

        A silently inert SDK is the worst outcome for a user who thinks they
        are collecting traces. Runtime failures stay quiet; setup failures do not.
        """
        if key in self._warned:
            return
        self._warned.add(key)
        print(message, file=sys.stderr)

    def _debug(self, message: str) -> None:
        if os.environ.get(ENV_DEBUG):
            print(message, file=sys.stderr)

    # -------------------------------------------------------------- context mgr

    def __enter__(self) -> IngestClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _stringify(value: object) -> str:
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - rule 2; a broken __str__ must not lose the batch
        return f"<unserialisable {type(value).__name__}>"
