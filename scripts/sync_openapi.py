#!/usr/bin/env python
"""Refresh the vendored server contract at ``tests/contract/openapi.json``.

    python scripts/sync_openapi.py                 # from ../llm-observe
    python scripts/sync_openapi.py path/to/openapi.json
    python scripts/sync_openapi.py https://.../openapi.json

The server generates the file with ``make openapi``. Copying it here keeps
``tests/test_contract.py`` deterministic and offline; run this after a server
schema change and commit the result. Exit status 3 means the file changed, so
a CI job can fail on a stale copy with ``sync && git diff --exit-code``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "tests" / "contract" / "openapi.json"
DEFAULT_SOURCE = ROOT.parent / "llm-observe" / "apps" / "api" / "openapi.json"


def _read(source: str) -> str:
    if source.startswith(("http://", "https://")):
        import httpx

        response = httpx.get(source, timeout=30.0)
        response.raise_for_status()
        return response.text
    return Path(source).read_text()


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_SOURCE)
    try:
        raw = _read(source)
    except (OSError, ValueError) as exc:
        print(f"cannot read {source}: {exc}", file=sys.stderr)
        return 2

    document = json.loads(raw)  # fail loudly on a half-written file
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"

    before = TARGET.read_text() if TARGET.exists() else None
    if before == text:
        print(f"unchanged: {TARGET.relative_to(ROOT)} already matches {source}")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text)
    print(f"updated: {TARGET.relative_to(ROOT)} from {source} (openapi {document.get('openapi')})")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
