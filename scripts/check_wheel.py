#!/usr/bin/env python
"""Assert what pip actually delivers, rather than what pyproject.toml promises.

Run against a built ``dist/``. Checks two things the rest of the suite cannot:

* ``py.typed`` ships. Losing it makes every type hint in the package invisible
  to downstream ``mypy`` while lint, tests, and ``twine check`` all stay green
  — a silent regression with no other alarm.
* A base install pulls in ``httpx`` and nothing else. The ``[openai]`` and
  ``[langchain]`` extras are load-bearing promises; an accidental top-level
  import of either would turn them into hard dependencies without any test
  noticing, because the dev environment always has both installed.

Used by ``make check-package`` and by the CI build job, so the two cannot drift.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REQUIRED_MEMBERS = ("llmobserve/py.typed", "llmobserve/__init__.py")

#: Run inside a throwaway venv holding only the built wheel and its
#: dependencies. Both assertions are about the installed artefact, not the
#: source tree, so they cannot be made from the dev environment.
SMOKE_TEST = """
import importlib.util as util
import threading

before = threading.active_count()
import llmobserve

assert threading.active_count() == before, (
    "importing llmobserve started a background thread; the pipeline must stay "
    "lazy so `import llmobserve` is safe in anything that forks after import"
)
for extra in ("openai", "langchain_core"):
    assert util.find_spec(extra) is None, (
        f"{extra} was installed by the base package; it is supposed to be an "
        f"optional extra, so something imports it at the top level"
    )
print(f"    installed wheel imports cleanly: llmobserve {llmobserve.__version__}")
"""


def fail(message: str) -> None:
    print(f"FAIL  {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def find_wheel(dist: Path) -> Path:
    wheels = sorted(glob.glob(str(dist / "*.whl")))
    if not wheels:
        fail(f"no wheel in {dist}/ — run `make build` first")
    if len(wheels) > 1:
        fail(f"{len(wheels)} wheels in {dist}/, so the check is ambiguous: {wheels}")
    return Path(wheels[0])


def check_contents(wheel: Path) -> None:
    names = set(zipfile.ZipFile(wheel).namelist())
    missing = [member for member in REQUIRED_MEMBERS if member not in names]
    if missing:
        fail(f"{wheel.name} is missing {missing}\n      contents: {sorted(names)}")
    modules = sorted(n for n in names if n.endswith(".py"))
    print(f"    {wheel.name}: py.typed present, {len(modules)} modules", flush=True)


def check_install(wheel: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        python = env / "bin" / "python"
        if not python.exists():  # Windows layout
            python = env / "Scripts" / "python.exe"
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", str(wheel)],
            check=True,
        )
        subprocess.run([str(python), "-c", SMOKE_TEST], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", default="dist", type=Path, help="directory holding the wheel")
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="skip the fresh-venv install check, which needs network to fetch httpx",
    )
    args = parser.parse_args()

    wheel = find_wheel(args.dist)
    print("packaging checks", flush=True)
    check_contents(wheel)
    if args.no_install:
        print("    skipped install check (--no-install)", flush=True)
    else:
        check_install(wheel)
    print("packaging checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
