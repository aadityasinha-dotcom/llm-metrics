"""Single source of truth for the SDK version.

Lives in its own module so ``client.py`` can stamp ``X-SDK-Version`` without
importing the package root, and so ``pyproject.toml`` can read it via hatch
instead of duplicating the number.
"""

__version__ = "0.2.0"
