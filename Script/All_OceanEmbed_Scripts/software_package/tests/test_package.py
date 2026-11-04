"""
Minimal package smoke test.

Verifies that the oceanembed package imports correctly and exposes
the expected version attribute.
"""

import importlib


def test_package_imports():
    """The oceanembed package must be importable without errors."""
    mod = importlib.import_module("oceanembed")
    assert mod is not None


def test_package_version():
    """The package must expose a __version__ string."""
    import oceanembed

    assert hasattr(oceanembed, "__version__")
    assert isinstance(oceanembed.__version__, str)
    assert len(oceanembed.__version__) > 0


def test_package_version_format():
    """The version must follow a basic semver-like pattern (MAJOR.MINOR.PATCH)."""
    import oceanembed

    parts = oceanembed.__version__.split(".")
    assert len(parts) == 3, f"Expected 3 version parts, got: {oceanembed.__version__}"
    for part in parts:
        assert part.isdigit(), f"Version part is not numeric: {part!r}"
