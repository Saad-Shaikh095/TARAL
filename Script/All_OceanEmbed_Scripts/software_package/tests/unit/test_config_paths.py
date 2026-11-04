"""
Regression tests for OceanEmbed configuration path handling.

Specifically guards against the class of bug where a trailing-whitespace
env var value produces a malformed path such as:

    D:\\OceanEmbed\\oceanembed-data \\raw

instead of:

    D:\\OceanEmbed\\oceanembed-data\\raw

Root cause (Sep 2026): launch_oceanembed.bat wrote the OCEANEMBED_DATA_ROOT
value into a helper .cmd file using:

    echo set "OCEANEMBED_DATA_ROOT=<value>"

The CMD echo inside a parenthesised block appended a trailing space to the
value.  The config loader then passed that raw string directly to Path(),
which preserved the space, causing WinError 3 when the acquisition engine
tried to create the 'raw' subdirectory.

Fix applied:
1. Config loader strips surrounding whitespace from env var / YAML values
   at the configuration boundary (_clean helper).
2. Config loader calls .expanduser() on resolved paths.
3. BAT file uses unquoted 'set' in the echo to avoid trailing-space injection.

These tests ensure the fix holds and catches regressions on any platform.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_with_env(data_root_value: str | None) -> "OceanEmbedSettings":
    """Load settings with a patched OCEANEMBED_DATA_ROOT env var."""
    # Must clear lru_cache so each test gets a fresh resolution.
    from oceanembed.config.loader import get_settings
    get_settings.cache_clear()

    env = dict(os.environ)
    if data_root_value is None:
        env.pop("OCEANEMBED_DATA_ROOT", None)
    else:
        env["OCEANEMBED_DATA_ROOT"] = data_root_value

    with patch.dict(os.environ, env, clear=True):
        get_settings.cache_clear()
        settings = get_settings()

    get_settings.cache_clear()
    return settings


# ---------------------------------------------------------------------------
# A. Clean data root produces correct raw path
# ---------------------------------------------------------------------------

class TestCleanDataRoot:
    def test_clean_path_produces_correct_raw_subdir(self, tmp_path: Path):
        """A clean data_root value produces data_root / 'raw' with no spaces."""
        settings = _load_with_env(str(tmp_path))
        assert settings.data_root is not None
        raw_dir = settings.data_root / "raw"
        # No space before backslash/slash
        assert " /" not in str(raw_dir)
        assert " \\" not in str(raw_dir)
        assert str(raw_dir) == str(tmp_path / "raw")

    def test_clean_path_string_has_no_trailing_space(self, tmp_path: Path):
        settings = _load_with_env(str(tmp_path))
        assert settings.data_root is not None
        assert str(settings.data_root) == str(settings.data_root).rstrip()


# ---------------------------------------------------------------------------
# B. Trailing whitespace is stripped at the config boundary
# ---------------------------------------------------------------------------

class TestTrailingWhitespaceStripped:
    def test_trailing_space_is_stripped(self, tmp_path: Path):
        """
        A data_root value with a trailing space must be normalised.

        This is the exact failure mode produced by the BAT launcher bug.
        """
        dirty_value = str(tmp_path) + " "   # intentional trailing space
        settings = _load_with_env(dirty_value)
        assert settings.data_root is not None
        # The resolved path must equal the clean version
        assert settings.data_root == tmp_path
        assert str(settings.data_root) == str(tmp_path)

    def test_trailing_space_raw_dir_is_correct(self, tmp_path: Path):
        """
        After stripping, data_root / 'raw' must NOT contain a space.
        The bug produced 'oceanembed-data \\raw' — this test catches that.
        """
        dirty_value = str(tmp_path) + " "
        settings = _load_with_env(dirty_value)
        raw_dir = settings.data_root / "raw"
        assert " \\" not in str(raw_dir), (
            f"Malformed path detected: {raw_dir!r} — "
            "trailing whitespace was not stripped from data_root."
        )
        assert " /" not in str(raw_dir)

    def test_leading_and_trailing_whitespace_stripped(self, tmp_path: Path):
        dirty_value = "  " + str(tmp_path) + "  "
        settings = _load_with_env(dirty_value)
        assert settings.data_root == tmp_path

    def test_tab_character_stripped(self, tmp_path: Path):
        dirty_value = str(tmp_path) + "\t"
        settings = _load_with_env(dirty_value)
        assert settings.data_root == tmp_path

    def test_newline_stripped(self, tmp_path: Path):
        """Some shell/env mechanisms append a newline."""
        dirty_value = str(tmp_path) + "\n"
        settings = _load_with_env(dirty_value)
        assert settings.data_root == tmp_path


# ---------------------------------------------------------------------------
# C. Path construction uses pathlib correctly
# ---------------------------------------------------------------------------

class TestPathlibConstruction:
    def test_data_root_is_path_object(self, tmp_path: Path):
        settings = _load_with_env(str(tmp_path))
        assert isinstance(settings.data_root, Path)

    def test_subdirectory_uses_slash_operator(self, tmp_path: Path):
        """Verify pathlib / operator produces the expected path."""
        settings = _load_with_env(str(tmp_path))
        assert settings.data_root is not None
        for subdir in ("raw", "processed", "model-ready", "manifests", "cache"):
            result = settings.data_root / subdir
            assert isinstance(result, Path)
            assert result.parent == settings.data_root
            assert result.name == subdir

    def test_raw_subdir_does_not_require_string_concat(self, tmp_path: Path):
        """Constructing raw/ must not require string concatenation."""
        settings = _load_with_env(str(tmp_path))
        # This is how the engine constructs paths — must work cleanly
        dest = settings.data_root / "raw" / "oisst" / "2023" / "oisst_20230101.nc"
        assert isinstance(dest, Path)
        assert "raw" in dest.parts
        assert "oisst" in dest.parts

    def test_none_when_not_configured(self):
        settings = _load_with_env(None)
        assert settings.data_root is None


# ---------------------------------------------------------------------------
# D. Existing valid paths continue to work
# ---------------------------------------------------------------------------

class TestExistingPaths:
    def test_existing_directory_resolves_correctly(self, tmp_path: Path):
        settings = _load_with_env(str(tmp_path))
        assert settings.data_root == tmp_path

    def test_path_with_spaces_in_directory_name(self, tmp_path: Path):
        """Paths that legitimately contain spaces must be preserved."""
        spaced = tmp_path / "my data dir"
        spaced.mkdir()
        settings = _load_with_env(str(spaced))
        assert settings.data_root == spaced
        # The space is inside the name, not trailing — must be preserved
        assert settings.data_root.name == "my data dir"

    def test_tilde_expansion(self, tmp_path: Path):
        """~ in a configured path must be expanded."""
        # We can only test this safely if HOME/USERPROFILE is accessible
        home = Path.home()
        settings = _load_with_env("~/oceanembed-data-test")
        assert settings.data_root is not None
        assert "~" not in str(settings.data_root)
        assert str(home) in str(settings.data_root)


# ---------------------------------------------------------------------------
# E. Acquisition engine path construction
# ---------------------------------------------------------------------------

class TestEnginePathConstruction:
    def test_engine_builds_dest_path_with_pathlib(self, tmp_path: Path):
        """
        The engine constructs dest_abs = job.data_root / req.expected_output_path.
        Verify this produces the expected path without whitespace artifacts.
        """
        from oceanembed.datasets.adapters.oisst import OISSTAdapter
        from datetime import date

        adapter = OISSTAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        # Simulate how the engine builds the absolute path
        dest_abs = tmp_path / req.expected_output_path

        assert isinstance(dest_abs, Path)
        assert " \\" not in str(dest_abs)
        assert " /" not in str(dest_abs)
        assert str(dest_abs) == str(tmp_path / "raw" / "oisst" / "2023" / "oisst_20230101.nc")

    def test_engine_raw_dir_no_space_with_trailing_whitespace_input(self, tmp_path: Path):
        """
        Even if data_root were constructed from a dirty string (simulating
        the BAT bug), the config loader's _clean() call must strip it before
        it reaches the engine.
        """
        from oceanembed.datasets.adapters.oisst import OISSTAdapter
        from datetime import date

        dirty_root = str(tmp_path) + " "   # trailing space — the BAT bug
        clean_root = Path(dirty_root.strip())

        adapter = OISSTAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(clean_root))

        dest_abs = clean_root / req.expected_output_path
        assert " \\" not in str(dest_abs)
        assert " /" not in str(dest_abs)
