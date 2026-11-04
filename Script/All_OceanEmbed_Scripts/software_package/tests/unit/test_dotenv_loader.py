"""
Unit tests for the OceanEmbed .env loader (load_dotenv in config/loader.py).

Tests cover:
- Basic KEY=VALUE loading
- Comments and blank lines skipped
- Existing os.environ values are NOT overridden
- Empty values in .env are skipped (template lines like KEY=)
- Surrounding quotes stripped from values
- Sensitive key values are never logged
- Credentials not present in os.environ when not configured
- .env.example file contains no real secrets (all active values are empty)
- load_dotenv returns an empty dict when .env does not exist
- load_dotenv returns only newly-loaded keys
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from oceanembed.config.loader import load_dotenv, _is_sensitive_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dotenv(tmp_path: Path, content: str) -> Path:
    """Write content to a .env file in tmp_path and return its path."""
    p = tmp_path / ".env"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Basic loading behaviour
# ---------------------------------------------------------------------------

class TestLoadDotenvBasic:
    def test_loads_simple_key_value(self, tmp_path):
        p = _write_dotenv(tmp_path, "MY_TEST_VAR_XYZ=hello\n")
        result = load_dotenv(p)
        assert "MY_TEST_VAR_XYZ" in result
        assert os.environ.get("MY_TEST_VAR_XYZ") == "hello"
        # cleanup
        del os.environ["MY_TEST_VAR_XYZ"]

    def test_returns_loaded_keys(self, tmp_path):
        p = _write_dotenv(tmp_path, "DOTENV_TEST_A=1\nDOTENV_TEST_B=2\n")
        result = load_dotenv(p)
        assert "DOTENV_TEST_A" in result
        assert "DOTENV_TEST_B" in result
        del os.environ["DOTENV_TEST_A"]
        del os.environ["DOTENV_TEST_B"]

    def test_skips_comment_lines(self, tmp_path):
        p = _write_dotenv(tmp_path, "# THIS_IS_A_COMMENT=secret\n")
        result = load_dotenv(p)
        assert "THIS_IS_A_COMMENT" not in result
        assert "THIS_IS_A_COMMENT" not in os.environ

    def test_skips_blank_lines(self, tmp_path):
        p = _write_dotenv(tmp_path, "\n\n   \n")
        result = load_dotenv(p)
        assert result == {}

    def test_skips_lines_without_equals(self, tmp_path):
        p = _write_dotenv(tmp_path, "NOTAKEYVALUE\n")
        result = load_dotenv(p)
        assert "NOTAKEYVALUE" not in result

    def test_skips_empty_values(self, tmp_path):
        """Lines like KEY= (template placeholders) must not set empty strings."""
        p = _write_dotenv(tmp_path, "EMPTY_KEY_DOTENV_TEST=\n")
        result = load_dotenv(p)
        assert "EMPTY_KEY_DOTENV_TEST" not in result
        assert "EMPTY_KEY_DOTENV_TEST" not in os.environ

    def test_strips_double_quotes(self, tmp_path):
        p = _write_dotenv(tmp_path, 'QUOTED_TEST_VAR="hello world"\n')
        result = load_dotenv(p)
        assert result.get("QUOTED_TEST_VAR") == "hello world"
        del os.environ["QUOTED_TEST_VAR"]

    def test_strips_single_quotes(self, tmp_path):
        p = _write_dotenv(tmp_path, "SINGLEQ_TEST_VAR='hello world'\n")
        result = load_dotenv(p)
        assert result.get("SINGLEQ_TEST_VAR") == "hello world"
        del os.environ["SINGLEQ_TEST_VAR"]

    def test_strips_surrounding_whitespace_from_value(self, tmp_path):
        p = _write_dotenv(tmp_path, "SPACED_TEST_VAR=  hello  \n")
        result = load_dotenv(p)
        assert result.get("SPACED_TEST_VAR") == "hello"
        del os.environ["SPACED_TEST_VAR"]

    def test_returns_empty_dict_when_file_not_found(self, tmp_path):
        result = load_dotenv(tmp_path / "nonexistent.env")
        assert result == {}

    def test_handles_mixed_content(self, tmp_path):
        content = (
            "# comment\n"
            "\n"
            "MIXED_A=value_a\n"
            "# another comment\n"
            "MIXED_B=value_b\n"
            "MIXED_EMPTY=\n"
        )
        p = _write_dotenv(tmp_path, content)
        result = load_dotenv(p)
        assert "MIXED_A" in result
        assert "MIXED_B" in result
        assert "MIXED_EMPTY" not in result
        del os.environ["MIXED_A"]
        del os.environ["MIXED_B"]


# ---------------------------------------------------------------------------
# Priority: existing os.environ values must not be overridden
# ---------------------------------------------------------------------------

class TestLoadDotenvPriority:
    def test_does_not_override_existing_env_var(self, tmp_path):
        os.environ["PRIORITY_TEST_VAR"] = "original"
        p = _write_dotenv(tmp_path, "PRIORITY_TEST_VAR=from_dotenv\n")
        result = load_dotenv(p)
        # Should NOT be in result (was already set)
        assert "PRIORITY_TEST_VAR" not in result
        # Value in os.environ must remain the original
        assert os.environ["PRIORITY_TEST_VAR"] == "original"
        del os.environ["PRIORITY_TEST_VAR"]

    def test_shell_export_wins_over_dotenv(self, tmp_path):
        """Simulates: shell export before backend start beats .env file."""
        os.environ["SHELL_EXPORT_TEST"] = "shell_value"
        p = _write_dotenv(tmp_path, "SHELL_EXPORT_TEST=dotenv_value\n")
        load_dotenv(p)
        assert os.environ["SHELL_EXPORT_TEST"] == "shell_value"
        del os.environ["SHELL_EXPORT_TEST"]

    def test_returns_only_newly_loaded_keys(self, tmp_path):
        """Keys already in os.environ must not appear in the return value."""
        os.environ["ALREADY_SET_DOTENV"] = "preexisting"
        p = _write_dotenv(tmp_path, (
            "ALREADY_SET_DOTENV=should_be_ignored\n"
            "NEW_DOTENV_KEY=new_value\n"
        ))
        result = load_dotenv(p)
        assert "ALREADY_SET_DOTENV" not in result
        assert "NEW_DOTENV_KEY" in result
        del os.environ["ALREADY_SET_DOTENV"]
        del os.environ["NEW_DOTENV_KEY"]


# ---------------------------------------------------------------------------
# Sensitive key detection
# ---------------------------------------------------------------------------

class TestSensitiveKeyDetection:
    def test_password_key_is_sensitive(self):
        assert _is_sensitive_key("COPERNICUSMARINE_SERVICE_PASSWORD") is True

    def test_username_key_is_not_sensitive(self):
        # Username is not inherently secret (it's often visible in logs)
        # — the secret is the PASSWORD.  This test documents the behaviour.
        assert _is_sensitive_key("COPERNICUSMARINE_SERVICE_USERNAME") is False

    def test_secret_key_is_sensitive(self):
        assert _is_sensitive_key("MY_API_SECRET") is True

    def test_token_key_is_sensitive(self):
        assert _is_sensitive_key("AUTH_TOKEN") is True

    def test_earthdata_password_is_sensitive(self):
        assert _is_sensitive_key("EARTHDATA_PASSWORD") is True

    def test_plain_path_not_sensitive(self):
        assert _is_sensitive_key("OCEANEMBED_DATA_ROOT") is False

    def test_allowed_origins_not_sensitive(self):
        assert _is_sensitive_key("OCEANEMBED_ALLOWED_ORIGINS") is False

    def test_sensitive_values_redacted_in_log(self, tmp_path, caplog):
        """
        load_dotenv must not log the raw value of sensitive keys.
        The return dict may contain the value (it's an internal return,
        not a log line) — but the debug log must show '<set>' not the value.
        """
        import logging
        p = _write_dotenv(
            tmp_path,
            "COPERNICUSMARINE_SERVICE_PASSWORD=super_secret_value\n"
        )
        with caplog.at_level(logging.DEBUG, logger="oceanembed.config.loader"):
            result = load_dotenv(p)
        # The log output must never contain the real value
        assert "super_secret_value" not in caplog.text, (
            "Sensitive value must not appear in log output from load_dotenv"
        )
        # cleanup
        os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)


# ---------------------------------------------------------------------------
# CMEMS credential configuration checks (using fake values)
# ---------------------------------------------------------------------------

class TestCmemsCredentialConfiguration:
    """
    Tests that verify credential detection logic.
    Uses FAKE credential values — never real CMEMS credentials.
    """

    def test_both_credentials_detected_when_set(self):
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = "fake_test_user"
        os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = "fake_test_pass"
        try:
            result = GlobCurrentAdapter().check_credentials()
            assert result["has_username"] is True
            assert result["has_password"] is True
            assert result["credentials_available"] is True
        finally:
            del os.environ["COPERNICUSMARINE_SERVICE_USERNAME"]
            del os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"]

    def test_missing_password_detected(self):
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = "fake_test_user"
        try:
            result = GlobCurrentAdapter().check_credentials()
            assert result["has_username"] is True
            assert result["has_password"] is False
            assert result["credentials_available"] is False
            assert "password" in result["message"].lower() or "PASSWORD" in result["message"]
        finally:
            del os.environ["COPERNICUSMARINE_SERVICE_USERNAME"]
            if saved_p:
                os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p

    def test_missing_username_detected(self):
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = "fake_test_pass"
        try:
            result = GlobCurrentAdapter().check_credentials()
            assert result["has_username"] is False
            assert result["has_password"] is True
            assert result["credentials_available"] is False
        finally:
            del os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"]
            if saved_u:
                os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u

    def test_message_never_contains_fake_password(self):
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = "fake_user_xyz"
        os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = "fake_pass_xyz"
        try:
            result = GlobCurrentAdapter().check_credentials()
            assert "fake_user_xyz" not in result["message"]
            assert "fake_pass_xyz" not in result["message"]
        finally:
            del os.environ["COPERNICUSMARINE_SERVICE_USERNAME"]
            del os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"]

    def test_both_missing_returns_clear_message(self):
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            result = GlobCurrentAdapter().check_credentials()
            assert result["credentials_available"] is False
            # Message should tell the user what to do, not just say "no"
            msg = result["message"].lower()
            assert "copernicusmarine_service_username" in msg or "username" in msg
            assert "copernicusmarine_service_password" in msg or "password" in msg
        finally:
            if saved_u:
                os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u
            if saved_p:
                os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p


# ---------------------------------------------------------------------------
# .env.example safety check
# ---------------------------------------------------------------------------

class TestDotenvExampleSafety:
    """Verify .env.example contains no real secrets."""

    @pytest.fixture
    def example_path(self) -> Path:
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            candidate = parent / ".env.example"
            if candidate.exists():
                return candidate
        pytest.skip(".env.example not found")

    def test_example_file_exists(self, example_path):
        assert example_path.exists()

    def test_cmems_username_value_is_empty(self, example_path):
        """COPERNICUSMARINE_SERVICE_USERNAME must have no value in .env.example."""
        content = example_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("COPERNICUSMARINE_SERVICE_USERNAME="):
                value = stripped.split("=", 1)[1].strip()
                assert value == "", (
                    f"COPERNICUSMARINE_SERVICE_USERNAME must be empty in .env.example, "
                    f"got: {value!r}"
                )

    def test_cmems_password_value_is_empty(self, example_path):
        """COPERNICUSMARINE_SERVICE_PASSWORD must have no value in .env.example."""
        content = example_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("COPERNICUSMARINE_SERVICE_PASSWORD="):
                value = stripped.split("=", 1)[1].strip()
                assert value == "", (
                    f"COPERNICUSMARINE_SERVICE_PASSWORD must be empty in .env.example, "
                    f"got: {value!r}"
                )

    def test_example_contains_no_obvious_passwords(self, example_path):
        """
        Heuristic: no non-comment line in .env.example should have a non-empty
        value for any key that ends with PASSWORD, SECRET, or TOKEN.
        """
        content = example_path.read_text(encoding="utf-8")
        for lineno, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip().upper()
            value = value.strip().strip('"').strip("'")
            if any(key.endswith(suffix) for suffix in ("PASSWORD", "SECRET", "TOKEN")):
                assert value == "", (
                    f"Line {lineno}: sensitive key {key!r} has a non-empty value "
                    f"in .env.example — remove or comment it out"
                )

    def test_example_not_ignored_by_git(self, example_path):
        """
        .env.example must be trackable by git (not in .gitignore).
        Runs git check-ignore and expects a non-zero exit code (= not ignored).
        """
        import subprocess
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(example_path)],
            capture_output=True,
            cwd=example_path.parent,
        )
        # Exit code 1 means "not ignored" — that's what we want
        assert result.returncode != 0, (
            f".env.example is being ignored by git — check .gitignore rules. "
            f"It must remain tracked."
        )
