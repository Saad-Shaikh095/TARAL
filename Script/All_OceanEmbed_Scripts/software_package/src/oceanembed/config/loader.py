"""
OceanEmbed configuration loader.

Resolves the data_root and other workspace paths from:
  1. Environment variable OCEANEMBED_DATA_ROOT (highest priority)
  2. configs/environments/<hostname>.yaml  (machine-specific override)
  3. configs/oceanembed.yaml               (project default, usually null)

.env file loading
-----------------
If a .env file exists at the repository root (same directory as pyproject.toml),
it is loaded automatically before environment-variable lookups.  This means
credentials and paths set in .env are available to the entire process — including
the copernicusmarine client, which reads its own credentials directly from
os.environ at import time.

Loading rules:
  - Only KEY=VALUE lines are processed (comments and blank lines are skipped).
  - Values already present in os.environ are NOT overridden (os.environ wins).
    This means shell exports and CI environment variables always take precedence
    over the local .env file.
  - Values are stripped of surrounding whitespace and optional surrounding quotes
    (single or double).
  - Credential values are NEVER logged or printed.
  - If .env does not exist, loading is silently skipped.

Usage
-----
    from oceanembed.config.loader import get_settings
    settings = get_settings()
    print(settings.data_root)   # Path or None if not configured
"""

from __future__ import annotations

import logging
import os
import socket
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive key prefixes — values for these keys are never logged
# ---------------------------------------------------------------------------
_SENSITIVE_PREFIXES = (
    "PASSWORD",
    "SECRET",
    "TOKEN",
    "KEY",
    "CREDENTIAL",
    "AUTH",
)


def _is_sensitive_key(key: str) -> bool:
    """Return True if the key name suggests it holds a secret value."""
    upper = key.upper()
    return any(upper.endswith(p) or p in upper for p in _SENSITIVE_PREFIXES)


# ---------------------------------------------------------------------------
# .env loader (no external dependency)
# ---------------------------------------------------------------------------

def load_dotenv(dotenv_path: Path) -> dict[str, str]:
    """
    Load KEY=VALUE pairs from *dotenv_path* into os.environ.

    Rules
    -----
    - Lines starting with ``#`` are comments and are skipped.
    - Blank lines are skipped.
    - Values already present in os.environ are NOT overridden.
    - Values are stripped of surrounding whitespace and optional quotes.
    - Credential values are never logged.

    Parameters
    ----------
    dotenv_path:
        Path to the .env file.  If the file does not exist, the function
        returns an empty dict without raising.

    Returns
    -------
    dict[str, str]
        The key/value pairs that were *newly loaded* (already-present keys
        not included).  Safe to inspect — sensitive values are redacted.
    """
    if not dotenv_path.exists():
        return {}

    loaded: dict[str, str] = {}

    with open(dotenv_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blanks and comments
            if not line or line.startswith("#"):
                continue

            # Must contain an = sign
            if "=" not in line:
                continue

            key, _, raw_value = line.partition("=")
            key = key.strip()
            raw_value = raw_value.strip()

            # Strip optional surrounding quotes (single or double)
            if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in ('"', "'"):
                raw_value = raw_value[1:-1]

            # Skip if key is empty or already in the environment
            if not key:
                continue
            if key in os.environ:
                logger.debug(
                    "load_dotenv: %s already set in environment, not overriding.", key
                )
                continue

            # Skip empty values — don't clobber a genuinely unset variable
            # with an empty string from the template.
            if not raw_value:
                continue

            os.environ[key] = raw_value
            loaded[key] = raw_value

    # Log which keys were loaded, but never their values
    if loaded:
        safe_report = {
            k: "<set>" if _is_sensitive_key(k) else v
            for k, v in loaded.items()
        }
        logger.debug("load_dotenv: loaded from %s: %s", dotenv_path, safe_report)

    return loaded


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class OceanEmbedSettings(BaseModel):
    """Resolved runtime settings for OceanEmbed."""

    model_config = {"extra": "allow"}

    data_root: Path | None = Field(
        None,
        description=(
            "Root directory of the OceanEmbed data workspace. "
            "Set via OCEANEMBED_DATA_ROOT environment variable, "
            "or configs/environments/<hostname>.yaml."
        ),
    )
    research_root: Path | None = Field(None)
    experiments_root: Path | None = Field(None)

    # Registry dir — resolved relative to the software repo
    registry_dir: Path = Field(
        ...,
        description="Path to configs/datasets/ registry directory.",
    )
    project_config_path: Path = Field(
        ...,
        description="Path to configs/oceanembed.yaml.",
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """
    Walk up from this file to locate the repository root.
    Identified by the presence of pyproject.toml.
    
    In Vercel serverless runtime, the OCEANEMBED_REPO_ROOT environment
    variable is set by web/api/index.py to provide an absolute path.
    """
    # Check if running in Vercel serverless environment
    if "OCEANEMBED_REPO_ROOT" in os.environ:
        return Path(os.environ["OCEANEMBED_REPO_ROOT"])
    
    # Local development: walk up from this file
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we're somewhere inside oceanembed-software/
    return Path.cwd()


@lru_cache(maxsize=1)
def get_settings() -> OceanEmbedSettings:
    """
    Load and return the resolved OceanEmbed settings.

    On first call, loads a .env file from the repository root (if present)
    into os.environ so that credentials and path overrides are available
    before any lookups occur.

    Resolution order for data_root:
    1. OCEANEMBED_DATA_ROOT environment variable
    2. configs/environments/<hostname>.yaml  paths.data_root
    3. configs/oceanembed.yaml               paths.data_root  (usually null)

    Returns
    -------
    OceanEmbedSettings
        Resolved settings.  data_root may be None if not configured.
    """
    repo_root = _find_repo_root()

    # Load .env FIRST — before any os.environ lookups and before
    # copernicusmarine reads its own credential variables.
    # Existing os.environ values are never overridden by .env.
    dotenv_path = repo_root / ".env"
    load_dotenv(dotenv_path)
    project_config_path = repo_root / "configs" / "oceanembed.yaml"
    registry_dir = repo_root / "configs" / "datasets"
    environments_dir = repo_root / "configs" / "environments"

    # --- Load project config ---
    project_cfg: dict = {}
    if project_config_path.exists():
        with open(project_config_path, encoding="utf-8") as f:
            project_cfg = yaml.safe_load(f) or {}

    paths_cfg = project_cfg.get("paths", {}) or {}

    # --- Try machine-specific override ---
    hostname = socket.gethostname().lower()
    env_file = environments_dir / f"{hostname}.yaml"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            env_cfg = yaml.safe_load(f) or {}
        paths_cfg.update(env_cfg.get("paths", {}) or {})

    # --- Env var takes highest priority ---
    # Strip surrounding whitespace from raw string values at the configuration
    # boundary. This prevents trailing spaces introduced by shell launchers,
    # YAML editors, or .env files from producing malformed paths such as
    #   "D:\OceanEmbed\oceanembed-data \raw"
    # instead of the intended
    #   "D:\OceanEmbed\oceanembed-data\raw"
    def _clean(v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    data_root_raw: str | None = _clean(
        os.environ.get("OCEANEMBED_DATA_ROOT")
        or paths_cfg.get("data_root")
    )
    research_root_raw: str | None = _clean(
        os.environ.get("OCEANEMBED_RESEARCH_ROOT")
        or paths_cfg.get("research_root")
    )
    experiments_root_raw: str | None = _clean(
        os.environ.get("OCEANEMBED_EXPERIMENTS_ROOT")
        or paths_cfg.get("experiments_root")
    )

    return OceanEmbedSettings(
        data_root=Path(data_root_raw).expanduser() if data_root_raw else None,
        research_root=Path(research_root_raw).expanduser() if research_root_raw else None,
        experiments_root=Path(experiments_root_raw).expanduser() if experiments_root_raw else None,
        registry_dir=registry_dir,
        project_config_path=project_config_path,
    )


def resolve_data_root() -> Path:
    """
    Return the configured data_root.

    Raises
    ------
    RuntimeError
        If data_root is not configured.  Instructs the user to set
        OCEANEMBED_DATA_ROOT.
    """
    settings = get_settings()
    if settings.data_root is None:
        raise RuntimeError(
            "data_root is not configured. "
            "Set the OCEANEMBED_DATA_ROOT environment variable to the path of "
            "your oceanembed-data/ workspace directory. "
            "Example (Windows):  $env:OCEANEMBED_DATA_ROOT = 'D:\\OceanEmbed\\oceanembed-data' "
            "Example (Linux/Mac): export OCEANEMBED_DATA_ROOT=/path/to/oceanembed-data"
        )
    return settings.data_root
