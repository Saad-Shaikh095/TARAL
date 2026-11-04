#!/usr/bin/env python3
"""
OceanEmbed – Portable Python Script Migration Tool
===================================================
Reads migration_manifest.json and moves scripts to their target locations.

Usage
-----
  # Dry-run (default — safe, shows what would happen):
  python tools/maintenance/organize_python_scripts.py

  # Apply the moves for real:
  python tools/maintenance/organize_python_scripts.py --apply

  # Point to repo explicitly (portable across machines):
  python tools/maintenance/organize_python_scripts.py --root D:\\OceanEmbed --apply
  python tools/maintenance/organize_python_scripts.py --root C:\\Users\\user\\OceanEmbed --apply

Rules
-----
- Never overwrites existing files.
- Detects filename collisions.
- Idempotent: already-moved files are skipped, not duplicated.
- Logs every operation to migrate_<timestamp>.log.
- Only uses Python standard library.
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """Walk up until we find a directory that contains .git or migration_manifest.json."""
    candidate = start.resolve()
    for _ in range(10):
        if (candidate / ".git").exists() or (candidate / "tools" / "maintenance" / "migration_manifest.json").exists():
            return candidate
        candidate = candidate.parent
    return start.resolve()


def load_manifest(root: Path) -> dict:
    manifest_path = root / "tools" / "maintenance" / "migration_manifest.json"
    if not manifest_path.exists():
        sys.exit(f"[ERROR] Manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def setup_logging(root: Path, dry_run: bool) -> logging.Logger:
    log_dir = root / "tools" / "maintenance" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dryrun" if dry_run else "apply"
    log_file = log_dir / f"migrate_{ts}_{mode}.log"

    logger = logging.getLogger("migrate")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(root: Path, dry_run: bool) -> None:
    log = setup_logging(root, dry_run)
    manifest = load_manifest(root)

    mode_label = "DRY-RUN" if dry_run else "APPLY"
    log.info("=" * 60)
    log.info("OceanEmbed Migration Tool  |  Mode: %s", mode_label)
    log.info("Repository root: %s", root)
    log.info("=" * 60)

    moves = manifest.get("moves", [])
    log.info("Manifest contains %d move entries.", len(moves))

    results = {"moved": [], "skipped_already_done": [], "skipped_src_missing": [],
               "skipped_collision": [], "failed": []}

    for entry in moves:
        mid  = entry["id"]
        src  = root / Path(entry["src"])
        dst  = root / Path(entry["dst"])

        # --- Source missing? ---
        if not src.exists():
            # Check if dst already exists (migration already ran)
            if dst.exists():
                log.info("[%s] SKIP (already done)  %s", mid, dst.relative_to(root))
                results["skipped_already_done"].append(mid)
            else:
                log.warning("[%s] SKIP src missing, dst also missing: %s", mid, entry["src"])
                results["skipped_src_missing"].append(mid)
            continue

        # --- Destination collision? ---
        if dst.exists():
            log.warning("[%s] SKIP collision – dst already exists: %s", mid, dst.relative_to(root))
            results["skipped_collision"].append(mid)
            continue

        # --- Special: double-dot filename (argo_scientific_audit..py) ---
        # Rename to clean name at destination
        log.info("[%s] %s  ->  %s", mid,
                 src.relative_to(root), dst.relative_to(root))

        if not dry_run:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                results["moved"].append(mid)
                log.debug("[%s] OK", mid)
            except Exception as exc:
                log.error("[%s] FAILED: %s", mid, exc)
                results["failed"].append(mid)
        else:
            results["moved"].append(mid)  # count as "would move" in dry-run

    # --- Summary ---
    log.info("=" * 60)
    log.info("SUMMARY  (%s)", mode_label)
    log.info("  Would move / moved:       %d", len(results["moved"]))
    log.info("  Already done (skipped):   %d", len(results["skipped_already_done"]))
    log.info("  Source missing (skipped): %d", len(results["skipped_src_missing"]))
    log.info("  Collision (skipped):      %d", len(results["skipped_collision"]))
    log.info("  Failed:                   %d", len(results["failed"]))
    log.info("=" * 60)

    if dry_run:
        log.info("This was a DRY-RUN. No files were changed.")
        log.info("Re-run with --apply to perform the moves.")

    if results["failed"]:
        log.error("Some moves failed. Review the log above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OceanEmbed portable Python script migration tool."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Path to repository root. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually move files. Without this flag the script is a dry-run.",
    )
    args = parser.parse_args()

    root = args.root if args.root else find_repo_root(Path(__file__).parent)
    dry_run = not args.apply

    run(root, dry_run)


if __name__ == "__main__":
    main()
