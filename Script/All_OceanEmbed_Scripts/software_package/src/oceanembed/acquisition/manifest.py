"""
OceanEmbed Acquisition Manifest.

Records what was acquired, when, from where, and what happened.

Demo implementation: JSON Lines file per dataset, one record per file.
Location: <data_root>/manifests/<dataset_id>.jsonl

A future version may use SQLite or PostgreSQL.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path


class ManifestStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class ManifestEntry:
    dataset_id: str
    request_date: date
    source_url: str
    local_path: str
    product_id: str
    status: ManifestStatus
    file_size_bytes: int | None = None
    md5_checksum: str | None = None
    acquired_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["request_date"] = self.request_date.isoformat()
        d["acquired_at"] = self.acquired_at.isoformat()
        d["status"] = self.status.value
        return d


class ManifestStore:
    """
    Append-only manifest store for acquisition records.

    Writes one JSON line per ManifestEntry to
    <data_root>/manifests/<dataset_id>.jsonl
    """

    def __init__(self, data_root: Path) -> None:
        self._data_root = Path(data_root)
        self._manifest_dir = self._data_root / "manifests"

    def _path(self, dataset_id: str) -> Path:
        return self._manifest_dir / f"{dataset_id}.jsonl"

    def record(self, entry: ManifestEntry) -> None:
        """Append a manifest entry to the dataset's manifest file."""
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._path(entry.dataset_id)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def read_all(self, dataset_id: str) -> list[dict]:
        """Read all manifest entries for a dataset."""
        p = self._path(dataset_id)
        if not p.exists():
            return []
        entries = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries
