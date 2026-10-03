"""Atomic local experiment archive, independent of the web server and job cache."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ID = re.compile(r"[a-f0-9]{32}\Z")


def _lossless(value: Any) -> Any:
    # Archive payloads are opaque JSON; preserve int64 ids when a browser downloads them.
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    if isinstance(value, dict):
        return {k: _lossless(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_lossless(v) for v in value]
    return value


class RunHistory:
    """Keep the latest 100 experiment records. Disabled archives never touch disk."""

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self._lock = threading.RLock()

    def _path(self, run_id: str) -> Path:
        if not _ID.fullmatch(run_id):
            raise ValueError("Invalid history id.")
        return self.root / f"{run_id}.json"

    def save(self, kind: str, label: str, index: str, data: dict[str, Any]) -> str | None:
        """Atomically archive a completed experiment, returning its id."""
        if not self.enabled:
            return None
        run_id = uuid.uuid4().hex
        record = {
            "id": run_id,
            "kind": kind,
            "label": label[:200],
            "index": index,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write(run_id, record)
            files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime)
            for path in files[:-100]:
                path.unlink(missing_ok=True)
        return run_id

    def _write(self, run_id: str, record: dict[str, Any]) -> None:
        path = self._path(run_id)
        temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temp.write_text(json.dumps(_lossless(record), allow_nan=False) + "\n")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    def get(self, run_id: str) -> dict[str, Any]:
        """Read a record; unavailable/disabled records raise FileNotFoundError."""
        path = self._path(run_id)
        if not self.enabled:
            raise FileNotFoundError(run_id)
        record: dict[str, Any] = json.loads(path.read_text())
        if record.get("id") != run_id or not isinstance(record.get("data"), dict):
            raise ValueError("Invalid history record.")
        return record

    def list(self) -> list[dict[str, Any]]:
        """Newest-first summaries; ignore incomplete or damaged records."""
        if not self.enabled or not self.root.exists():
            return []
        records = []
        with self._lock:
            for path in self.root.glob("*.json"):
                try:
                    record = self.get(path.stem)
                    records.append(
                        {k: record[k] for k in ("id", "kind", "label", "index", "created_at")}
                    )
                except (OSError, ValueError, KeyError):
                    continue
        return sorted(records, key=lambda r: str(r["created_at"]), reverse=True)

    def rename(self, run_id: str, label: str) -> None:
        """Rename an existing experiment without changing its measurements."""
        if not label.strip() or len(label) > 200:
            raise ValueError("Name must contain 1-200 characters.")
        with self._lock:
            record = self.get(run_id)
            record["label"] = label.strip()
            self._write(run_id, record)

    def delete(self, run_id: str) -> None:
        """Delete one archived experiment."""
        with self._lock:
            self.get(run_id)
            self._path(run_id).unlink()
