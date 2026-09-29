"""Chunk metadata (text, source, title, ...) keyed by user-facing id, for RAG debugging."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PARQUET_HINT = 'Reading .parquet needs pyarrow: `pip install "faissight[parquet]"`.'
TEXT_COLUMNS = ("text", "content", "chunk", "page_content", "passage", "body")
TITLE_COLUMNS = ("title", "source", "name", "doc_id")


class MetadataError(ValueError):
    """Metadata can't be loaded or doesn't have the required shape."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


class Metadata:
    """Rows keyed by integer id. Every row must have an ``id`` column."""

    def __init__(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._rows: dict[int, dict[str, Any]] = {}
        columns: dict[str, None] = {}
        for i, row in enumerate(rows):
            if "id" not in row:
                raise MetadataError(
                    f"Metadata row {i} has no 'id' column.",
                    "Each row needs an integer 'id' matching the index ids.",
                )
            try:
                rid = int(row["id"])
            except (TypeError, ValueError):
                raise MetadataError(
                    f"Metadata row {i} has a non-integer id: {row['id']!r}.",
                    "Ids must be integers matching the index ids.",
                ) from None
            if rid in self._rows:
                raise MetadataError(f"Duplicate metadata id {rid}.", "Each id must appear once.")
            clean = {k: _json_safe(v) for k, v in row.items() if k != "id"}
            self._rows[rid] = clean
            columns.update(dict.fromkeys(clean))
        self.columns = list(columns)
        self.text_column = next((c for c in TEXT_COLUMNS if c in columns), None)
        self.title_column = next((c for c in TITLE_COLUMNS if c in columns), None)

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, id: object) -> bool:
        return id in self._rows

    def get(self, id: int) -> dict[str, Any] | None:
        """The full row for ``id`` (without the id itself), or ``None``."""
        row = self._rows.get(int(id))
        return dict(row) if row is not None else None

    def snippet(self, id: int, max_chars: int = 200) -> dict[str, str] | None:
        """Short ``{"title", "text"}`` preview for tables and tooltips, or ``None``."""
        row = self._rows.get(int(id))
        if row is None:
            return None
        out: dict[str, str] = {}
        if self.title_column and row.get(self.title_column) is not None:
            out["title"] = str(row[self.title_column])
        if self.text_column and row.get(self.text_column) is not None:
            text = " ".join(str(row[self.text_column]).split())
            out["text"] = text if len(text) <= max_chars else text[: max_chars - 1] + "…"
        return out

    def coverage(self, ids: Iterable[int]) -> float:
        """Fraction of ``ids`` that have a metadata row."""
        ids = list(ids)
        return sum(int(i) in self._rows for i in ids) / len(ids) if ids else 1.0


def _json_safe(v: Any) -> Any:
    """Make values JSON-serialisable (NaN -> None, numpy/arrow scalars -> Python)."""
    if hasattr(v, "item") and callable(v.item):
        try:
            v = v.item()
        except (ValueError, TypeError):
            v = str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return str(v)


def load_metadata(source: Any) -> Metadata:
    """Load metadata from a ``.jsonl``/``.csv``/``.parquet`` path, a list of dicts, or a
    DataFrame-like object with ``to_dict("records")`` (pandas is never imported)."""
    if isinstance(source, Metadata):
        return source
    if isinstance(source, (str, os.PathLike)):
        return _load_path(Path(source))
    if hasattr(source, "to_dict"):
        return Metadata(source.to_dict("records"))
    if isinstance(source, Iterable):
        return Metadata(source)
    raise TypeError(f"Unsupported metadata source: {type(source).__name__}.")


def _load_path(path: Path) -> Metadata:
    if not path.is_file():
        raise MetadataError(f"Metadata file not found: {path}", "Check the --meta path.")
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson", ".json"):
        return Metadata(_read_jsonl(path))
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return Metadata(csv.DictReader(f))
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise MetadataError("pyarrow is not installed.", PARQUET_HINT) from e
        return Metadata(pq.read_table(path).to_pylist())
    raise MetadataError(
        f"Unsupported metadata format: {path.suffix or path.name}",
        "Use .jsonl, .csv or .parquet with an 'id' column.",
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise MetadataError(
                    f"{path.name} line {lineno} is not valid JSON: {e.msg}.",
                    "Metadata must be JSON Lines: one JSON object per line.",
                ) from None
            if not isinstance(row, dict):
                raise MetadataError(
                    f"{path.name} line {lineno} is not a JSON object.",
                    'Each line must be an object like {"id": 1, "text": "..."}.',
                )
            yield row
