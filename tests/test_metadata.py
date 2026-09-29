import json
import sys

import numpy as np
import pytest

from faissight.core.metadata import Metadata, MetadataError, load_metadata
from tests.conftest import SMALL


def test_load_jsonl(synthetic) -> None:
    m = load_metadata(synthetic["chunks"])
    assert len(m) == SMALL["n"]
    assert m.columns == ["text", "source", "title"]
    assert (m.text_column, m.title_column) == ("text", "title")
    row = m.get(3)
    assert row["text"].startswith("Synthetic passage 3")
    assert "id" not in row
    assert 3 in m
    assert m.get(10**9) is None


def test_snippet_truncates_and_collapses_whitespace() -> None:
    m = Metadata([{"id": 1, "text": "a  b\n\nc " + "x" * 300, "title": "T"}])
    s = m.snippet(1, max_chars=20)
    assert s["title"] == "T"
    assert len(s["text"]) == 20
    assert s["text"].startswith("a b c")
    assert s["text"].endswith("…")
    assert m.snippet(2) is None


def test_snippet_without_known_columns() -> None:
    assert Metadata([{"id": 1, "foo": "bar"}]).snippet(1) == {}


def test_load_csv(tmp_path) -> None:
    p = tmp_path / "m.csv"
    p.write_text("id,content,source\n5,hello,a.txt\n6,world,b.txt\n")
    m = load_metadata(p)
    assert m.get(5) == {"content": "hello", "source": "a.txt"}
    assert (m.text_column, m.title_column) == ("content", "source")


def test_load_parquet(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    p = tmp_path / "m.parquet"
    pq.write_table(pa.table({"id": [1, 2], "text": ["a", "b"], "score": [0.5, float("nan")]}), p)
    m = load_metadata(p)
    assert m.get(1) == {"text": "a", "score": 0.5}
    assert m.get(2)["score"] is None  # NaN is not valid JSON


def test_parquet_without_pyarrow(tmp_path, monkeypatch) -> None:
    p = tmp_path / "m.parquet"
    p.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(MetadataError, match="pyarrow") as e:
        load_metadata(p)
    assert "faissight[parquet]" in e.value.hint


def test_from_records_and_dataframe_like() -> None:
    class FakeFrame:
        def to_dict(self, orient):
            assert orient == "records"
            return [{"id": np.int64(1), "text": "hi", "v": np.float32(0.5)}]

    m = load_metadata(FakeFrame())
    assert m.get(1) == {"text": "hi", "v": 0.5}
    assert isinstance(m.get(1)["v"], float)
    assert load_metadata([{"id": 2}]).get(2) == {}
    assert load_metadata(m) is m


def test_json_safe_values() -> None:
    m = Metadata([{"id": 1, "tags": ("a", 1), "nested": {"k": float("inf")}, "obj": object}])
    row = m.get(1)
    assert row["tags"] == ["a", 1]
    assert row["nested"] == {"k": None}
    assert isinstance(row["obj"], str)
    json.dumps(row)


def test_coverage() -> None:
    m = Metadata([{"id": 1}, {"id": 2}])
    assert m.coverage([1, 2, 3, 4]) == 0.5
    assert m.coverage([]) == 1.0


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([{"text": "no id"}], "no 'id'"),
        ([{"id": "abc"}], "non-integer"),
        ([{"id": 1}, {"id": 1}], "Duplicate"),
    ],
)
def test_bad_rows(rows, match) -> None:
    with pytest.raises(MetadataError, match=match) as e:
        Metadata(rows)
    assert e.value.hint


def test_bad_files(tmp_path) -> None:
    with pytest.raises(MetadataError, match="not found"):
        load_metadata(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": 1}\n{oops\n')
    with pytest.raises(MetadataError, match="line 2"):
        load_metadata(bad)
    arr = tmp_path / "arr.jsonl"
    arr.write_text("[1, 2]\n")
    with pytest.raises(MetadataError, match="not a JSON object"):
        load_metadata(arr)
    txt = tmp_path / "m.txt"
    txt.write_text("x")
    with pytest.raises(MetadataError, match="Unsupported"):
        load_metadata(txt)
    with pytest.raises(TypeError):
        load_metadata(42)


def test_blank_lines_ignored(tmp_path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_text('{"id": 1, "text": "a"}\n\n{"id": 2, "text": "b"}\n')
    assert len(load_metadata(p)) == 2
