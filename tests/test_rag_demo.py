"""The RAG demo builder's pure parts and ``faissight demo`` (download/embedding need the
network, so the build itself is stubbed out)."""

import pytest
from typer.testing import CliRunner

import faissight.cli as cli_mod
from faissight import demo as demo_mod
from faissight.cli import app


@pytest.fixture
def rag():
    return demo_mod


def test_chunking_keeps_passage_ids_and_sizes(rag) -> None:
    passages = [" ".join(f"w{i}" for i in range(400)), "short passage here"]
    chunks = rag.chunk(passages, max_chunks=100)
    assert [c["doc_id"] for c in chunks] == [0, 0, 0, 1]
    assert [c["id"] for c in chunks] == [0, 1, 2, 3]
    assert all(len(str(c["text"]).split()) <= rag.CHUNK_WORDS for c in chunks)
    assert chunks[0]["title"].startswith("w0 w1")
    assert "Apache-2.0" in chunks[0]["source"]


def test_chunking_drops_tiny_tails_and_caps(rag) -> None:
    passages = [" ".join(["x"] * (rag.CHUNK_WORDS + 5))]
    assert len(rag.chunk(passages, 100)) == 1  # the 5-word tail is dropped
    many = [" ".join(["y"] * 50)] * 30
    assert len(rag.chunk(many, 7)) == 7


def test_data_license_attribution(rag) -> None:
    assert "Apache License 2.0" in rag.DATA_LICENSE
    assert "ODC-By" in rag.DATA_LICENSE


def test_is_built_and_default_dir(tmp_path, monkeypatch) -> None:
    assert not demo_mod.is_built(tmp_path)
    for f in (*demo_mod.INDEX_FILES, "vectors.npy", "chunks.jsonl", "queries.npy"):
        (tmp_path / f).write_text("x")
    assert demo_mod.is_built(tmp_path)
    monkeypatch.setenv("FAISSIGHT_CACHE_DIR", str(tmp_path / "c"))
    assert demo_mod.default_dir() == tmp_path / "c" / "demo-rag"


@pytest.fixture
def fake_serve(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_mod, "serve", lambda **kw: calls.append(kw))
    return calls


def test_demo_serves_existing_build(tmp_path, monkeypatch, fake_serve) -> None:
    monkeypatch.setattr(demo_mod, "is_built", lambda out: True)
    monkeypatch.setattr(demo_mod, "build_demo", lambda *a, **k: pytest.fail("rebuilt"))
    result = CliRunner().invoke(app, ["demo", "--data-dir", str(tmp_path), "--index", "hnsw"])
    assert result.exit_code == 0, result.output
    (kw,) = fake_serve
    assert kw["index_path"] == tmp_path / "hnsw_flat.index"
    assert kw["vectors"] == tmp_path / "vectors.npy"
    assert kw["meta"] == tmp_path / "chunks.jsonl"
    assert kw["queries"] == tmp_path / "queries.npy"
    assert kw["embedder"] == "all-MiniLM-L6-v2"
    assert kw["normalize_text"] is True


def test_demo_builds_when_missing(tmp_path, monkeypatch, fake_serve) -> None:
    built = []
    monkeypatch.setattr(
        demo_mod, "build_demo", lambda out, max_chunks: built.append((out, max_chunks))
    )
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    result = CliRunner().invoke(app, ["demo", "--data-dir", str(tmp_path), "--max-chunks", "500"])
    assert result.exit_code == 0, result.output
    assert built == [(tmp_path, 500)]
    assert fake_serve[0]["index_path"] == tmp_path / "ivf_pq.index"


def test_demo_missing_extras(tmp_path, monkeypatch, fake_serve) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    result = CliRunner().invoke(app, ["demo", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "faissight[all]" in result.output
    assert fake_serve == []


def test_demo_unknown_index(tmp_path, fake_serve) -> None:
    result = CliRunner().invoke(app, ["demo", "--index", "bogus", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Unknown --index" in result.output
