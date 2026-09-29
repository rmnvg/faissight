"""The RAG demo builder's pure parts (the download and embedding need the network)."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "examples" / "make_rag_demo.py"


@pytest.fixture(scope="module")
def rag():
    spec = importlib.util.spec_from_file_location("make_rag_demo", PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
