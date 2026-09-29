import sys
import types

import numpy as np
import pytest

from faissight.core import embed as E


@pytest.fixture
def fake_st(monkeypatch):
    """A stand-in sentence_transformers module (the real one imports torch: ~15 s)."""
    calls = {}

    class SentenceTransformer:
        def __init__(self, name):
            calls["name"] = name

        def encode(self, texts, normalize_embeddings, convert_to_numpy):
            calls["normalize"] = normalize_embeddings
            v = np.array([[3.0, 4.0]], dtype=np.float64)
            return v / 5.0 if normalize_embeddings else v

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def test_sentence_transformer_embedder(fake_st) -> None:
    embed = E.sentence_transformer_embedder("all-MiniLM-L6-v2", normalize=True)
    v = embed("hello")
    assert fake_st == {"name": "all-MiniLM-L6-v2", "normalize": True}
    assert v.dtype == np.float32
    np.testing.assert_allclose(v, [0.6, 0.8])
    assert "all-MiniLM-L6-v2" in embed.__name__


def test_embedder_without_normalize(fake_st) -> None:
    v = E.sentence_transformer_embedder("m", normalize=False)("x")
    np.testing.assert_allclose(v, [3.0, 4.0])


def test_missing_sentence_transformers(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(E.EmbedderUnavailableError) as e:
        E.sentence_transformer_embedder("m", normalize=True)
    assert "faissight[text]" in e.value.hint


def test_looks_normalized() -> None:
    x = np.random.default_rng(0).standard_normal((5000, 16)).astype(np.float32)
    assert not E.looks_normalized(x)
    assert E.looks_normalized(x / np.linalg.norm(x, axis=1, keepdims=True))
    assert not E.looks_normalized(np.empty((0, 4)))
