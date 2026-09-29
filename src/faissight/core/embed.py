"""Optional text -> vector embedders for text queries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

TEXT_HINT = 'Text queries need sentence-transformers: `pip install "faissight[text]"`.'

Embedder = Callable[[str], npt.ArrayLike]


class EmbedderUnavailableError(ImportError):
    """sentence-transformers is required but not installed."""

    def __init__(self) -> None:
        super().__init__(f"sentence-transformers is not installed. {TEXT_HINT}")
        self.hint = TEXT_HINT


def sentence_transformer_embedder(model_name: str, *, normalize: bool) -> Embedder:
    """Load a sentence-transformers model and return ``text -> float32 vector``.

    Loading imports torch and may download the model, so call it off the request path.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EmbedderUnavailableError() from e
    model: Any = SentenceTransformer(model_name)

    def embed(text: str) -> npt.NDArray[np.float32]:
        vec = model.encode([text], normalize_embeddings=normalize, convert_to_numpy=True)
        return np.asarray(vec[0], dtype=np.float32)

    embed.__name__ = f"sentence_transformer[{model_name}]"
    return embed


def looks_normalized(
    vectors: npt.ArrayLike, sample: int = 1000, tol: float = 1e-3, seed: int = 0
) -> bool:
    """True if (a seeded sample of) the vectors all have unit L2 norm.

    Used to decide whether text-query embeddings should be normalised to match.
    """
    x = np.asarray(vectors)
    if len(x) == 0:
        return False
    rows = np.random.default_rng(seed).choice(len(x), size=min(sample, len(x)), replace=False)
    norms = np.linalg.norm(x[rows].astype(np.float64), axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= tol))
