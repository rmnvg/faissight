"""Where exact vectors come from: user-supplied raw vectors, or reconstruction from the index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from faissight.core._faiss import import_faiss
from faissight.core.types import LoadedIndex

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]

_BATCH = 65_536


class VectorMismatchError(ValueError):
    """Supplied vectors/ids don't line up with the index."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class VectorSource:
    """Input-space vectors for every stored id, sorted by user-facing id.

    ``reconstructed`` is True when the vectors were decoded from the index rather than
    supplied by the user; distances computed on them include any PQ/SQ error.
    """

    vectors: FloatArray
    ids: IntArray
    reconstructed: bool

    def __len__(self) -> int:
        return len(self.ids)

    def positions(self, ids: npt.ArrayLike) -> IntArray:
        """Row of each id in :attr:`vectors`, ``-1`` if absent."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(self.ids) == 0:
            return np.full(ids.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.ids, ids)
        pos_clipped = np.minimum(pos, len(self.ids) - 1)
        found = self.ids[pos_clipped] == ids
        return np.where(found, pos_clipped, -1).astype(np.int64)

    def get(self, ids: npt.ArrayLike) -> FloatArray:
        """Vectors for the given ids; raises ``KeyError`` for unknown ids."""
        ids = np.asarray(ids, dtype=np.int64)
        pos = self.positions(ids)
        if (pos < 0).any():
            raise KeyError(f"Unknown id(s): {ids[pos < 0][:5].tolist()}")
        return self.vectors[pos]


def from_arrays(
    li: LoadedIndex, vectors: npt.ArrayLike, ids: npt.ArrayLike | None = None
) -> VectorSource:
    """Wrap user-supplied raw vectors (row ``i`` <-> ``ids[i]``, or id ``i`` if no ids)."""
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    if x.ndim != 2:
        raise VectorMismatchError(
            f"Vectors must be a 2-D array, got shape {x.shape}.", "Save an (n, d) float32 array."
        )
    if x.shape[1] != li.d:
        raise VectorMismatchError(
            f"Vectors have dimension {x.shape[1]} but the index expects {li.d}.",
            "Pass the raw vectors that were added to this index (before any PreTransform).",
        )
    if x.shape[0] != li.ntotal:
        raise VectorMismatchError(
            f"Got {x.shape[0]:,} vectors but the index holds {li.ntotal:,}.",
            "Pass exactly the vectors that were added to the index.",
        )
    id_arr = np.arange(len(x), dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    if id_arr.shape != (len(x),):
        raise VectorMismatchError(
            f"Got {id_arr.size:,} ids for {len(x):,} vectors.", "Ids must be a 1-D int64 array."
        )
    order = np.argsort(id_arr, kind="stable")
    sorted_ids = id_arr[order]
    if len(sorted_ids) > 1 and (np.diff(sorted_ids) == 0).any():
        raise VectorMismatchError("Ids contain duplicates.", "Each vector needs a unique id.")
    return VectorSource(x[order], sorted_ids, reconstructed=False)


def reconstruct_all(li: LoadedIndex) -> VectorSource:
    """Decode every stored vector from the index, in input space.

    For IVF indexes this enables a direct map on the index (a lookup table from id to list
    position); it doesn't change search results. For PreTransform indexes the transform is
    inverted, which is lossy for dimensionality reduction like PCA.
    """
    if not li.is_supported:
        raise ValueError(f"Cannot reconstruct: {li.unsupported_reason}")
    if li.ivf is not None:
        stored, core_vecs = _reconstruct_ivf(li.ivf)
    else:
        stored = np.arange(li.ntotal, dtype=np.int64)
        core_vecs = _batched(lambda i0, n: li.core.reconstruct_n(i0, n), li.ntotal, li.core_d)
    user = li.user_ids(stored)
    order = np.argsort(user, kind="stable")
    x = li.from_core_space(core_vecs[order]) if li.vector_transforms else core_vecs[order]
    return VectorSource(np.ascontiguousarray(x, dtype=np.float32), user[order], True)


def _batched(fn: Any, n: int, d: int) -> FloatArray:
    out = np.empty((n, d), dtype=np.float32)
    for i0 in range(0, n, _BATCH):
        m = min(_BATCH, n - i0)
        out[i0 : i0 + m] = fn(i0, m)
    return out


def _reconstruct_ivf(ivf: Any) -> tuple[IntArray, FloatArray]:
    faiss = import_faiss()
    from faissight.core.ivf import stored_ids  # local: ivf imports nothing from here

    stored = np.concatenate(
        [stored_ids(ivf, i) for i in range(ivf.nlist)] or [np.empty(0, dtype=np.int64)]
    )
    if ivf.direct_map.type == faiss.DirectMap.NoMap:
        # An array direct map needs ids 0..n-1; ids added via add_with_ids need a hashtable.
        sequential = np.array_equal(np.sort(stored), np.arange(len(stored)))
        if sequential:
            ivf.make_direct_map()
        else:
            ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
    vecs = _batched(
        lambda i0, n: ivf.reconstruct_batch(stored[i0 : i0 + n]), len(stored), int(ivf.d)
    )
    return stored, vecs
