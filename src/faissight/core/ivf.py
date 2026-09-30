"""IVF inspection: inverted-list sizes, balance, members, centroids and assignments.

For ``IndexPreTransform``-wrapped indexes the centroids live in the *transformed* space
(dimension ``LoadedIndex.core_d``), not the input space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from faissight.core._faiss import import_faiss
from faissight.core.types import LoadedIndex

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]


class NotAnIVFIndexError(ValueError):
    """The loaded index has no IVF core."""


@dataclass(frozen=True)
class ListStats:
    """Summary of how vectors are spread over the inverted lists."""

    sizes: IntArray
    n_empty: int
    imbalance_factor: float
    min: int
    median: float
    max: int
    top_lists: list[tuple[int, int]]
    """``(list_no, size)`` for the largest lists, biggest first."""
    top_5pct_share: float
    """Fraction of all vectors held by the largest 5% of lists (at least one list)."""


@dataclass(frozen=True)
class Assignments:
    """Which inverted list each stored vector lives in, keyed by user-facing id.

    ``list_nos[i]`` is the list holding ``ids[i]``. Sorted by id.
    """

    ids: IntArray
    list_nos: IntArray

    def lookup(self, ids: npt.ArrayLike) -> IntArray:
        """List number for each id, ``-1`` for ids that are not stored."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(self.ids) == 0:
            return np.full(ids.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.ids, ids)
        pos_clipped = np.minimum(pos, len(self.ids) - 1)
        found = (pos < len(self.ids)) & (self.ids[pos_clipped] == ids)
        return np.where(found, self.list_nos[pos_clipped], -1).astype(np.int64)


def _ivf(li: LoadedIndex) -> Any:
    if li.ivf is None:
        raise NotAnIVFIndexError(f"{' → '.join(li.class_chain)} is not an IVF index.")
    return li.ivf


def list_sizes(li: LoadedIndex) -> IntArray:
    """Number of vectors in each inverted list, shape ``(nlist,)``."""
    # Same as faiss.contrib.inspect_tools.get_invlist_sizes, which mypy sees as untyped.
    invlists = _ivf(li).invlists
    return np.array([invlists.list_size(i) for i in range(invlists.nlist)], dtype=np.int64)


def imbalance_factor(sizes: npt.ArrayLike) -> float:
    """FAISS imbalance factor ``nlist * Σ size² / (Σ size)²``.

    1.0 means perfectly balanced; larger is worse. Returns 0.0 for an empty index
    (FAISS itself returns NaN there, which is not JSON-friendly).
    """
    s = np.asarray(sizes, dtype=np.float64)
    total = s.sum()
    if total == 0:
        return 0.0
    return float(len(s) * np.square(s).sum() / total**2)


def list_stats(li: LoadedIndex, top: int = 20) -> ListStats:
    """Sizes plus the summary numbers the overview needs."""
    sizes = list_sizes(li)
    order = np.argsort(-sizes, kind="stable")
    n_top5 = max(1, int(np.ceil(0.05 * len(sizes))))
    total = int(sizes.sum())
    return ListStats(
        sizes=sizes,
        n_empty=int((sizes == 0).sum()),
        imbalance_factor=imbalance_factor(sizes),
        min=int(sizes.min()),
        median=float(np.median(sizes)),
        max=int(sizes.max()),
        top_lists=[(int(i), int(sizes[i])) for i in order[:top]],
        top_5pct_share=float(sizes[order[:n_top5]].sum() / total) if total else 0.0,
    )


def stored_ids(ivf: Any, list_no: int) -> IntArray:
    """Ids stored in one inverted list (without copying its codes).

    These are internal offsets when an IDMap wraps the IVF, and user ids when vectors
    were added to the IVF directly with ``add_with_ids``; ``LoadedIndex.user_ids``
    handles both.
    """
    faiss = import_faiss()
    invlists = faiss.downcast_InvertedLists(ivf.invlists)
    n = int(invlists.list_size(list_no))
    if n == 0:
        return np.empty(0, dtype=np.int64)
    ptr = invlists.get_ids(list_no)
    try:
        return np.array(faiss.rev_swig_ptr(ptr, n), dtype=np.int64, copy=True)
    finally:
        invlists.release_ids(list_no, ptr)


def list_members(li: LoadedIndex, list_no: int) -> IntArray:
    """User-facing ids of the vectors in inverted list ``list_no``."""
    ivf = _ivf(li)
    if not 0 <= list_no < ivf.nlist:
        raise IndexError(f"list_no {list_no} out of range [0, {ivf.nlist}).")
    return li.user_ids(stored_ids(ivf, list_no))


def centroids(li: LoadedIndex) -> FloatArray:
    """Coarse centroids, shape ``(nlist, core_d)``."""
    faiss = import_faiss()
    ivf = _ivf(li)
    quantizer = faiss.downcast_index(ivf.quantizer)
    c: FloatArray = quantizer.reconstruct_n(0, ivf.nlist).astype(np.float32, copy=False)
    return c


def assignments(li: LoadedIndex) -> Assignments:
    """Map every user-facing id to its inverted list. O(ntotal); cache the result."""
    ivf = _ivf(li)
    per_list = [li.user_ids(stored_ids(ivf, i)) for i in range(ivf.nlist)]
    ids = np.concatenate(per_list) if per_list else np.empty(0, dtype=np.int64)
    list_nos = np.repeat(np.arange(ivf.nlist, dtype=np.int64), [len(p) for p in per_list])
    order = np.argsort(ids, kind="stable")
    return Assignments(ids=ids[order], list_nos=list_nos[order])
