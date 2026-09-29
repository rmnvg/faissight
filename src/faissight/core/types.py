"""Types shared across ``faissight.core``."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


class IndexKind(str, Enum):
    """What faissight knows how to inspect."""

    FLAT = "FLAT"
    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"
    IVF_SQ = "IVF_SQ"
    HNSW_FLAT = "HNSW_FLAT"
    HNSW_OTHER = "HNSW_OTHER"
    UNSUPPORTED = "UNSUPPORTED"

    @property
    def is_ivf(self) -> bool:
        return self in (IndexKind.IVF_FLAT, IndexKind.IVF_PQ, IndexKind.IVF_SQ)

    @property
    def is_hnsw(self) -> bool:
        return self in (IndexKind.HNSW_FLAT, IndexKind.HNSW_OTHER)


class Metric(str, Enum):
    """Distance metric. For ``IP``, larger values mean closer."""

    L2 = "L2"
    IP = "IP"
    OTHER = "OTHER"

    @property
    def higher_is_closer(self) -> bool:
        return self is Metric.IP


@dataclass(frozen=True)
class IndexParams:
    """Structural and search parameters read from the index. ``None`` = not applicable."""

    nlist: int | None = None
    nprobe: int | None = None
    by_residual: bool | None = None
    pq_m: int | None = None
    pq_nbits: int | None = None
    sq_type: str | None = None
    hnsw_m: int | None = None
    ef_search: int | None = None
    ef_construction: int | None = None
    max_level: int | None = None
    entry_point: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Only the parameters that apply to this index."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class TransformInfo:
    """One step of an ``IndexPreTransform`` chain."""

    name: str
    d_in: int
    d_out: int


@dataclass
class LoadedIndex:
    """A FAISS index plus everything faissight learned about it while loading.

    ``index`` is the object the user gave us (or that we read from disk). It owns the
    C++ memory; ``core``/``ivf`` are non-owning views into it, so ``index`` must stay
    referenced for as long as those are used.
    """

    index: Any
    core: Any
    kind: IndexKind
    class_chain: list[str]
    d: int
    core_d: int
    ntotal: int
    metric: Metric
    is_trained: bool
    params: IndexParams
    ivf: Any = None
    transforms: list[TransformInfo] = field(default_factory=list)
    has_refine: bool = False
    ids: npt.NDArray[np.int64] | None = None
    path: Path | None = None
    unsupported_reason: str | None = None

    @property
    def has_id_map(self) -> bool:
        return self.ids is not None

    @property
    def is_supported(self) -> bool:
        return self.kind is not IndexKind.UNSUPPORTED

    def user_ids(self, internal: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Map internal offsets (``-1`` = no result) to user-facing ids."""
        internal = np.asarray(internal, dtype=np.int64)
        if self.ids is None:
            return internal
        out = np.full_like(internal, -1)
        valid = internal >= 0
        out[valid] = self.ids[internal[valid]]
        return out
