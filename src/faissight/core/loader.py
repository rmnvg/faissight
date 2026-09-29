"""Read a FAISS index, unwrap wrapper indexes, detect its kind and read its parameters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from faissight.core._faiss import class_name, import_faiss
from faissight.core.types import IndexKind, IndexParams, LoadedIndex, Metric, TransformInfo

GPU_HINT = "Convert it first with `faiss.index_gpu_to_cpu(index)`."


class IndexLoadError(RuntimeError):
    """The file exists but FAISS could not read it as an index."""


def load_index(source: str | os.PathLike[str] | Any) -> LoadedIndex:
    """Load and describe a FAISS index.

    ``source`` is a path to an index file or an in-memory ``faiss.Index``. Unsupported
    indexes (binary, GPU, exotic types) do not raise: they come back with
    ``kind=UNSUPPORTED`` and an ``unsupported_reason`` so callers can still show basic stats.
    """
    faiss = import_faiss()
    path: Path | None = None
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        index = _read(path)
    else:
        index = source

    if isinstance(index, faiss.IndexBinary):
        return _unsupported_basic(index, path, "Binary indexes are not supported.")
    if not isinstance(index, faiss.Index):
        raise TypeError(f"Expected a path or a faiss.Index, got {type(source).__name__}.")

    root = faiss.downcast_index(index)
    if class_name(root).startswith("Gpu"):
        return _unsupported_basic(index, path, f"GPU indexes are not supported. {GPU_HINT}")

    core, class_chain, transforms, ids, has_refine = _unwrap(root)
    metric = _metric(root.metric_type)
    ivf = core if isinstance(core, faiss.IndexIVF) else None
    kind, reason = _detect_kind(core)
    if kind is not IndexKind.UNSUPPORTED and metric is Metric.OTHER:
        kind, reason = IndexKind.UNSUPPORTED, "Only L2 and inner-product metrics are supported."

    return LoadedIndex(
        index=index,
        core=core,
        kind=kind,
        class_chain=class_chain,
        d=int(root.d),
        core_d=int(core.d),
        ntotal=int(root.ntotal),
        metric=metric,
        is_trained=bool(root.is_trained),
        params=_read_params(core, ivf),
        ivf=ivf,
        transforms=transforms,
        has_refine=has_refine,
        ids=ids,
        path=path,
        unsupported_reason=reason,
    )


def _read(path: Path) -> Any:
    faiss = import_faiss()
    if not path.is_file():
        raise FileNotFoundError(f"Index file not found: {path}")
    try:
        return faiss.read_index(str(path))
    except RuntimeError as e:
        # read_index rejects binary index files; they need read_index_binary.
        try:
            return faiss.read_index_binary(str(path))
        except RuntimeError:
            raise IndexLoadError(f"FAISS could not read {path} as an index: {e}") from e


def _unwrap(
    root: Any,
) -> tuple[Any, list[str], list[TransformInfo], np.ndarray | None, bool]:
    """Walk PreTransform / IDMap / Refine wrappers down to the core index."""
    faiss = import_faiss()
    cur = root
    chain: list[str] = []
    transforms: list[TransformInfo] = []
    ids: np.ndarray | None = None
    has_refine = False
    while True:
        chain.append(class_name(cur))
        if isinstance(cur, faiss.IndexPreTransform):
            for i in range(cur.chain.size()):
                vt = faiss.downcast_VectorTransform(cur.chain.at(i))
                transforms.append(TransformInfo(class_name(vt), int(vt.d_in), int(vt.d_out)))
            cur = faiss.downcast_index(cur.index)
        elif isinstance(cur, faiss.IndexIDMap):  # also matches IndexIDMap2
            inner = faiss.vector_to_array(cur.id_map).astype(np.int64)
            # Nested maps compose: user id = outer[inner[offset]].
            ids = inner if ids is None else ids[inner]
            cur = faiss.downcast_index(cur.index)
        elif isinstance(cur, faiss.IndexRefine):
            has_refine = True
            cur = faiss.downcast_index(cur.base_index)
        else:
            return cur, chain, transforms, ids, has_refine


def _detect_kind(core: Any) -> tuple[IndexKind, str | None]:
    faiss = import_faiss()
    name = class_name(core)
    if isinstance(core, faiss.IndexIVF):
        # FastScan variants are not subclasses of IndexIVFPQ, so check them first anyway.
        if isinstance(core, faiss.IndexIVFFastScan):
            return IndexKind.UNSUPPORTED, f"{name}: IVF FastScan internals are not supported yet."
        if isinstance(core, faiss.IndexIVFFlat):
            return IndexKind.IVF_FLAT, None
        if isinstance(core, faiss.IndexIVFPQ):
            return IndexKind.IVF_PQ, None
        if isinstance(core, faiss.IndexIVFScalarQuantizer):
            return IndexKind.IVF_SQ, None
        return IndexKind.UNSUPPORTED, f"{name} is not supported yet."
    if isinstance(core, faiss.IndexHNSW):
        storage = faiss.downcast_index(core.storage)
        if isinstance(storage, faiss.IndexFlat):
            return IndexKind.HNSW_FLAT, None
        return IndexKind.HNSW_OTHER, None
    if isinstance(core, faiss.IndexFlat):
        return IndexKind.FLAT, None
    return IndexKind.UNSUPPORTED, f"{name} is not supported yet."


def _metric(metric_type: int) -> Metric:
    faiss = import_faiss()
    if metric_type == faiss.METRIC_L2:
        return Metric.L2
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return Metric.IP
    return Metric.OTHER


def _sq_type_name(qtype: int) -> str:
    faiss = import_faiss()
    names = {
        getattr(faiss.ScalarQuantizer, a): a[3:]
        for a in dir(faiss.ScalarQuantizer)
        if a.startswith("QT_") and a != "QT_count"
    }
    return names.get(qtype, str(qtype))


def _read_params(core: Any, ivf: Any) -> IndexParams:
    faiss = import_faiss()
    p: dict[str, Any] = {}
    if ivf is not None:
        p["nlist"] = int(ivf.nlist)
        p["nprobe"] = int(ivf.nprobe)
        if hasattr(ivf, "by_residual"):
            p["by_residual"] = bool(ivf.by_residual)
        if isinstance(ivf, faiss.IndexIVFPQ):
            p["pq_m"] = int(ivf.pq.M)
            p["pq_nbits"] = int(ivf.pq.nbits)
        if isinstance(ivf, faiss.IndexIVFScalarQuantizer):
            p["sq_type"] = _sq_type_name(int(ivf.sq.qtype))
    if isinstance(core, faiss.IndexHNSW):
        hnsw = core.hnsw
        # Level 0 holds 2*M links; every upper level holds M.
        p["hnsw_m"] = int(hnsw.nb_neighbors(1))
        p["ef_search"] = int(hnsw.efSearch)
        p["ef_construction"] = int(hnsw.efConstruction)
        p["max_level"] = int(hnsw.max_level)
        p["entry_point"] = int(hnsw.entry_point)
    return IndexParams(**p)


def _unsupported_basic(index: Any, path: Path | None, reason: str) -> LoadedIndex:
    return LoadedIndex(
        index=index,
        core=index,
        kind=IndexKind.UNSUPPORTED,
        class_chain=[class_name(index)],
        d=int(index.d),
        core_d=int(index.d),
        ntotal=int(index.ntotal),
        metric=Metric.OTHER,
        is_trained=bool(index.is_trained),
        params=IndexParams(),
        path=path,
        unsupported_reason=reason,
    )
