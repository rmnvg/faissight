import os
import sys
import time

import numpy as np
import pytest

from faissight.core import ivf
from faissight.core import projection as P
from faissight.core import vectors as V
from faissight.core.loader import load_index
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]


@pytest.fixture(scope="module")
def x(synthetic):
    return np.load(synthetic["vectors"])


@pytest.fixture
def ivf_flat(synthetic, x):
    li = load_index(synthetic["ivf_flat"])
    return li, V.from_arrays(li, x), ivf.assignments(li)


# --- PCA ---------------------------------------------------------------------------------


def test_pca_matches_svd(x) -> None:
    model = P.fit_pca(x, 3)
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc.astype(np.float64), full_matrices=False)
    for i in range(3):
        assert abs(float(model.components[i] @ vt[i])) == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(model.explained_variance_ratio, (s**2 / (s**2).sum())[:3], rtol=1e-4)
    np.testing.assert_allclose(model.components @ model.components.T, np.eye(3), atol=1e-5)
    assert (np.diff(model.explained_variance_ratio) <= 0).all()


def test_pca_deterministic_and_centered(x) -> None:
    a, b = P.fit_pca(x, 2), P.fit_pca(x.copy(), 2)
    np.testing.assert_array_equal(a.components, b.components)
    np.testing.assert_allclose(a.transform(x).mean(0), 0, atol=1e-3)


def test_pca_bad_dims(x) -> None:
    with pytest.raises(ValueError, match="dims"):
        P.fit_pca(x, D + 1)


def test_pca_speed() -> None:
    # Acceptance: PCA of 20k x 64 in < 2 s.
    big = np.random.default_rng(0).standard_normal((20_000, 64)).astype(np.float32)
    t0 = time.perf_counter()
    P.fit_pca(big, 2).transform(big)
    assert time.perf_counter() - t0 < 2.0


# --- sampling ----------------------------------------------------------------------------


def test_sample_all_when_small() -> None:
    np.testing.assert_array_equal(P.stratified_sample(5, 10), np.arange(5))


def test_sample_uniform() -> None:
    rows = P.stratified_sample(1000, 100, seed=1)
    assert len(rows) == 100
    assert len(np.unique(rows)) == 100
    assert (np.diff(rows) > 0).all()
    np.testing.assert_array_equal(rows, P.stratified_sample(1000, 100, seed=1))
    assert not np.array_equal(rows, P.stratified_sample(1000, 100, seed=2))


def test_sample_stratified_keeps_small_groups() -> None:
    # One huge group and nine tiny ones: every group must survive sampling.
    groups = np.concatenate([np.zeros(10_000), np.arange(1, 10).repeat(3)]).astype(np.int64)
    rows = P.stratified_sample(len(groups), 100, groups, seed=0)
    assert len(rows) == 100
    assert len(np.unique(rows)) == 100
    assert set(groups[rows]) == set(range(10))


def test_sample_stratified_is_proportional() -> None:
    groups = np.repeat([0, 1, 2], [6000, 3000, 1000])
    counts = np.bincount(groups[P.stratified_sample(len(groups), 1000, groups)])
    np.testing.assert_allclose(counts, [600, 300, 100], atol=2)


def test_sample_more_groups_than_points() -> None:
    groups = np.arange(500)
    rows = P.stratified_sample(500, 50, groups)
    assert len(rows) == 50
    assert len(set(groups[rows])) == 50


def test_sample_bad_max_points() -> None:
    with pytest.raises(ValueError, match="max_points"):
        P.stratified_sample(10, 0)


# --- compute_projection ------------------------------------------------------------------


def test_projection_ivf(ivf_flat) -> None:
    li, src, a = ivf_flat
    events = []
    proj = P.compute_projection(li, src, assignments=a, progress=lambda f, m: events.append(f))
    assert proj.method is P.ProjectionMethod.PCA
    assert proj.coords.shape == (N, 2)
    assert proj.coords.dtype == np.float32
    assert proj.centroid_coords.shape == (NLIST, 2)
    assert not proj.sampled
    np.testing.assert_array_equal(proj.list_nos, a.lookup(proj.ids))
    assert events[0] == 0.0
    assert events[-1] == 1.0


def test_projection_sampled_stratified(ivf_flat) -> None:
    li, src, a = ivf_flat
    proj = P.compute_projection(li, src, assignments=a, max_points=200, dims=3)
    assert proj.sampled
    assert proj.n_total == N
    assert proj.coords.shape == (200, 3)
    assert set(proj.list_nos.tolist()) == set(range(NLIST))


def test_projection_non_ivf(synthetic, x) -> None:
    li = load_index(synthetic["hnsw_flat"])
    proj = P.compute_projection(li, V.from_arrays(li, x), max_points=300)
    assert proj.list_nos is None
    assert proj.centroid_coords is None
    assert proj.coords.shape == (300, 2)


def test_projection_pretransform_uses_core_space(synthetic, x) -> None:
    li = load_index(synthetic["pca_ivf_flat"])
    proj = P.compute_projection(li, V.from_arrays(li, x), assignments=ivf.assignments(li))
    assert proj.pca.mean.shape == (li.core_d,)
    # Centroids and points share one space: each centroid sits amid its own list's points.
    for list_no in range(NLIST):
        members = proj.coords[proj.list_nos == list_no]
        if len(members) >= 5:
            dist = np.linalg.norm(proj.centroid_coords - members.mean(0), axis=1)
            assert dist.argsort()[:3].tolist().count(list_no) == 1


def test_projection_bad_dims(ivf_flat) -> None:
    li, src, _ = ivf_flat
    with pytest.raises(ValueError, match="dims"):
        P.compute_projection(li, src, dims=4)


def test_query_lands_near_its_neighbours(ivf_flat, synthetic) -> None:
    # Acceptance: a query's nearest projected points overlap its true top-10 more than chance.
    li, src, a = ivf_flat
    proj = P.compute_projection(li, src, assignments=a)
    queries = np.load(synthetic["queries"])
    placed = P.place_points(proj, queries, None)
    overlaps = []
    for q, qp in zip(queries, placed, strict=True):
        true10 = np.argsort(np.square(src.vectors - q).sum(1))[:10]
        proj10 = np.argsort(np.square(proj.coords - qp).sum(1))[:10]
        overlaps.append(len(set(src.ids[true10]) & set(proj.ids[proj10])))
    chance = 10 * 10 / N
    assert np.mean(overlaps) > 10 * chance


# --- disk cache --------------------------------------------------------------------------


def test_cache_hit_skips_fitting(ivf_flat, tmp_path, monkeypatch) -> None:
    li, src, a = ivf_flat
    cache = P.ProjectionCache(P.index_fingerprint(li), root=tmp_path)
    first = P.compute_projection(li, src, assignments=a, cache=cache)
    assert len(list(cache.dir.glob("*.npz"))) == 1

    def boom(*args, **kwargs):
        raise AssertionError("should have hit the cache")

    monkeypatch.setattr(P, "fit_pca", boom)
    second = P.compute_projection(li, src, assignments=a, cache=cache)
    np.testing.assert_array_equal(first.coords, second.coords)
    np.testing.assert_array_equal(first.ids, second.ids)
    np.testing.assert_array_equal(first.list_nos, second.list_nos)
    np.testing.assert_array_equal(first.centroid_coords, second.centroid_coords)
    np.testing.assert_array_equal(first.pca.components, second.pca.components)


def test_cache_hit_takes_list_nos_from_the_request(ivf_flat, tmp_path) -> None:
    li, src, a = ivf_flat
    cache = P.ProjectionCache("x", root=tmp_path)
    # Every point fits, so the sample (and cache key) is the same with or without assignments.
    without = P.compute_projection(li, src, cache=cache)
    assert without.list_nos is None
    with_lists = P.compute_projection(li, src, assignments=a, cache=cache)
    assert len(list(cache.dir.glob("*.npz"))) == 1  # served from the cache
    np.testing.assert_array_equal(with_lists.coords, without.coords)
    np.testing.assert_array_equal(with_lists.list_nos, a.lookup(with_lists.ids))
    cached = P.cached_projection(li, src, cache, method="pca", dims=2, assignments=a)
    np.testing.assert_array_equal(cached.list_nos, with_lists.list_nos)
    assert P.compute_projection(li, src, cache=cache).list_nos is None


def test_cache_key_depends_on_inputs(synthetic, x, tmp_path) -> None:
    li = load_index(synthetic["ivf_pq"])
    cache = P.ProjectionCache("abc", root=tmp_path)
    a = ivf.assignments(li)
    P.compute_projection(li, V.from_arrays(li, x), assignments=a, cache=cache)
    P.compute_projection(li, V.reconstruct_all(li), assignments=a, cache=cache)
    P.compute_projection(li, V.from_arrays(li, x), assignments=a, cache=cache, dims=3)
    assert len(list(cache.dir.glob("*.npz"))) == 3


def test_cache_non_ivf_roundtrip(synthetic, x, tmp_path) -> None:
    li = load_index(synthetic["flat_l2"])
    cache = P.ProjectionCache("flat", root=tmp_path)
    first = P.compute_projection(li, V.from_arrays(li, x), cache=cache)
    second = P.compute_projection(li, V.from_arrays(li, x), cache=cache)
    assert second.list_nos is None
    assert second.centroid_coords is None
    np.testing.assert_array_equal(first.coords, second.coords)


def test_corrupt_cache_is_recomputed(ivf_flat, tmp_path) -> None:
    li, src, a = ivf_flat
    cache = P.ProjectionCache("x", root=tmp_path)
    P.compute_projection(li, src, assignments=a, cache=cache)
    (path,) = cache.dir.glob("*.npz")
    path.write_bytes(b"garbage")
    assert P.compute_projection(li, src, assignments=a, cache=cache).coords.shape == (N, 2)


def test_index_fingerprint(synthetic) -> None:
    from_file = load_index(synthetic["ivf_flat"])
    fp = P.index_fingerprint(from_file)
    assert len(fp) == 40
    assert fp == P.index_fingerprint(load_index(synthetic["ivf_flat"]))
    assert fp != P.index_fingerprint(load_index(synthetic["ivf_pq"]))
    # An in-memory index hashes its serialized form, which matches the file's bytes.
    assert P.index_fingerprint(load_index(from_file.index)) == fp


def test_default_cache_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FAISSIGHT_CACHE_DIR", str(tmp_path / "c"))
    assert P.default_cache_root() == tmp_path / "c"
    monkeypatch.delenv("FAISSIGHT_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert P.default_cache_root() == tmp_path / "faissight"


# --- UMAP --------------------------------------------------------------------------------


def test_umap_missing(ivf_flat, monkeypatch) -> None:
    li, src, _ = ivf_flat
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(P.ProjectionUnavailableError, match="faissight\\[umap\\]"):
        P.compute_projection(li, src, method="umap")


@pytest.mark.slow
def test_umap_projection(ivf_flat) -> None:
    pytest.importorskip("umap")
    li, src, a = ivf_flat
    proj = P.compute_projection(li, src, method="umap", assignments=a, max_points=400)
    assert proj.method is P.ProjectionMethod.UMAP
    assert proj.pca is None
    assert proj.coords.shape == (400, 2)
    assert proj.centroid_coords.shape == (NLIST, 2)
    ref = P.core_vectors(li, src, proj.ids)
    # A stored point placed via kNN interpolation lands next to its own projection.
    placed = P.place_points(proj, ref[:5], ref)
    nearest = np.linalg.norm(proj.coords[None] - placed[:, None], axis=2).argmin(1)
    assert (nearest == np.arange(5)).mean() >= 0.8


# --- cache write failures, usage and cleanup ------------------------------------------------


def test_cache_write_failure_keeps_the_projection(ivf_flat, tmp_path, monkeypatch) -> None:
    li, src, assign = ivf_flat
    cache = P.ProjectionCache("x" * 40, root=tmp_path)

    def full_disk(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(P.np, "savez", full_disk)
    proj = P.compute_projection(li, src, assignments=assign, cache=cache)
    assert proj.coords.shape == (N, 2)
    assert "No space left on device" in proj.cache_warning
    assert str(cache.dir) in proj.cache_warning
    assert not list(cache.dir.glob("*"))  # the partial file is cleaned up
    monkeypatch.undo()
    assert P.compute_projection(li, src, assignments=assign, cache=cache).cache_warning is None


def _entry(root, sha, name, size, mtime):
    d = root / sha
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_bytes(b"x" * size)
    os.utime(f, (mtime, mtime))
    return f


def test_cache_entries_and_clear(tmp_path) -> None:
    a, b = "a" * 40, "b" * 40
    old = _entry(tmp_path, a, "proj-old.npz", 100, 1_000)
    mid = _entry(tmp_path, b, "proj-mid.npz", 200, 2_000)
    new = _entry(tmp_path, a, "proj-new.npz", 300, 3_000)
    tmp = _entry(tmp_path, b, "abc.npz.tmp", 50, 500)
    demo = tmp_path / "demo-rag"
    demo.mkdir()
    (demo / "data.npz").write_bytes(b"keep")
    (tmp_path / "notes").mkdir()  # not a sha1 folder

    entries = P.cache_entries(tmp_path)
    assert [e.path for e in entries] == [tmp, old, mid, new]  # least recently used first
    assert {e.index_sha1 for e in entries} == {a, b}

    # Unused for 1000 s (at t=3000): the leftover temp file and the oldest entry.
    now = 3_000 + 0.5 * 86400
    removed = P.clear_cache(tmp_path, older_than_days=(now - 1_500) / 86400, now=now)
    assert [e.path for e in removed] == [tmp, old]
    # Least recently used first until 300 bytes remain.
    removed = P.clear_cache(tmp_path, max_bytes=300)
    assert [e.path for e in removed] == [mid]
    assert not (tmp_path / b).exists()  # emptied folders go too
    assert [e.path for e in P.clear_cache(tmp_path)] == [new]
    assert not P.cache_entries(tmp_path)
    assert (demo / "data.npz").exists()
    assert P.cache_entries(tmp_path / "missing") == []


def test_cache_hit_marks_the_entry_used(ivf_flat, tmp_path) -> None:
    li, src, assign = ivf_flat
    cache = P.ProjectionCache("c" * 40, root=tmp_path)
    P.compute_projection(li, src, assignments=assign, cache=cache)
    (path,) = cache.dir.glob("*.npz")
    os.utime(path, (1_000, 1_000))
    P.compute_projection(li, src, assignments=assign, cache=cache)
    assert path.stat().st_mtime > 1_000
