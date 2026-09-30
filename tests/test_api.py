import time

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from faissight import __version__
from faissight.server.app import create_app
from faissight.session import Session
from tests.conftest import SMALL

N, D, NLIST = SMALL["n"], SMALL["d"], SMALL["nlist"]


def _client(session: Session, static_dir=None) -> TestClient:
    return TestClient(create_app(session, static_dir=static_dir), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def ivf_client(synthetic):
    s = Session(
        synthetic["ivf_flat"],
        vectors=synthetic["vectors"],
        metadata=synthetic["chunks"],
        queries=synthetic["queries"],
    )
    return _client(s)


@pytest.fixture(scope="module")
def pq_client(synthetic):
    return _client(Session(synthetic["ivf_pq"]))  # no raw vectors: reconstructed truth


@pytest.fixture(scope="module")
def hnsw_client(synthetic):
    return _client(Session(synthetic["hnsw_flat"], vectors=synthetic["vectors"]))


def _wait_projection(client, **params):
    for _ in range(200):
        r = client.get("/api/projection", params=params)
        if r.status_code != 202:
            return r
        assert r.json()["status"] == "running"
        time.sleep(0.02)
    raise AssertionError("projection never finished")


def _assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["error_code"] == code
    assert body["message"]
    return body


# --- basics ------------------------------------------------------------------------------


def test_health(ivf_client) -> None:
    assert ivf_client.get("/api/health").json() == {"status": "ok"}


def test_info(ivf_client) -> None:
    info = ivf_client.get("/api/info").json()
    assert info["version"] == __version__
    assert info["name"] == "ivf_flat.index"
    assert info["kind"] == "IVF_FLAT"
    assert (info["d"], info["ntotal"], info["metric"]) == (D, N, "L2")
    assert info["higher_is_closer"] is False
    assert info["params"]["nlist"] == NLIST
    assert info["ground_truth_source"] == "raw"
    assert info["supported"] is True
    inputs = info["inputs"]
    assert inputs["raw_vectors"] is True
    assert inputs["metadata_rows"] == N
    assert inputs["metadata_coverage"] == 1.0
    assert inputs["queries"] == SMALL["n_queries"]
    assert inputs["embedder"] is None


def test_info_reconstructed(pq_client) -> None:
    info = pq_client.get("/api/info").json()
    assert info["ground_truth_source"] == "reconstructed"
    assert info["params"]["pq_m"] == 8


def test_metadata(ivf_client) -> None:
    body = ivf_client.get("/api/metadata/7").json()
    assert body["id"] == 7
    assert body["row"]["text"].startswith("Synthetic passage 7")
    _assert_error(ivf_client.get(f"/api/metadata/{N + 1}"), 404, "NOT_FOUND")


def test_metadata_absent(pq_client) -> None:
    _assert_error(pq_client.get("/api/metadata/1"), 404, "NO_METADATA")


# --- IVF ---------------------------------------------------------------------------------


def test_ivf_lists(ivf_client) -> None:
    body = ivf_client.get("/api/ivf/lists").json()
    assert body["nlist"] == NLIST
    assert sum(body["sizes"]) == N
    assert body["imbalance_factor"] >= 1
    assert len(body["top_lists"]) == NLIST  # fewer than 20 lists
    assert body["top_lists"][0]["size"] == body["max"]
    assert len(ivf_client.get("/api/ivf/lists", params={"top": 3}).json()["top_lists"]) == 3


def test_ivf_list_members_paginated(ivf_client) -> None:
    sizes = ivf_client.get("/api/ivf/lists").json()["sizes"]
    full = ivf_client.get("/api/ivf/list/0", params={"limit": 10_000}).json()
    assert full["size"] == sizes[0] == len(full["members"])
    page = ivf_client.get("/api/ivf/list/0", params={"offset": 2, "limit": 3}).json()
    assert [m["id"] for m in page["members"]] == [m["id"] for m in full["members"][2:5]]
    assert page["members"][0]["snippet"]["text"].startswith("Synthetic passage")


def test_ivf_list_errors(ivf_client, hnsw_client) -> None:
    _assert_error(ivf_client.get(f"/api/ivf/list/{NLIST}"), 404, "NOT_FOUND")
    _assert_error(ivf_client.get("/api/ivf/list/0", params={"limit": 0}), 422, "VALIDATION_ERROR")
    _assert_error(hnsw_client.get("/api/ivf/lists"), 400, "NOT_IVF")


# --- projection --------------------------------------------------------------------------


def test_projection_points_and_centroids(ivf_client) -> None:
    pts = _wait_projection(ivf_client).json()
    assert pts["status"] == "done"
    assert pts["kind"] == "points"
    assert len(pts["ids"]) == len(pts["x"]) == len(pts["y"]) == len(pts["list_nos"]) == N
    assert pts["z"] is None
    assert pts["sampled"] is False
    assert len(pts["explained_variance"]) == 2
    cents = _wait_projection(ivf_client, kind="centroids").json()
    assert cents["ids"] == list(range(NLIST))
    assert len(cents["x"]) == NLIST


def test_projection_3d(ivf_client) -> None:
    body = _wait_projection(ivf_client, dims=3).json()
    assert len(body["z"]) == N


def test_failed_projection_stays_failed_until_retried(synthetic, monkeypatch) -> None:
    import faissight.session as session_mod

    real = session_mod.compute_projection
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("out of memory")
        return real(*args, **kwargs)

    monkeypatch.setattr(session_mod, "compute_projection", flaky)
    client = _client(Session(synthetic["ivf_flat"], vectors=synthetic["vectors"], disk_cache=False))
    r = _wait_projection(client)
    _assert_error(r, 500, "PROJECTION_FAILED")
    # Polling again reports the same failure instead of quietly restarting (202 running).
    _assert_error(client.get("/api/projection"), 500, "PROJECTION_FAILED")
    assert len(calls) == 1
    assert _wait_projection(client, retry=True).status_code in (200, 202)
    assert _wait_projection(client).status_code == 200
    assert len(calls) == 2


def test_projection_first_call_is_202(synthetic) -> None:
    client = _client(Session(synthetic["flat_l2"], vectors=synthetic["vectors"]))
    first = client.get("/api/projection")
    assert first.status_code in (200, 202)
    if first.status_code == 202:
        assert set(first.json()) == {"status", "progress", "message", "error"}
    assert _wait_projection(client).status_code == 200


def test_projection_errors(hnsw_client, monkeypatch) -> None:
    _assert_error(hnsw_client.get("/api/projection", params={"kind": "centroids"}), 400, "NOT_IVF")
    _assert_error(hnsw_client.get("/api/projection", params={"dims": 4}), 422, "VALIDATION_ERROR")
    import faissight.session as session_mod

    monkeypatch.setattr(session_mod.importlib.util, "find_spec", lambda name: None)
    body = _assert_error(
        hnsw_client.get("/api/projection", params={"method": "umap"}), 400, "UMAP_UNAVAILABLE"
    )
    assert "faissight[umap]" in body["hint"]


# --- search ------------------------------------------------------------------------------


def test_search_by_id_full_probe(ivf_client) -> None:
    body = ivf_client.post(
        "/api/search", json={"query": {"id": 11}, "k": 10, "nprobe": NLIST}
    ).json()
    assert body["query_kind"] == "id"
    assert body["params"] == {"nprobe": NLIST}
    assert body["recall"] == 1.0
    assert body["truth_source"] == "raw"
    assert len(body["results"]) == 10
    assert 11 not in [r["id"] for r in body["results"]]
    assert all(r["in_truth"] for r in body["results"])
    assert all(t["reason"] == "FOUND" for t in body["truth"])
    assert body["results"][0]["snippet"]["text"]
    assert body["min_nprobe"] >= 1
    assert body["reason_counts"]["FOUND"] == 10


def test_search_low_nprobe_explains_misses(ivf_client) -> None:
    q = np.random.default_rng(5).normal(scale=4.0, size=D).tolist()
    body = ivf_client.post("/api/search", json={"query": {"vector": q}, "nprobe": 1}).json()
    reasons = {t["reason"] for t in body["truth"]}
    assert reasons <= {"FOUND", "CELL_NOT_PROBED"}
    for t in body["truth"]:
        assert (t["found_rank"] is not None) == (t["reason"] == "FOUND")
        assert t["list_no"] is not None
        assert t["probe_rank"] is not None


def test_search_no_compare(ivf_client) -> None:
    body = ivf_client.post(
        "/api/search", json={"query": {"id": 1}, "compare": False, "k": 3}
    ).json()
    assert body["truth"] is None
    assert body["recall"] is None
    assert body["results"][0]["in_truth"] is None


def test_search_with_projection_coords(ivf_client) -> None:
    _wait_projection(ivf_client)
    body = ivf_client.post(
        "/api/search", json={"query": {"id": 3}, "projection": {"method": "pca", "dims": 2}}
    ).json()
    assert len(body["query_coords"]) == 2


def test_search_hnsw_ef(hnsw_client) -> None:
    body = hnsw_client.post("/api/search", json={"query": {"id": 2}, "efSearch": 64}).json()
    assert body["params"] == {"efSearch": 64}
    assert body["min_nprobe"] is None
    assert body["recall"] is not None
    assert all(t["reason"] in ("FOUND", None) for t in body["truth"])


def test_search_reconstructed_pq(pq_client) -> None:
    body = pq_client.post("/api/search", json={"query": {"id": 0}, "nprobe": NLIST}).json()
    assert body["truth_source"] == "reconstructed"


def test_search_text_without_embedder(ivf_client) -> None:
    _assert_error(
        ivf_client.post("/api/search", json={"query": {"text": "hi"}}), 400, "QUERY_ERROR"
    )


def test_search_text_with_embedder(synthetic) -> None:
    client = _client(Session(synthetic["flat_l2"], embedder=lambda t: np.ones(D)))
    body = client.post("/api/search", json={"query": {"text": "hello"}, "k": 4}).json()
    assert body["query_kind"] == "text"
    assert len(body["results"]) == 4


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"query": {}}, 422, "VALIDATION_ERROR"),
        ({"query": {"id": 1, "text": "x"}}, 422, "VALIDATION_ERROR"),
        ({"query": {"id": 1}, "k": 0}, 422, "VALIDATION_ERROR"),
        ({"query": {"vector": [1.0, 2.0]}}, 400, "QUERY_ERROR"),
        ({"query": {"id": 10**9}}, 400, "QUERY_ERROR"),
        ({"query": {"id": 1}, "nprobe": NLIST + 1}, 400, "BAD_REQUEST"),
        ({"query": {"id": 1}, "efSearch": 32}, 400, "BAD_REQUEST"),
    ],
)
def test_search_errors(ivf_client, payload, status, code) -> None:
    _assert_error(ivf_client.post("/api/search", json=payload), status, code)


# --- IVF trace ---------------------------------------------------------------------------


def test_trace_ivf(ivf_client) -> None:
    q = np.random.default_rng(7).normal(scale=4.0, size=D).tolist()
    body = ivf_client.post(
        "/api/trace/ivf", json={"query": {"vector": q}, "nprobe": 2, "compare": False}
    ).json()
    assert body["nprobe"] == 2
    assert body["nlist"] == NLIST
    probes = body["probes"]
    assert [p["probe_rank"] for p in probes] == list(range(NLIST))
    assert [p["probed"] for p in probes] == [True, True] + [False] * (NLIST - 2)
    assert sum(p["n_true_neighbours"] for p in probes) == len(body["neighbours"]) == 10
    dists = [p["centroid_distance"] for p in probes]
    assert dists == sorted(dists)  # L2: closest first
    assert body["min_nprobe"] == 1 + max(n["probe_rank"] for n in body["neighbours"])


def test_search_with_trace_matches_trace_endpoint(ivf_client) -> None:
    req = {"query": {"id": 5}, "nprobe": 2}
    body = ivf_client.post("/api/search", json={**req, "trace": True}).json()
    trace = ivf_client.post("/api/trace/ivf", json=req).json()
    assert body["ivf_trace"] == trace
    assert body["ivf_trace"]["min_nprobe"] == body["min_nprobe"]


def test_search_trace_is_opt_in(ivf_client, hnsw_client) -> None:
    assert ivf_client.post("/api/search", json={"query": {"id": 5}}).json()["ivf_trace"] is None
    body = hnsw_client.post("/api/search", json={"query": {"id": 5}, "trace": True}).json()
    assert body["ivf_trace"] is None


def test_search_trace_needs_compare(ivf_client) -> None:
    resp = ivf_client.post(
        "/api/search", json={"query": {"id": 5}, "trace": True, "compare": False}
    )
    _assert_error(resp, 422, "VALIDATION_ERROR")


def test_trace_ivf_needs_ivf(hnsw_client) -> None:
    _assert_error(hnsw_client.post("/api/trace/ivf", json={"query": {"id": 1}}), 400, "NOT_IVF")


# --- unsupported index -------------------------------------------------------------------


def test_unsupported_index(binary_index_path) -> None:
    client = _client(Session(binary_index_path))
    info = client.get("/api/info").json()
    assert info["supported"] is False
    assert "Binary" in info["unsupported_reason"]
    assert info["ground_truth_source"] is None
    for r in (
        client.post("/api/search", json={"query": {"id": 1}}),
        client.get("/api/projection"),
        client.get("/api/ivf/lists"),
    ):
        _assert_error(r, 400, "UNSUPPORTED_INDEX")


# --- frontend ----------------------------------------------------------------------------


def test_frontend_spa_fallback(synthetic, tmp_path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html>app</html>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    client = _client(Session(synthetic["flat_l2"]), static_dir=tmp_path)
    assert client.get("/").text == "<html>app</html>"
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/query/explorer").text == "<html>app</html>"  # client-side route
    assert client.get("/../../etc/passwd").text == "<html>app</html>"
    _assert_error(client.get("/api/nope"), 404, "NOT_FOUND")


def test_frontend_not_built(synthetic, tmp_path) -> None:
    client = _client(Session(synthetic["flat_l2"]), static_dir=tmp_path / "missing")
    r = client.get("/")
    assert r.status_code == 200
    assert "npm run build" in r.text


def test_openapi_docs(ivf_client) -> None:
    assert ivf_client.get("/api/openapi.json").json()["info"]["title"] == "faissight"


# --- sweep -------------------------------------------------------------------------------


def _wait_sweep(client, job_id):
    for _ in range(500):
        r = client.get(f"/api/sweep/{job_id}")
        if r.status_code != 202:
            return r
        time.sleep(0.01)
    raise AssertionError("sweep never finished")


def test_info_sweep_defaults(ivf_client, hnsw_client, synthetic) -> None:
    s = ivf_client.get("/api/info").json()["sweep"]
    assert s == {"param": "nprobe", "values": [1, 2, 4, 8, 16], "max_value": NLIST}
    h = hnsw_client.get("/api/info").json()["sweep"]
    assert h["param"] == "efSearch"
    assert h["max_value"] is None
    flat = _client(Session(synthetic["flat_l2"]))
    assert flat.get("/api/info").json()["sweep"] is None


def test_sweep_flow(ivf_client) -> None:
    r = ivf_client.post("/api/sweep", json={"k": 10, "n_queries": 40})
    assert r.status_code in (200, 202)
    job_id = r.json()["job_id"]
    body = _wait_sweep(ivf_client, job_id).json()
    assert body["status"] == "done"
    res = body["result"]
    assert res["param"] == "nprobe"
    # --queries was given to this session, so those are used (first n_queries of them).
    assert (res["query_origin"], res["n_queries"], res["truth_source"]) == ("given", 40, "raw")
    assert [p["value"] for p in res["points"]] == [1, 2, 4, 8, 16]
    recalls = [p["recall"] for p in res["points"]]
    assert recalls == sorted(recalls)
    assert recalls[-1] == 1.0
    assert set(res["pareto_values"]) <= {1, 2, 4, 8, 16}
    first = res["points"][0]
    assert sum(n for _, n in first["recall_distribution"]) == 40
    assert first["recall_ci_low"] <= first["recall"] <= first["recall_ci_high"]
    # Given queries have no stored id to open in the explorer.
    assert first["worst_queries"][0]["id"] is None
    assert first["worst_queries"][0]["recall"] == first["recall_distribution"][0][0]
    # Same request -> same job, answered straight from the finished result.
    again = ivf_client.post("/api/sweep", json={"k": 10, "n_queries": 40})
    assert again.status_code == 200
    assert again.json()["job_id"] == job_id


def test_sweep_sampled_queries_hnsw(hnsw_client) -> None:
    r = hnsw_client.post("/api/sweep", json={"values": [16, 64], "n_queries": 30})
    body = _wait_sweep(hnsw_client, r.json()["job_id"]).json()
    res = body["result"]
    assert (res["param"], res["query_origin"]) == ("efSearch", "sampled")
    assert [p["value"] for p in res["points"]] == [16, 64]
    worst = res["points"][0]["worst_queries"]
    assert all(isinstance(w["id"], int) for w in worst)


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"values": [NLIST + 1]}, 400, "BAD_REQUEST"),
        ({"param": "efSearch"}, 400, "BAD_REQUEST"),
        ({"values": []}, 422, "VALIDATION_ERROR"),
        ({"k": 0}, 422, "VALIDATION_ERROR"),
        ({"n_queries": 0}, 422, "VALIDATION_ERROR"),
    ],
)
def test_sweep_errors(ivf_client, payload, status, code) -> None:
    _assert_error(ivf_client.post("/api/sweep", json=payload), status, code)


def test_sweep_unknown_job(ivf_client) -> None:
    _assert_error(ivf_client.get("/api/sweep/nope"), 404, "NOT_FOUND")


def test_sweep_flat_index(synthetic) -> None:
    client = _client(Session(synthetic["flat_l2"]))
    body = _assert_error(client.post("/api/sweep", json={}), 400, "BAD_REQUEST")
    assert "no search parameter" in body["message"]


def test_sweep_unsupported(binary_index_path) -> None:
    client = _client(Session(binary_index_path))
    _assert_error(client.post("/api/sweep", json={}), 400, "UNSUPPORTED_INDEX")


# --- HNSW --------------------------------------------------------------------------------


def test_hnsw_stats(hnsw_client) -> None:
    body = hnsw_client.get("/api/hnsw/stats").json()
    assert body["m"] == SMALL["hnsw_m"]
    assert [lv["level"] for lv in body["levels"]] == list(range(body["max_level"], -1, -1))
    assert body["levels"][-1]["n_nodes"] == N
    assert body["levels"][-1]["max_links"] == 2 * SMALL["hnsw_m"]
    assert sum(body["levels"][-1]["degree_hist"]) == N


def test_hnsw_graph_top_level_complete(hnsw_client) -> None:
    stats = hnsw_client.get("/api/hnsw/stats").json()
    top = stats["max_level"]
    body = hnsw_client.get("/api/hnsw/graph", params={"level": 1}).json()
    assert body["level"] == 1
    assert not body["sampled"]
    assert len(body["ids"]) == body["n_level_nodes"] == len(body["x"]) == len(body["top_levels"])
    assert all(t >= 1 for t in body["top_levels"])
    assert len(body["edges_src"]) == len(body["edges_dst"]) > 0
    n = len(body["ids"])
    assert all(0 <= i < n for i in body["edges_src"] + body["edges_dst"])
    pairs = list(zip(body["edges_src"], body["edges_dst"], strict=True))
    assert len(pairs) == len({tuple(sorted(p)) for p in pairs})  # undirected, deduplicated
    assert (
        stats["entry_point"]
        in hnsw_client.get("/api/hnsw/graph", params={"level": top}).json()["ids"]
    )


def test_hnsw_graph_level0_sampled_around(hnsw_client) -> None:
    body = hnsw_client.get("/api/hnsw/graph", params={"level": 0, "limit": 150, "around": 7}).json()
    assert body["sampled"]
    assert len(body["ids"]) == 150
    assert body["ids"][0] == 7
    assert body["n_level_nodes"] == N


@pytest.mark.parametrize(
    ("params", "status", "code"),
    [
        ({"level": 99}, 404, "NOT_FOUND"),
        ({"level": 0, "around": 10**9}, 404, "NOT_FOUND"),
        ({"limit": 0}, 422, "VALIDATION_ERROR"),
    ],
)
def test_hnsw_graph_errors(hnsw_client, params, status, code) -> None:
    _assert_error(hnsw_client.get("/api/hnsw/graph", params=params), status, code)


def test_hnsw_graph_around_node_not_on_level(hnsw_client) -> None:
    body = hnsw_client.get("/api/hnsw/graph", params={"level": 0, "limit": 5000}).json()
    ground = next(i for i, t in zip(body["ids"], body["top_levels"], strict=True) if t == 0)
    _assert_error(
        hnsw_client.get("/api/hnsw/graph", params={"level": 1, "around": ground}),
        400,
        "BAD_REQUEST",
    )


def test_hnsw_endpoints_need_hnsw(ivf_client) -> None:
    for r in (
        ivf_client.get("/api/hnsw/stats"),
        ivf_client.get("/api/hnsw/graph"),
        ivf_client.post("/api/trace/hnsw", json={"query": {"id": 1}}),
    ):
        _assert_error(r, 400, "NOT_HNSW")


def test_trace_hnsw(hnsw_client) -> None:
    _wait_projection(hnsw_client)
    body = hnsw_client.post(
        "/api/trace/hnsw", json={"query": {"id": 12}, "k": 10, "efSearch": 32}
    ).json()
    assert (body["ef_search"], body["k"]) == (32, 10)
    assert [lv["level"] for lv in body["levels"]] == list(range(body["max_level"], -1, -1))
    assert body["levels"][0]["entry"] == body["entry_point"]
    assert body["overlap_with_faiss"] >= 0.95
    ids = [r["id"] for r in body["results"]]
    assert len(ids) == 10
    assert 12 not in ids  # the query's own vector is excluded, like in /search
    dists = [r["distance"] for r in body["results"]]
    assert dists == sorted(dists)
    assert body["recall"] is not None
    outcomes = {t["outcome"] for t in body["truth"]}
    assert outcomes <= {"FOUND", "NOT_REACHED", "VISITED_NOT_KEPT"}
    # Every node the UI draws has a position.
    drawn = set(body["nodes"]["ids"])
    for lv in body["levels"]:
        for st in lv["steps"]:
            assert st["expanded"] in drawn
            assert all(v["node"] in drawn for v in st["visits"])
    assert set(ids) <= drawn
    assert len(body["query_xy"]) == 2
    assert len(body["levels"][-1]["steps"]) > 0


def test_trace_hnsw_inner_product_shows_similarity() -> None:
    x = np.random.default_rng(0).standard_normal((800, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    index = faiss.IndexHNSWFlat(16, 8, faiss.METRIC_INNER_PRODUCT)
    index.add(x)
    client = _client(Session(index, vectors=x))
    body = client.post("/api/trace/hnsw", json={"query": {"vector": x[3].tolist()}, "k": 5}).json()
    assert body["higher_is_closer"] is True
    sims = [r["distance"] for r in body["results"]]
    assert sims == sorted(sims, reverse=True)
    assert sims[0] == pytest.approx(1.0, abs=1e-4)  # the vector itself
    assert body["overlap_with_faiss"] == 1.0


def test_trace_hnsw_through_idmap() -> None:
    x = np.random.default_rng(1).standard_normal((800, 16)).astype(np.float32)
    index = faiss.IndexIDMap(faiss.IndexHNSWFlat(16, 8))
    index.add_with_ids(x, np.arange(800, dtype=np.int64) * 4 + 100)
    client = _client(Session(index, vectors=x, ids=np.arange(800, dtype=np.int64) * 4 + 100))
    body = client.post("/api/trace/hnsw", json={"query": {"id": 104}, "k": 5}).json()
    assert all(r["id"] >= 100 and (r["id"] - 100) % 4 == 0 for r in body["results"])
    assert 104 not in [r["id"] for r in body["results"]]
    assert body["overlap_with_faiss"] == 1.0


# --- quantization ------------------------------------------------------------------------


def _wait_pq(client):
    for _ in range(500):
        r = client.get("/api/pq/error")
        if r.status_code != 202:
            return r
        time.sleep(0.02)
    raise AssertionError("pq analysis never finished")


def test_pq_error_ivfpq(synthetic) -> None:
    client = _client(
        Session(synthetic["ivf_pq"], vectors=synthetic["vectors"], metadata=synthetic["chunks"])
    )
    body = _wait_pq(client).json()
    assert body["available"] is True
    assert (body["kind"], body["n"], body["code_size"], body["raw_bytes"]) == (
        "IVF_PQ",
        N,
        8,
        4 * D,
    )
    assert body["mean"] > 1
    assert body["median"] <= body["p95"] <= body["max"]
    assert sum(body["histogram"]["counts"]) == N
    assert len(body["histogram"]["edges"]) == len(body["histogram"]["counts"]) + 1
    assert len(body["per_list"]) == NLIST
    assert sum(r["size"] for r in body["per_list"]) == N
    worst = body["worst"]
    assert len(worst) == 50
    assert worst[0]["error"] == pytest.approx(body["max"], rel=1e-5)
    assert worst[0]["list_no"] is not None
    assert worst[0]["snippet"]["text"]
    d = body["distortion"]
    assert len(d["true"]) == len(d["approx"]) == len(d["near"])
    assert d["near_correlation"] < d["correlation"]


def test_pq_error_ivfflat_zero(ivf_client) -> None:
    body = _wait_pq(ivf_client).json()
    assert body["available"] is True
    assert body["mean"] == pytest.approx(0.0, abs=1e-6)


def test_pq_error_needs_raw_vectors(pq_client) -> None:
    r = pq_client.get("/api/pq/error")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "Provide --vectors to measure quantization error."
    assert body["hint"]


def test_pq_error_hnsw_no_lists(hnsw_client) -> None:
    body = _wait_pq(hnsw_client).json()
    assert body["available"] is True
    assert body["per_list"] is None


def test_pq_error_unsupported(binary_index_path) -> None:
    _assert_error(
        _client(Session(binary_index_path)).get("/api/pq/error"), 400, "UNSUPPORTED_INDEX"
    )


# --- demo mode ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_client(synthetic, tmp_path_factory):
    cache = tmp_path_factory.mktemp("demo-cache")
    s = Session(
        synthetic["ivf_flat"], vectors=synthetic["vectors"], cache_root=cache, demo_mode=True
    )
    return _client(s), cache


def test_demo_info_reports_limits(demo_client, ivf_client) -> None:
    client, _ = demo_client
    limits = client.get("/api/info").json()["demo_limits"]
    assert limits["max_k"] == 100
    assert limits["max_sweep_queries"] == 200
    assert ivf_client.get("/api/info").json()["demo_limits"] is None


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/search", {"query": {"id": 1}, "k": 101}),
        ("/api/sweep", {"n_queries": 201}),
        ("/api/sweep", {"values": list(range(1, 14))}),
        ("/api/sweep", {"k": 51}),
    ],
)
def test_demo_caps(demo_client, path, payload) -> None:
    client, _ = demo_client
    body = _assert_error(client.post(path, json=payload), 403, "DEMO_LIMIT")
    assert "locally" in body["hint"]


def test_demo_allows_normal_requests(demo_client) -> None:
    client, _ = demo_client
    assert client.post("/api/search", json={"query": {"id": 1}, "k": 10}).status_code == 200
    assert client.post("/api/sweep", json={"n_queries": 20}).status_code in (200, 202)


def test_demo_umap_only_when_precomputed(synthetic, demo_client, monkeypatch) -> None:
    client, cache = demo_client
    _assert_error(client.get("/api/projection", params={"method": "umap"}), 403, "DEMO_LIMIT")
    # Precompute (as the Docker build does) with a normal session sharing the cache; fake
    # the UMAP fit so the test stays fast.
    import faissight.core.projection as P

    def fake_umap(x, dims, seed):
        class Model:
            embedding_ = x[:, :dims]

            def transform(self, c):
                return c[:, :dims]

        return Model()

    monkeypatch.setattr(P, "_fit_umap", fake_umap)
    monkeypatch.setattr("faissight.session.importlib.util.find_spec", lambda name: object())
    warm = Session(synthetic["ivf_flat"], vectors=synthetic["vectors"], cache_root=cache)
    assert warm.projection_job("umap", 2).wait(30)
    assert _wait_projection(client, method="umap").status_code == 200


def test_demo_hnsw_ef_cap(synthetic, tmp_path) -> None:
    client = _client(Session(synthetic["hnsw_flat"], cache_root=tmp_path, demo_mode=True))
    _assert_error(
        client.post("/api/search", json={"query": {"id": 1}, "efSearch": 5000}), 403, "DEMO_LIMIT"
    )
    _assert_error(client.post("/api/sweep", json={"values": [16, 4096]}), 403, "DEMO_LIMIT")


@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
def test_int64_ids_remain_exact_across_api(kind) -> None:
    x = np.random.default_rng(4).normal(size=(200, 8)).astype(np.float32)
    ids = np.arange(len(x), dtype=np.int64) + 2**53 + 1
    if kind == "ivf":
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 4)
        index.train(x)
        index.nprobe = 4
    else:
        index = faiss.IndexIDMap2(faiss.IndexHNSWFlat(8, 8))
    index.add_with_ids(x, ids)
    client = _client(
        Session(
            index,
            vectors=x,
            ids=ids,
            disk_cache=False,
            metadata=[{"id": int(i), "text": str(i)} for i in ids],
        )
    )
    request = {"query": {"id": str(ids[0])}, "k": 5}
    response = client.post("/api/search", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["recall"] == 1.0
    assert all(isinstance(r["id"], str) for r in body["results"] + body["truth"])
    assert all(int(r["id"]) in ids for r in body["results"])
    assert client.get(f"/api/metadata/{ids[0]}").json()["id"] == str(ids[0])
    projection = _wait_projection(client).json()
    assert set(projection["ids"]) == set(map(str, ids))
    trace = client.post(f"/api/trace/{kind}", json=request)
    assert trace.status_code == 200, trace.text
    if kind == "hnsw":
        tr = trace.json()
        assert all(isinstance(i, str) for i in tr["nodes"]["ids"])
        assert all(isinstance(s["expanded"], str) for lv in tr["levels"] for s in lv["steps"])
        assert all(int(i) in ids for i in tr["faiss_ids"])
    else:
        members = client.get("/api/ivf/list/0").json()["members"]
        assert all(isinstance(r["id"], str) for r in members)
    sweep = client.post("/api/sweep", json={"values": [1 if kind == "ivf" else 4], "n_queries": 20})
    worst = _wait_sweep(client, sweep.json()["job_id"]).json()["result"]["points"][0]
    assert all(isinstance(w["id"], str) for w in worst["worst_queries"])
    assert all(int(w["id"]) in ids for w in worst["worst_queries"])


def test_sweep_capacity_and_cancellation_api(synthetic) -> None:
    import threading

    from faissight.core.jobs import JobRunner

    session = Session(synthetic["ivf_flat"], vectors=synthetic["vectors"], disk_cache=False)
    session.jobs = JobRunner(max_workers=1, max_pending=1)
    release = threading.Event()
    blocker = session.jobs.get_or_start("blocker", lambda p: release.wait(5))
    client = _client(session)
    try:
        first = client.post("/api/sweep", json={"values": [1], "n_queries": 10}).json()
        assert first["status"] == "running"
        overflow = client.post("/api/sweep", json={"values": [2], "n_queries": 10})
        _assert_error(overflow, 429, "JOB_CAPACITY")
        cancelled = client.delete(f"/api/sweep/{first['job_id']}")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert client.get(f"/api/sweep/{first['job_id']}").json()["status"] == "cancelled"
        _assert_error(client.delete("/api/sweep/missing"), 404, "NOT_FOUND")
    finally:
        release.set()
        assert blocker.wait(5)


# --- comparison --------------------------------------------------------------------------


def _wait_compare(client, job_id):
    for _ in range(400):
        r = client.get(f"/api/compare/{job_id}")
        if r.status_code != 202:
            return r
        time.sleep(0.02)
    raise AssertionError("comparison never finished")


@pytest.fixture(scope="module")
def compare_client(synthetic):
    s = Session(
        synthetic["ivf_flat"],
        vectors=synthetic["vectors"],
        compare=[synthetic["ivf_pq"], synthetic["hnsw_flat"]],
        disk_cache=False,
    )
    return _client(s)


def test_info_lists_compare_candidates(compare_client, ivf_client) -> None:
    cands = compare_client.get("/api/info").json()["compare"]
    assert [(c["index"], c["name"], c["kind"]) for c in cands] == [
        (0, "ivf_pq.index", "IVF_PQ"),
        (1, "hnsw_flat.index", "HNSW_FLAT"),
    ]
    assert cands[0]["search_param"] == "nprobe"
    assert cands[0]["max_value"] == cands[0]["params"]["nlist"]
    assert (cands[1]["search_param"], cands[1]["max_value"]) == ("efSearch", None)
    assert ivf_client.get("/api/info").json()["compare"] == []


def test_compare_flow(compare_client) -> None:
    payload = {"candidate": 0, "n_queries": 30, "left_nprobe": NLIST, "right_nprobe": 1}
    r = compare_client.post("/api/compare", json=payload)
    assert r.status_code in (200, 202), r.text
    body = _wait_compare(compare_client, r.json()["job_id"]).json()
    assert body["status"] == "done", body
    res = body["result"]
    assert (res["left"]["name"], res["right"]["name"]) == ("ivf_flat.index", "ivf_pq.index")
    assert res["left"]["params"] == {"nprobe": NLIST}
    assert res["right"]["params"] == {"nprobe": 1}
    assert res["left"]["recall"] == 1.0
    assert res["right"]["recall"] < 1.0
    assert res["right"]["recall_ci_low"] <= res["right"]["recall"] <= res["right"]["recall_ci_high"]
    assert res["left"]["serialized_bytes"] > res["right"]["serialized_bytes"]
    assert res["n_worsened"] > 0
    assert res["n_improved"] == 0
    assert res["n_changed"] == len(res["changes"])
    deltas = [abs(c["right_recall"] - c["left_recall"]) for c in res["changes"]]
    assert deltas == sorted(deltas, reverse=True)
    first = res["changes"][0]
    assert isinstance(first["id"], int)
    assert first["left_only"]
    # Same request -> same job.
    again = compare_client.post("/api/compare", json=payload)
    assert again.status_code == 200
    assert again.json()["job_id"] == r.json()["job_id"]


def test_compare_hnsw_candidate(compare_client) -> None:
    r = compare_client.post(
        "/api/compare", json={"candidate": 1, "n_queries": 20, "right_ef_search": 64}
    )
    body = _wait_compare(compare_client, r.json()["job_id"]).json()
    assert body["result"]["right"]["params"] == {"efSearch": 64}


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"candidate": 2}, 400, "BAD_REQUEST"),
        ({"right_ef_search": 16}, 400, "BAD_REQUEST"),  # candidate 0 is IVF
        ({"left_nprobe": NLIST + 1}, 400, "BAD_REQUEST"),
        ({"k": 0}, 422, "VALIDATION_ERROR"),
    ],
)
def test_compare_errors(compare_client, payload, status, code) -> None:
    _assert_error(compare_client.post("/api/compare", json=payload), status, code)


def test_compare_without_candidates_and_unknown_job(ivf_client) -> None:
    body = _assert_error(ivf_client.post("/api/compare", json={}), 400, "NO_CANDIDATES")
    assert "--compare" in body["hint"]
    _assert_error(ivf_client.get("/api/compare/nope"), 404, "NOT_FOUND")
    _assert_error(ivf_client.delete("/api/compare/nope"), 404, "NOT_FOUND")


def test_compare_cancel(synthetic) -> None:
    import threading

    from faissight.core.jobs import JobRunner

    session = Session(
        synthetic["ivf_flat"], vectors=synthetic["vectors"], compare=[synthetic["ivf_pq"]]
    )
    session.jobs = JobRunner(max_workers=1, max_pending=2)
    release = threading.Event()
    blocker = session.jobs.get_or_start("blocker", lambda p: release.wait(5))
    client = _client(session)
    try:
        job_id = client.post("/api/compare", json={"n_queries": 10}).json()["job_id"]
        cancelled = client.delete(f"/api/compare/{job_id}")
        assert cancelled.json()["status"] == "cancelled"
    finally:
        release.set()
        blocker.wait(5)
    # Starting it again reruns it.
    rerun = client.post("/api/compare", json={"n_queries": 10}).json()
    assert _wait_compare(client, rerun["job_id"]).json()["status"] == "done"
