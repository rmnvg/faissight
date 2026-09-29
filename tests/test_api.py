import time

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
