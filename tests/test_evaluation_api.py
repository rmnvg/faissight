import json
import threading
import time

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from faissight.core.jobs import JobCancelledError
from faissight.core.relevance import evaluate_relevance, parse_judgements
from faissight.server.app import create_app
from faissight.session import Session


def make_session(tmp_path, **kwargs):
    x = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
    ids = np.array([2**53 + 1, 2**53 + 2, 2**63 - 1], dtype=np.int64)
    index.add_with_ids(x, ids)
    return Session(index, vectors=x, ids=ids, queries=x[:2], cache_root=tmp_path, **kwargs)


def payload():
    return {
        "judgements": "\n".join(
            json.dumps({"row": row, "relevant": {str(i): 1}})
            for row, i in enumerate([2**53 + 1, 2**53 + 2])
        ),
        "k": 1,
        "candidates": 3,
    }


def wait(client, job_id):
    for _ in range(200):
        response = client.get(f"/api/evaluation/{job_id}")
        if response.status_code != 202:
            return response.json()
        time.sleep(0.01)
    raise AssertionError("Evaluation did not finish")


def test_evaluate_rerank_archive_and_restart(tmp_path):
    session = make_session(tmp_path)
    with TestClient(create_app(session)) as client:
        started = client.post("/api/evaluation", json=payload())
        assert started.status_code in (200, 202), started.text
        job_id = started.json()["job_id"]
        body = wait(client, job_id)
        assert body["status"] == "done", body
        assert body["result"]["metrics"] == {"recall": 1, "mrr": 1, "ndcg": 1}
        assert body["result"]["reranked_metrics"] == body["result"]["metrics"]
        assert body["result"]["queries"][0]["ids"] == [str(2**53 + 1)]
        history = client.get("/api/history").json()
        records = history["runs"]
        assert history["current_index"] == session.index_sha1 == records[0]["index"]
        assert len(records) == 1
        assert records[0]["label"] == "Relevance@1, rerank 3 · 2 queries · FLAT"
        run_id = records[0]["id"]
        assert client.patch(f"/api/history/{run_id}", json={"label": "Baseline"}).status_code == 200
    with TestClient(create_app(make_session(tmp_path))) as client:
        assert client.get("/api/history").json()["runs"][0]["label"] == "Baseline"
        assert client.get(f"/api/history/{run_id}").json()["data"]["query_sha256"]
        assert client.delete(f"/api/history/{run_id}").status_code == 200
        assert client.get(f"/api/history/{run_id}").status_code == 404


def test_evaluation_validation_and_demo_history(tmp_path):
    session = make_session(tmp_path, demo_mode=True)
    with TestClient(create_app(session)) as client:
        assert client.get("/api/history").json()["enabled"] is False
        assert (
            client.post("/api/evaluation", json={**payload(), "judgements": "{}"}).status_code
            == 400
        )
        assert client.post("/api/evaluation", json={**payload(), "k": 5}).status_code == 400
        assert client.post("/api/evaluation", json={**payload(), "k": 0}).status_code == 422
        assert client.get("/api/evaluation/missing").status_code == 404
    session = make_session(tmp_path)
    session.queries = None
    with TestClient(create_app(session)) as client:
        assert client.post("/api/evaluation", json=payload()).status_code == 400


def test_evaluation_cancels_between_queries(tmp_path):
    session = make_session(tmp_path)
    labels = parse_judgements(payload()["judgements"], 2)
    calls = []

    def progress(fraction, message):
        calls.append(fraction)
        if fraction > 0:
            raise JobCancelledError("cancelled")

    with pytest.raises(JobCancelledError):
        evaluate_relevance(session.li, session.queries, labels, session.source, progress=progress)
    assert calls == [0, 0.5]


def test_cancel_evaluation_job_does_not_archive(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, progress, **kwargs):
        entered.set()
        assert release.wait(5)
        progress(0.5, "checkpoint")
        raise AssertionError("Cancelled work must not continue")

    monkeypatch.setattr("faissight.session.evaluate_relevance", blocked)
    with TestClient(create_app(session)) as client:
        job_id = client.post("/api/evaluation", json=payload()).json()["job_id"]
        assert entered.wait(5)
        try:
            assert client.delete(f"/api/evaluation/{job_id}").status_code in (200, 202)
        finally:
            release.set()
        assert wait(client, job_id)["status"] == "cancelled"
        assert client.get("/api/history").json()["runs"] == []


def test_disk_failure_keeps_evaluation_result(tmp_path, monkeypatch):
    session = make_session(tmp_path)

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(session.history, "save", fail)
    with TestClient(create_app(session)) as client:
        job_id = client.post("/api/evaluation", json=payload()).json()["job_id"]
        assert wait(client, job_id)["status"] == "done"
        assert client.get("/api/history").json()["warning"]


def test_sweeps_are_archived_and_comparable_after_restart(synthetic, tmp_path):
    def session():
        return Session(
            synthetic["ivf_flat"],
            vectors=synthetic["vectors"],
            queries=synthetic["queries"],
            cache_root=tmp_path,
        )

    first = session()
    for values in ([1, 2], [1, 2, 4]):
        _, job = first.sweep_job(values=values, n_queries=5, repeats=1)
        assert job.wait(10)
        assert job.error is None
    first.jobs.close()
    with TestClient(create_app(session())) as client:
        records = client.get("/api/history").json()["runs"]
        assert len(records) == 2
        assert (
            records[1]["label"] == f"nprobe 1-2 · k=10 · 5 queries · {synthetic['ivf_flat'].name}"
        )
        response = client.post(
            "/api/history/compare", json={"baseline": records[1]["id"], "current": records[0]["id"]}
        )
        assert response.status_code == 200, response.text
        assert response.json()["comparable"] is True
        assert len(response.json()["points"]) == 2
