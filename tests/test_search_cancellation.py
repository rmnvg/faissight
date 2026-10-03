"""Interactive searches: bounded, cancelled when the client leaves, isolated from sweeps."""

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from faissight.core.jobs import JobRunner, JobStatus
from faissight.server.app import create_app
from faissight.server.routes.common import ApiError
from faissight.server.routes.search import search
from faissight.server.schemas import SearchRequest
from tests.test_evaluation_api import make_session


def test_disconnected_search_stops_at_next_checkpoint(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    entered, release = threading.Event(), threading.Event()
    passed_checkpoint = []

    def query(**kwargs):
        entered.set()
        assert release.wait(5)
        kwargs["progress"](0.5, "After native call")
        passed_checkpoint.append(True)
        raise AssertionError("Should have cancelled")

    class Disconnected:
        async def is_disconnected(self):
            assert await asyncio.to_thread(entered.wait, 5)
            return True

    monkeypatch.setattr(session, "query", query)
    try:
        with pytest.raises(ApiError) as exc:
            asyncio.run(search(SearchRequest(query={"id": 2**53 + 1}), session, Disconnected()))
        assert exc.value.code == "CANCELLED"
    finally:
        release.set()
    jobs = list(session.search_jobs._jobs.values())
    assert len(jobs) == 1
    assert jobs[0].wait(5)
    assert jobs[0].status == JobStatus.CANCELLED
    assert not passed_checkpoint


def test_queued_search_never_starts_after_disconnect(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    session.search_jobs = JobRunner(max_workers=1, max_pending=4)
    release = threading.Event()
    session.search_jobs.get_or_start("occupied", lambda _: release.wait(5))
    ran = []
    monkeypatch.setattr(session, "query", lambda **kw: ran.append(kw))

    class Disconnected:
        async def is_disconnected(self):
            return True

    try:
        with pytest.raises(ApiError):
            asyncio.run(search(SearchRequest(query={"id": 2**53 + 1}), session, Disconnected()))
    finally:
        release.set()
    session.search_jobs.get("occupied").wait(5)
    assert ran == []


def test_search_pool_full_is_busy(tmp_path):
    session = make_session(tmp_path)
    session.search_jobs = JobRunner(max_workers=1, max_pending=0)
    release = threading.Event()
    session.search_jobs.get_or_start("occupied", lambda _: release.wait(5))
    try:
        with TestClient(create_app(session)) as client:
            response = client.post("/api/search", json={"query": {"id": str(2**53 + 1)}})
            assert response.status_code == 429
            assert response.json()["error_code"] == "BUSY"
    finally:
        release.set()


def test_search_does_not_wait_for_background_jobs(tmp_path):
    session = make_session(tmp_path)
    session.jobs = JobRunner(max_workers=1, max_pending=0)
    release = threading.Event()
    session.jobs.get_or_start("long sweep", lambda _: release.wait(5))
    try:
        with TestClient(create_app(session)) as client:
            response = client.post("/api/search", json={"query": {"id": str(2**53 + 1)}, "k": 2})
            assert response.status_code == 200, response.text
            # A stored-id query excludes itself, so its nearest other vector comes first.
            assert response.json()["results"][0]["id"] == str(2**53 + 2)
    finally:
        release.set()


def test_searches_do_not_evict_finished_background_jobs(tmp_path):
    session = make_session(tmp_path)
    session.jobs = JobRunner(max_completed=2)
    sweep = session.jobs.get_or_start("finished sweep", lambda _: "result")
    assert sweep.wait(5)
    with TestClient(create_app(session)) as client:
        for _ in range(5):
            response = client.post("/api/search", json={"query": {"id": str(2**53 + 1)}})
            assert response.status_code == 200, response.text
    assert session.jobs.get("finished sweep") is sweep
