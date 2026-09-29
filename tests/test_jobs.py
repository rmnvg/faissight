import threading

import numpy as np
import pytest

from faissight.core import ivf
from faissight.core import projection as P
from faissight.core import vectors as V
from faissight.core.jobs import JobCancelledError, JobCapacityError, JobRunner, JobStatus
from faissight.core.loader import load_index


def test_job_runs_in_background_with_progress() -> None:
    release = threading.Event()
    seen = threading.Event()

    def work(progress):
        progress(0.5, "halfway")
        seen.set()
        release.wait(5)
        return 42

    runner = JobRunner()
    job = runner.get_or_start("k", work)
    assert seen.wait(5)
    assert job.status is JobStatus.RUNNING
    assert (job.progress, job.message) == (0.5, "halfway")
    assert job.as_dict() == {
        "status": "running",
        "progress": 0.5,
        "message": "halfway",
        "error": None,
    }
    release.set()
    assert job.wait(5)
    assert job.is_done
    assert job.result == 42
    assert job.progress == 1.0
    assert job.finished_at is not None


def test_same_key_is_deduplicated() -> None:
    calls = []
    release = threading.Event()

    def work(progress):
        calls.append(1)
        release.wait(5)
        return "x"

    runner = JobRunner()
    a = runner.get_or_start("k", work)
    b = runner.get_or_start("k", work)
    assert a is b
    release.set()
    a.wait(5)
    assert runner.get_or_start("k", work) is a  # finished jobs are reused too
    assert calls == [1]
    assert runner.get("k") is a
    assert runner.get("other") is None


def test_failed_job_is_kept_until_an_explicit_retry() -> None:
    attempts = []

    def flaky(progress):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return "ok"

    runner = JobRunner()
    job = runner.get_or_start("k", flaky)
    job.wait(5)
    assert job.status is JobStatus.FAILED
    assert job.as_dict()["error"] == "boom"
    # Polling must report the failure, not quietly start over.
    assert runner.get_or_start("k", flaky) is job
    assert len(attempts) == 1
    retry = runner.get_or_start("k", flaky, retry=True)
    assert retry is not job
    retry.wait(5)
    assert retry.result == "ok"
    assert runner.get_or_start("k", flaky, retry=True) is retry  # finished work isn't redone


def test_cancelled_job_is_kept_until_an_explicit_retry() -> None:
    gate = threading.Event()

    def slow(progress):
        gate.wait(5)
        progress(0.5, "checkpoint")
        return "ok"

    runner = JobRunner()
    job = runner.get_or_start("k", slow)
    runner.cancel("k")
    gate.set()
    job.wait(5)
    assert job.status is JobStatus.CANCELLED
    assert runner.get_or_start("k", slow) is job
    again = runner.get_or_start("k", slow, retry=True)
    assert again.wait(5)
    assert again.result == "ok"


def test_run_sync_retries_a_failure() -> None:
    runner = JobRunner()
    with pytest.raises(ValueError, match="bad"):
        runner.run_sync("k", lambda p: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ValueError, match="bad"):
        runner.run_sync("k", lambda p: 1)  # still the kept failure
    assert runner.run_sync("k", lambda p: 1, retry=True) == 1


def test_run_sync_reraises() -> None:
    runner = JobRunner()
    assert runner.run_sync("a", lambda p: 7) == 7
    with pytest.raises(ValueError, match="bad"):
        runner.run_sync("b", lambda p: (_ for _ in ()).throw(ValueError("bad")))


def test_progress_is_clamped() -> None:
    runner = JobRunner()
    job = runner.get_or_start("k", lambda p: p(1.7, "over") or "done")
    job.wait(5)
    assert job.progress == 1.0


def test_projection_as_background_job(synthetic) -> None:
    li = load_index(synthetic["ivf_flat"])
    src = V.from_arrays(li, np.load(synthetic["vectors"]))
    a = ivf.assignments(li)
    messages = []

    def work(progress):
        def tee(frac, msg):
            messages.append(msg)
            progress(frac, msg)

        return P.compute_projection(li, src, assignments=a, progress=tee)

    job = JobRunner().get_or_start(("projection", "pca", 2), work)
    assert job.wait(10)
    assert job.is_done
    assert job.result.coords.shape[1] == 2
    assert messages[0] == "Sampling points"
    assert messages[-1] == "Done"


def test_run_sync_allows_none_result() -> None:
    assert JobRunner().run_sync("n", lambda p: None) is None


def test_queue_capacity_cancel_and_retry() -> None:
    release = threading.Event()
    started = threading.Event()
    calls = []

    def work(progress):
        started.set()
        release.wait(5)
        progress(0.5, "checkpoint")
        return 1

    runner = JobRunner(max_workers=1, max_pending=1)
    running = runner.get_or_start("running", work)
    assert started.wait(5)
    try:
        queued = runner.get_or_start("queued", lambda p: calls.append(1))
        assert runner.get_or_start("queued", work) is queued
        with pytest.raises(JobCapacityError, match="queue is full"):
            runner.get_or_start("overflow", work)
        assert runner.cancel("queued")
        assert queued.wait(1)
        assert queued.status is JobStatus.CANCELLED
        assert not calls
        retry = runner.get_or_start("queued", lambda p: 7, retry=True)
        assert runner.cancel("running")
    finally:
        release.set()
    assert running.wait(5)
    assert running.status is JobStatus.CANCELLED
    assert isinstance(running.error, JobCancelledError)
    assert running.result is None
    assert retry.wait(5)
    assert retry.result == 7


def test_completed_jobs_are_evicted_and_expire(monkeypatch) -> None:
    runner = JobRunner(max_completed=2, ttl=10)
    assert runner.run_sync("a", lambda p: 1) == 1
    assert runner.run_sync("b", lambda p: 2) == 2
    assert runner.get("a") is not None  # recently used results survive first
    assert runner.run_sync("c", lambda p: 3) == 3
    assert runner.get("b") is None
    assert runner.get("a") is not None
    future = runner.get("c").finished_at + 11
    monkeypatch.setattr("faissight.core.jobs.time.monotonic", lambda: future)
    assert runner.get("a") is None
    assert runner.get("c") is None


def test_close_cancels_work_and_rejects_new_jobs() -> None:
    release = threading.Event()
    runner = JobRunner(max_workers=1)
    running = runner.get_or_start("a", lambda p: release.wait(5))
    queued = runner.get_or_start("b", lambda p: 2)
    try:
        runner.close()
        assert queued.wait(1)
        assert queued.status is JobStatus.CANCELLED
        with pytest.raises(JobCapacityError, match="stopped"):
            runner.get_or_start("c", lambda p: 3)
    finally:
        release.set()
    assert running.wait(5)
    assert running.status is JobStatus.CANCELLED
    assert not runner.cancel("a")
    assert not runner.cancel("missing")


def test_retried_job_is_retained_as_recent() -> None:
    runner = JobRunner(max_completed=2)
    with pytest.raises(ValueError, match="retry me"):
        runner.run_sync("a", lambda p: (_ for _ in ()).throw(ValueError("retry me")))
    runner.run_sync("b", lambda p: 2)
    runner.run_sync("a", lambda p: 3, retry=True)
    runner.run_sync("c", lambda p: 4)
    assert runner.get("a").result == 3
    assert runner.get("b") is None
