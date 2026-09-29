import threading

import numpy as np
import pytest

from faissight.core import ivf
from faissight.core import projection as P
from faissight.core import vectors as V
from faissight.core.jobs import JobRunner, JobStatus
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


def test_failed_job_reports_and_restarts() -> None:
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
    retry = runner.get_or_start("k", flaky)
    assert retry is not job
    retry.wait(5)
    assert retry.result == "ok"


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
