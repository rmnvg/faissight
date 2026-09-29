"""Run expensive computations (projections, sweeps) in background threads with progress.

A :class:`JobRunner` deduplicates by key: asking for the same key again returns the
running or finished job instead of starting another. Failed jobs are restarted on the
next request.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")
ProgressFn = Callable[[float, str], None]


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job(Generic[T]):
    """One background computation. Read fields freely; only the worker thread writes them."""

    key: Hashable
    status: JobStatus = JobStatus.RUNNING
    progress: float = 0.0
    message: str = ""
    result: T | None = None
    error: BaseException | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def is_done(self) -> bool:
        return self.status is JobStatus.DONE

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the job finishes (either way); returns False on timeout."""
        return self._done.wait(timeout)

    def _report(self, progress: float, message: str) -> None:
        self.progress = min(max(float(progress), 0.0), 1.0)
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        """Status summary suitable for a JSON response."""
        return {
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": str(self.error) if self.error else None,
        }


class JobRunner:
    """Starts and tracks keyed background jobs in daemon threads."""

    def __init__(self) -> None:
        self._jobs: dict[Hashable, Job[Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Job[Any] | None:
        with self._lock:
            return self._jobs.get(key)

    def get_or_start(self, key: Hashable, fn: Callable[[ProgressFn], T]) -> Job[T]:
        """Return the job for ``key``, starting ``fn(progress)`` if none exists or it failed."""
        with self._lock:
            job = self._jobs.get(key)
            if job is not None and job.status is not JobStatus.FAILED:
                return job
            job = Job(key=key)
            self._jobs[key] = job
        thread = threading.Thread(
            target=self._run, args=(job, fn), name=f"faissight-job-{key}", daemon=True
        )
        thread.start()
        return job

    def run_sync(self, key: Hashable, fn: Callable[[ProgressFn], T]) -> T:
        """Start (or join) the job for ``key`` and wait for its result; re-raise failures."""
        job = self.get_or_start(key, fn)
        job.wait()
        if job.error is not None:
            raise job.error
        return cast(T, job.result)

    @staticmethod
    def _run(job: Job[T], fn: Callable[[ProgressFn], T]) -> None:
        try:
            job.result = fn(job._report)
            job.progress = 1.0
            job.status = JobStatus.DONE
        except BaseException as e:  # surfaced via job.error, never lost in the thread
            job.error = e
            job.status = JobStatus.FAILED
        finally:
            job.finished_at = time.monotonic()
            job._done.set()
