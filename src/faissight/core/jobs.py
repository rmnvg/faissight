"""Bounded, deduplicated background work with cooperative cancellation.

Workers are daemon threads and exit when the queue drains. Cancellation is checked at
progress callbacks; a native computation already in progress finishes before stopping.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
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
    CANCELLED = "cancelled"


class JobCancelledError(RuntimeError):
    """The caller cancelled this computation."""


class JobCapacityError(RuntimeError):
    """All workers and waiting slots are occupied."""


@dataclass
class Job(Generic[T]):
    """A computation and its progress; use ``wait`` before reading the final result."""

    key: Hashable
    status: JobStatus = JobStatus.RUNNING
    progress: float = 0.0
    message: str = "Queued"
    result: T | None = None
    error: BaseException | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    @property
    def is_done(self) -> bool:
        return self.status is JobStatus.DONE

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for success, failure or cancellation; return False on timeout."""
        return self._done.wait(timeout)

    def cancel(self) -> bool:
        """Request cooperative cancellation; return False if already finished."""
        with self._lock:
            if self.status is not JobStatus.RUNNING:
                return False
            self._cancel.set()
            self.message = "Cancelling…"
            return True

    def _report(self, progress: float, message: str) -> None:
        with self._lock:
            if self._cancel.is_set():
                raise JobCancelledError("Job cancelled.")
            self.progress = min(max(float(progress), 0.0), 1.0)
            self.message = message

    def as_dict(self) -> dict[str, Any]:
        """Consistent status summary suitable for a JSON response."""
        with self._lock:
            return {
                "status": self.status.value,
                "progress": self.progress,
                "message": self.message,
                "error": str(self.error) if self.error else None,
            }


class JobRunner:
    """Run at most ``max_workers`` jobs, queue a bounded number, and retain recent results."""

    def __init__(
        self, max_workers: int = 2, max_pending: int = 8, max_completed: int = 32, ttl: float = 3600
    ) -> None:
        if max_workers < 1 or max_pending < 0 or max_completed < 1 or ttl <= 0:
            raise ValueError("Invalid job capacity or expiry.")
        self._jobs: OrderedDict[Hashable, Job[Any]] = OrderedDict()
        self._pending: deque[tuple[Job[Any], Callable[[ProgressFn], Any]]] = deque()
        self._lock = threading.Lock()
        self._active = 0
        self._closed = False
        self._max_workers = max_workers
        self._max_pending = max_pending
        self._max_completed = max_completed
        self._ttl = ttl

    def _prune(self) -> None:
        completed = [key for key, job in self._jobs.items() if job.finished_at is not None]
        now = time.monotonic()
        for i, key in enumerate(completed):
            finished = self._jobs[key].finished_at
            if i < len(completed) - self._max_completed or (
                finished is not None and now - finished > self._ttl
            ):
                del self._jobs[key]

    def get(self, key: Hashable) -> Job[Any] | None:
        """Get a retained job, or None if absent/expired."""
        with self._lock:
            self._prune()
            job = self._jobs.get(key)
            if job is not None:
                self._jobs.move_to_end(key)
            return job

    def get_or_start(self, key: Hashable, fn: Callable[[ProgressFn], T]) -> Job[T]:
        """Reuse running/successful work, or queue a new attempt within capacity limits."""
        with self._lock:
            if self._closed:
                raise JobCapacityError("This session has stopped.")
            self._prune()
            job = self._jobs.get(key)
            if job is not None and job.status in (JobStatus.RUNNING, JobStatus.DONE):
                self._jobs.move_to_end(key)
                return job
            if self._active >= self._max_workers and len(self._pending) >= self._max_pending:
                raise JobCapacityError("Background job queue is full. Wait for a job to finish.")
            job = Job(key=key)
            self._jobs[key] = job
            self._jobs.move_to_end(key)
            if self._active < self._max_workers:
                self._active += 1
                threading.Thread(
                    target=self._worker, args=(job, fn), name="faissight-worker", daemon=True
                ).start()
            else:
                self._pending.append((job, fn))
            return job

    def cancel(self, key: Hashable) -> bool:
        """Cancel queued work immediately or ask a running job to stop at its next checkpoint."""
        with self._lock:
            job = self._jobs.get(key)
            if job is None or not job.cancel():
                return False
            for pending in list(self._pending):
                if pending[0] is job:
                    self._pending.remove(pending)
                    self._finish(job, None, JobCancelledError("Job cancelled."))
                    self._prune()
                    break
            return True

    def close(self) -> None:
        """Reject new work and cancel queued/running jobs without blocking native calls."""
        with self._lock:
            self._closed = True
            keys = list(self._jobs)
        for key in keys:
            self.cancel(key)

    def run_sync(self, key: Hashable, fn: Callable[[ProgressFn], T]) -> T:
        """Start/join a job and wait for its result; re-raise failures and cancellation."""
        job = self.get_or_start(key, fn)
        job.wait()
        if job.error is not None:
            raise job.error
        return cast(T, job.result)

    @staticmethod
    def _finish(job: Job[Any], result: Any, error: BaseException | None) -> None:
        with job._lock:
            if job._cancel.is_set():
                error = JobCancelledError("Job cancelled.")
            job.error = error
            if error is None:
                job.result, job.progress, job.status = result, 1.0, JobStatus.DONE
            else:
                job.status = (
                    JobStatus.CANCELLED
                    if isinstance(error, JobCancelledError)
                    else JobStatus.FAILED
                )
            job.finished_at = time.monotonic()
            job._done.set()

    def _worker(self, job: Job[Any], fn: Callable[[ProgressFn], Any]) -> None:
        while True:
            result, error = None, None
            try:
                job._report(0.0, "Starting")
                result = fn(job._report)
            except BaseException as exc:
                error = exc
            self._finish(job, result, error)
            with self._lock:
                self._prune()
                if not self._pending:
                    self._active -= 1
                    return
                job, fn = self._pending.popleft()
