#
# sync_job.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-18.
#

"""A library refresh that can be started and polled separately."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from research_lite.manifest import IngestionManifest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from research_lite.library_refresh import RefreshProgress, RefreshResult

STATE_KEY = "sync_job"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


@dataclass(frozen=True)
class JobState:
    """What a caller polling a refresh is told."""

    job_id: str
    phase: str
    pid: int
    started_at: float
    file: str = ""
    index: int = 0
    total: int = 0
    done_bytes: int = 0
    total_bytes: int = 0
    finished_at: float | None = None
    indexed: int = 0
    failed: int = 0
    added_chunks: int = 0
    error: str = ""

    @property
    def running(self) -> bool:
        return self.phase == RUNNING


_live: dict[str, JobState] = {}
_lock = threading.Lock()


def _record(library_root: Path, state: JobState) -> None:
    manifest = IngestionManifest.open(library_root)
    try:
        manifest.set_state_value(STATE_KEY, json.dumps(asdict(state)))
    finally:
        manifest.close()


def _set(state: JobState) -> None:
    with _lock:
        _live[state.job_id] = state


def start(
    library_root: Path,
    refresh: Callable[[Callable[[RefreshProgress], None]], RefreshResult],
) -> JobState:
    """Begin a refresh in the background and return its state straight away."""
    job_id = f"{int(time.time() * 1000):x}-{os.getpid():x}"
    state = JobState(job_id=job_id, phase=RUNNING, pid=os.getpid(), started_at=time.time())
    _set(state)
    _record(library_root, state)

    def observe(progress: RefreshProgress) -> None:
        with _lock:
            current = _live.get(job_id)
            if current is None or not current.running:
                return
            _live[job_id] = replace(
                current,
                file=progress.source.name,
                index=progress.index,
                total=progress.total,
                done_bytes=progress.done_bytes,
                total_bytes=progress.total_bytes,
            )

    def work() -> None:
        try:
            result = refresh(observe)
        except Exception as exc:  # noqa: BLE001 - reported through the job, not raised here
            # the type matters: a bare message from a background thread leaves a caller
            # unable to tell a locked library from a programming error
            finished = replace(
                _live[job_id],
                phase=FAILED,
                finished_at=time.time(),
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            finished = replace(
                _live[job_id],
                phase=DONE,
                finished_at=time.time(),
                indexed=len(result.indexed_sources),
                failed=len(result.failed_sources),
                added_chunks=result.added_chunks,
            )
        # recorded before the in-process state, so that a caller elsewhere never sees this
        # job finished here while the manifest still says it is running
        try:
            _record(library_root, finished)
        except Exception:  # noqa: BLE001 - the in-process state is still worth setting
            pass
        _set(finished)

    threading.Thread(target=work, name=f"researchlite-sync-{job_id}", daemon=True).start()
    return state


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def read(library_root: Path, job_id: str | None = None) -> JobState | None:
    """The state of the running or most recent refresh, or None if there has not been one."""
    with _lock:
        if job_id is not None and job_id in _live:
            return _live[job_id]
        if job_id is None and _live:
            return max(_live.values(), key=lambda state: state.started_at)
    manifest = IngestionManifest.open(library_root)
    try:
        payload = manifest.state_value(STATE_KEY)
    finally:
        manifest.close()
    if not payload:
        return None
    try:
        fields = json.loads(payload)
        state = JobState(**fields)
    except (json.JSONDecodeError, TypeError):
        return None
    if job_id is not None and state.job_id != job_id:
        return None
    if state.running and not _alive(state.pid):
        return replace(
            state,
            phase=FAILED,
            finished_at=state.finished_at or time.time(),
            error="The process running this refresh exited before it finished.",
        )
    return state


__all__ = ["DONE", "FAILED", "RUNNING", "JobState", "read", "start"]
