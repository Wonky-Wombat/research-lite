"""Process-safe locking for operations that mutate a document library."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

REFRESH_LOCK_FILENAME = "refresh.lock"


class LibraryRefreshLockedError(RuntimeError):
    """Raised when another process is already refreshing the same library."""


@contextmanager
def acquire_refresh_lock(state_dir: Path, *, timeout: float = 0.0) -> Iterator[None]:
    """Acquire an exclusive library refresh lock for the duration of an operation."""
    lock = FileLock(str(state_dir / REFRESH_LOCK_FILENAME))
    try:
        with lock.acquire(timeout=timeout):
            yield
    except Timeout as exc:
        msg = f"Library at '{state_dir.parent}' is already being refreshed. Try again shortly."
        raise LibraryRefreshLockedError(msg) from exc


__all__ = ["LibraryRefreshLockedError", "REFRESH_LOCK_FILENAME", "acquire_refresh_lock"]
