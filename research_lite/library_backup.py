"""Best-effort crash recovery for a single local document library."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

BACKUP_DIRNAME = ".refresh-backup"
BACKUP_DATABASE_NAME = "manifest.sqlite"
READY_MARKER = "ready"
COMPLETED_MARKER = "completed"


@dataclass(frozen=True)
class RefreshBackup:
    """A pre-refresh snapshot that can restore an interrupted local refresh."""

    state_dir: Path
    backup_dir: Path

    @classmethod
    def recover_interrupted_refresh(cls, state_dir: Path) -> None:
        """Restore an unfinished refresh, or remove a completed backup left behind."""
        backup = cls(state_dir=state_dir, backup_dir=state_dir / BACKUP_DIRNAME)
        if not backup.backup_dir.exists():
            return
        if (backup.backup_dir / COMPLETED_MARKER).is_file():
            backup.discard()
        elif (backup.backup_dir / READY_MARKER).is_file():
            backup.restore()
        else:
            backup.discard()

    @classmethod
    def create(cls, state_dir: Path) -> RefreshBackup:
        """Snapshot the current manifest and FAISS artifacts before a refresh starts."""
        backup = cls(state_dir=state_dir, backup_dir=state_dir / BACKUP_DIRNAME)
        backup.discard()
        backup.backup_dir.mkdir()
        try:
            backup._backup_database()
            for artifact in _index_artifacts(state_dir):
                shutil.copy2(artifact, backup.backup_dir / artifact.name)
            (backup.backup_dir / READY_MARKER).touch()
        except Exception:
            backup.discard()
            raise
        return backup

    def complete(self) -> None:
        """Keep the refreshed library and remove its obsolete recovery snapshot."""
        (self.backup_dir / COMPLETED_MARKER).touch()
        try:
            self.discard()
        except OSError:
            pass

    def restore(self) -> None:
        """Replace manifest and index artifacts with the pre-refresh snapshot."""
        for artifact in _index_artifacts(self.state_dir):
            artifact.unlink()
        for filename in ("manifest.sqlite", "manifest.sqlite-wal", "manifest.sqlite-shm"):
            (self.state_dir / filename).unlink(missing_ok=True)

        shutil.copy2(self.backup_dir / BACKUP_DATABASE_NAME, self.state_dir / "manifest.sqlite")
        for artifact in _index_artifacts(self.backup_dir):
            shutil.copy2(artifact, self.state_dir / artifact.name)
        self.discard()

    def discard(self) -> None:
        """Remove the recovery snapshot without changing the active library."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)

    def _backup_database(self) -> None:
        source = sqlite3.connect(self.state_dir / "manifest.sqlite")
        destination = sqlite3.connect(self.backup_dir / BACKUP_DATABASE_NAME)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()


def _index_artifacts(directory: Path) -> list[Path]:
    """Return persisted FAISS artifacts stored directly in a library state directory."""
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix in {".faiss", ".pkl"}
        ),
        key=lambda path: path.name,
    )


__all__ = ["BACKUP_DIRNAME", "RefreshBackup"]
