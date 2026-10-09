#
# library_backup.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

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
        """Snapshot the current manifest before a refresh starts."""
        backup = cls(state_dir=state_dir, backup_dir=state_dir / BACKUP_DIRNAME)
        backup.discard()
        backup.backup_dir.mkdir()
        try:
            backup._backup_database()
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
        """Replace the manifest with the pre-refresh snapshot."""
        for filename in ("manifest.sqlite", "manifest.sqlite-wal", "manifest.sqlite-shm"):
            (self.state_dir / filename).unlink(missing_ok=True)

        shutil.copy2(self.backup_dir / BACKUP_DATABASE_NAME, self.state_dir / "manifest.sqlite")
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


__all__ = ["BACKUP_DIRNAME", "RefreshBackup"]
