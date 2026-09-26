#
# manifest.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

"""SQLite-backed state for a local ResearchLite document library."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from research_lite.preprocessing import SplitConfig

SCHEMA_VERSION = "2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS library_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    canonical_path TEXT PRIMARY KEY,
    indexed_source_id TEXT,
    indexed_config_fingerprint TEXT,
    indexed_at TEXT,
    last_attempt_source_id TEXT,
    last_attempt_at TEXT NOT NULL,
    last_refresh_error TEXT,
    CHECK (
        (indexed_source_id IS NULL
         AND indexed_config_fingerprint IS NULL
         AND indexed_at IS NULL)
        OR
        (indexed_source_id IS NOT NULL
         AND indexed_config_fingerprint IS NOT NULL
         AND indexed_at IS NOT NULL)
    )
);
"""


@dataclass(frozen=True)
class SourceRecord:
    """The manifest's file-level record for one library source."""

    canonical_path: str
    indexed_source_id: str | None
    indexed_config_fingerprint: str | None
    indexed_at: str | None
    last_attempt_source_id: str | None
    last_attempt_at: str
    last_refresh_error: str | None

    @property
    def has_indexed_content(self) -> bool:
        """Whether this source still has a successfully indexed version."""
        return self.indexed_source_id is not None


def config_fingerprint(*, model_name: str, split_config: SplitConfig) -> str:
    """Return a stable digest of settings that affect chunk vectors."""
    payload = {
        "embedding_model": model_name,
        "chunk_size": split_config.chunk_size,
        "chunk_overlap": split_config.chunk_overlap,
        "keep_separator": split_config.keep_separator,
        "separators": list(split_config.separators),
        "separators_by_ext": {
            extension: list(separators)
            for extension, separators in sorted(split_config.separators_by_ext.items())
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_library_root(path: str | Path, library_dir: str | Path | None = None) -> Path:
    """Choose the directory that owns the hidden ResearchLite state directory."""
    if library_dir is not None:
        return Path(library_dir).expanduser().resolve()
    candidate = Path(path).expanduser()
    return (candidate.parent if candidate.is_file() else candidate).resolve()


class IngestionManifest:
    """Own the local SQLite manifest for a single document library."""

    def __init__(self, state_dir: Path, connection: sqlite3.Connection) -> None:
        self._state_dir = state_dir
        self._connection = connection

    @property
    def state_dir(self) -> Path:
        """Return the hidden directory containing this manifest database."""
        return self._state_dir

    @classmethod
    def initialize(
        cls, library_root: Path, *, current_config_fingerprint: str
    ) -> IngestionManifest:
        """Create or open a manifest without loading sources or FAISS."""
        state_dir = library_root.expanduser().resolve() / ".researchlite"
        state_dir.mkdir(parents=True, exist_ok=True)
        connection = _connect(state_dir / "manifest.sqlite")
        existing_schema_version = _stored_schema_version(connection)
        if existing_schema_version is not None and existing_schema_version != SCHEMA_VERSION:
            connection.close()
            msg = (
                f"Manifest at {state_dir} uses schema version {existing_schema_version}. "
                "Delete .researchlite and run --init-library again."
            )
            raise RuntimeError(msg)
        connection.executescript(SCHEMA)

        connection.execute(
            "INSERT OR IGNORE INTO library_state(key, value) VALUES ('initialized_at', ?)",
            (_utc_now(),),
        )
        for key, value in {
            "schema_version": SCHEMA_VERSION,
            "library_root": str(library_root.expanduser().resolve()),
            "config_fingerprint": current_config_fingerprint,
        }.items():
            connection.execute(
                "INSERT INTO library_state(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        connection.commit()
        return cls(state_dir, connection)

    @classmethod
    def open(cls, library_root: Path) -> IngestionManifest:
        """Open an existing manifest without changing its recorded state."""
        state_dir = library_root.expanduser().resolve() / ".researchlite"
        database_path = state_dir / "manifest.sqlite"
        if not database_path.is_file():
            msg = f"No ResearchLite manifest found at {database_path}. Run --init-library first."
            raise FileNotFoundError(msg)

        connection = _connect(database_path)
        try:
            schema_version = connection.execute(
                "SELECT value FROM library_state WHERE key = 'schema_version'"
            ).fetchone()
            if schema_version is None:
                msg = f"Manifest at {database_path} has no schema version."
                raise RuntimeError(msg)
            if str(schema_version["value"]) != SCHEMA_VERSION:
                msg = (
                    f"Unsupported manifest schema version at {database_path}. "
                    "Delete .researchlite and run --init-library again."
                )
                raise RuntimeError(msg)
        except Exception:
            connection.close()
            raise
        return cls(state_dir, connection)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()

    def get_source(self, path: str | Path) -> SourceRecord | None:
        """Return the existing record for a source path, if any."""
        row = self._connection.execute(
            "SELECT canonical_path, indexed_source_id, indexed_config_fingerprint, "
            "indexed_at, last_attempt_source_id, last_attempt_at, last_refresh_error "
            "FROM sources WHERE canonical_path = ?",
            (_canonical_path(path),),
        ).fetchone()
        return _source_record(row) if row is not None else None

    def list_sources(self) -> list[SourceRecord]:
        """Return all source records in stable path order."""
        rows = self._connection.execute(
            """
            SELECT canonical_path, indexed_source_id, indexed_config_fingerprint,
                   indexed_at, last_attempt_source_id, last_attempt_at, last_refresh_error
            FROM sources
            ORDER BY canonical_path
            """
        ).fetchall()
        return [_source_record(row) for row in rows]

    def record_indexed_source(
        self,
        *,
        path: str | Path,
        source_id: str,
        config_fingerprint: str,
    ) -> None:
        """Record a source whose current version was successfully indexed."""
        now = _utc_now()
        self._connection.execute(
            "INSERT INTO sources("
            "canonical_path, indexed_source_id, indexed_config_fingerprint, indexed_at, "
            "last_attempt_source_id, last_attempt_at, last_refresh_error"
            ") VALUES (?, ?, ?, ?, ?, ?, NULL) "
            "ON CONFLICT(canonical_path) DO UPDATE SET "
            "indexed_source_id = excluded.indexed_source_id, "
            "indexed_config_fingerprint = excluded.indexed_config_fingerprint, "
            "indexed_at = excluded.indexed_at, "
            "last_attempt_source_id = excluded.last_attempt_source_id, "
            "last_attempt_at = excluded.last_attempt_at, "
            "last_refresh_error = NULL",
            (
                _canonical_path(path),
                source_id,
                config_fingerprint,
                now,
                source_id,
                now,
            ),
        )
        self._connection.commit()

    def record_refresh_failure(
        self,
        *,
        path: str | Path,
        source_id: str | None,
        error_message: str,
    ) -> None:
        """Record a failed refresh without replacing a prior indexed version."""
        now = _utc_now()
        self._connection.execute(
            "INSERT INTO sources("
            "canonical_path, indexed_source_id, indexed_config_fingerprint, indexed_at, "
            "last_attempt_source_id, last_attempt_at, last_refresh_error"
            ") VALUES (?, NULL, NULL, NULL, ?, ?, ?) "
            "ON CONFLICT(canonical_path) DO UPDATE SET "
            "last_attempt_source_id = excluded.last_attempt_source_id, "
            "last_attempt_at = excluded.last_attempt_at, "
            "last_refresh_error = excluded.last_refresh_error",
            (_canonical_path(path), source_id, now, error_message),
        )
        self._connection.commit()

    def remove_source(self, path: str | Path) -> None:
        """Remove a source record after its indexed chunks have been removed."""
        self._connection.execute(
            "DELETE FROM sources WHERE canonical_path = ?", (_canonical_path(path),)
        )
        self._connection.commit()

    def set_state_value(self, key: str, value: str) -> None:
        """Store a manifest-wide state value."""
        self._connection.execute(
            "INSERT INTO library_state(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._connection.commit()

    def state_value(self, key: str) -> str | None:
        """Return a manifest-wide state value, if present."""
        row = self._connection.execute(
            "SELECT value FROM library_state WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else None


def _canonical_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _stored_schema_version(connection: sqlite3.Connection) -> str | None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'library_state'"
    ).fetchone()
    if table is None:
        return None
    row = connection.execute(
        "SELECT value FROM library_state WHERE key = 'schema_version'"
    ).fetchone()
    return str(row["value"]) if row is not None else None


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _source_record(row: sqlite3.Row) -> SourceRecord:
    return SourceRecord(
        canonical_path=str(row["canonical_path"]),
        indexed_source_id=(
            str(row["indexed_source_id"]) if row["indexed_source_id"] is not None else None
        ),
        indexed_config_fingerprint=(
            str(row["indexed_config_fingerprint"])
            if row["indexed_config_fingerprint"] is not None
            else None
        ),
        indexed_at=str(row["indexed_at"]) if row["indexed_at"] is not None else None,
        last_attempt_source_id=(
            str(row["last_attempt_source_id"])
            if row["last_attempt_source_id"] is not None
            else None
        ),
        last_attempt_at=str(row["last_attempt_at"]),
        last_refresh_error=(
            str(row["last_refresh_error"]) if row["last_refresh_error"] is not None else None
        ),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
