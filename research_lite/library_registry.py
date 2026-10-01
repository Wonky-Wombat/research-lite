#
# library_registry.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Persistent local library selection."""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from research_lite.settings import settings_path

_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


@dataclass(frozen=True)
class LibraryRecord:
    name: str
    path: Path
    last_synced_at: datetime | None = None


@dataclass(frozen=True)
class LibraryRegistry:
    active_name: str | None = None
    libraries: tuple[LibraryRecord, ...] = ()

    @property
    def active(self) -> LibraryRecord | None:
        return next((item for item in self.libraries if item.name == self.active_name), None)


def registry_path() -> Path:
    override = os.environ.get("RESEARCHLITE_LIBRARY_REGISTRY_PATH")
    if override:
        return Path(override).expanduser()
    return settings_path().parent / "libraries.toml"


def load_library_registry() -> LibraryRegistry:
    path = registry_path()
    if path.is_file():
        return _parse_registry(path, path.read_text(encoding="utf-8"))
    registry = _legacy_registry()
    if registry.libraries:
        save_library_registry(registry)
    return registry


def save_library_registry(registry: LibraryRegistry) -> Path:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_registry(registry)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def register_library(
    path: Path,
    *,
    name: str | None = None,
    activate: bool = True,
    synced: bool = False,
) -> LibraryRegistry:
    registry = load_library_registry()
    normalized = path.expanduser().resolve()
    existing = next((item for item in registry.libraries if item.path == normalized), None)
    library_name = (
        _validate_name(name)
        if name is not None
        else (existing.name if existing is not None else _next_name(normalized, registry))
    )
    named = next((item for item in registry.libraries if item.name == library_name), None)
    if named is not None and named.path != normalized:
        raise ValueError(f"Library name '{library_name}' is already in use.")
    timestamp = datetime.now(UTC) if synced else (existing.last_synced_at if existing else None)
    record = LibraryRecord(library_name, normalized, timestamp)
    libraries = tuple(
        item for item in registry.libraries if item.name != library_name and item.path != normalized
    ) + (record,)
    updated = LibraryRegistry(library_name if activate else registry.active_name, libraries)
    save_library_registry(updated)
    return updated


def select_library(name: str) -> LibraryRegistry:
    registry = load_library_registry()
    library_name = _validate_name(name)
    if not any(item.name == library_name for item in registry.libraries):
        raise KeyError(f"No ResearchLite library named '{library_name}'.")
    updated = replace(registry, active_name=library_name)
    save_library_registry(updated)
    return updated


def _legacy_registry() -> LibraryRegistry:
    path = settings_path()
    if not path.is_file():
        return LibraryRegistry()
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Could not read ResearchLite settings at '{path}'.") from exc
    library = payload.get("library")
    if not isinstance(library, dict):
        return LibraryRegistry()
    paths = library.get("paths", [])
    active = library.get("active")
    if not isinstance(paths, list) or not all(isinstance(item, str) and item for item in paths):
        raise RuntimeError(f"Invalid legacy library settings at '{path}'.")
    if active is not None and (not isinstance(active, str) or not active):
        raise RuntimeError(f"Invalid legacy library settings at '{path}'.")
    candidates = [Path(item).expanduser().resolve() for item in paths]
    if active is not None:
        candidates.append(Path(active).expanduser().resolve())
    records: list[LibraryRecord] = []
    for candidate in candidates:
        if any(item.path == candidate for item in records):
            continue
        name = _next_name(candidate, LibraryRegistry(libraries=tuple(records)))
        records.append(LibraryRecord(name, candidate))
    active_path = Path(active).expanduser().resolve() if active is not None else None
    active_name = next((item.name for item in records if item.path == active_path), None)
    return LibraryRegistry(active_name, tuple(records))


def _parse_registry(path: Path, raw: str) -> LibraryRegistry:
    try:
        payload = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"Could not read library registry at '{path}'.") from exc
    if payload.get("version") != 1:
        raise RuntimeError(f"Unsupported library registry at '{path}'.")
    active = payload.get("active")
    entries = payload.get("libraries", {})
    if active is not None and not isinstance(active, str):
        raise RuntimeError(f"Invalid library registry at '{path}'.")
    if not isinstance(entries, dict):
        raise RuntimeError(f"Invalid library registry at '{path}'.")
    records: list[LibraryRecord] = []
    for name, value in entries.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise RuntimeError(f"Invalid library registry at '{path}'.")
        source_path = value.get("path")
        timestamp = value.get("last_synced_at")
        if not isinstance(source_path, str) or not source_path:
            raise RuntimeError(f"Invalid library registry at '{path}'.")
        if timestamp is not None and not isinstance(timestamp, str):
            raise RuntimeError(f"Invalid library registry at '{path}'.")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp) if timestamp else None
        except ValueError as exc:
            raise RuntimeError(f"Invalid library registry at '{path}'.") from exc
        records.append(LibraryRecord(_validate_name(name), Path(source_path), parsed_timestamp))
    if len({item.path for item in records}) != len(records):
        raise RuntimeError(f"Duplicate library paths in registry at '{path}'.")
    if active is not None and not any(item.name == active for item in records):
        raise RuntimeError(f"Invalid active library in registry at '{path}'.")
    return LibraryRegistry(active, tuple(records))


def _serialize_registry(registry: LibraryRegistry) -> str:
    active = (
        "" if registry.active_name is None else f"active = {json.dumps(registry.active_name)}\n"
    )
    entries = []
    for item in registry.libraries:
        timestamp = (
            ""
            if item.last_synced_at is None
            else (f"last_synced_at = {json.dumps(item.last_synced_at.isoformat())}\n")
        )
        entries.append(f"[libraries.{item.name}]\npath = {json.dumps(str(item.path))}\n{timestamp}")
    return f"version = 1\n{active}\n" + "\n".join(entries)


def _next_name(path: Path, registry: LibraryRegistry) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", path.name.lower()).strip("-") or "library"
    names = {item.name for item in registry.libraries}
    candidate = stem[:64].rstrip("-")
    suffix = 2
    while candidate in names:
        suffix_text = f"-{suffix}"
        candidate = f"{stem[: 64 - len(suffix_text)].rstrip('-')}{suffix_text}"
        suffix += 1
    return _validate_name(candidate)


def _validate_name(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError("Library names must use lowercase letters, numbers, and hyphens.")
    return name
