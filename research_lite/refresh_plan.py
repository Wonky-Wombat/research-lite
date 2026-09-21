"""Plan safe incremental updates for a ResearchLite document library."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from research_lite.manifest import IngestionManifest
from research_lite.utils.loader_utils import build_source_metadata, discover_files


@dataclass(frozen=True)
class RefreshFailure:
    """A source that could not be inspected while planning a refresh."""

    path: Path
    reason: str


@dataclass
class RefreshPlan:
    """The file-level changes required to synchronize a document library."""

    configuration_changed: bool
    new: list[Path] = field(default_factory=list)
    changed: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    failed: list[RefreshFailure] = field(default_factory=list)

    @property
    def sources_requiring_indexing(self) -> int:
        """Return the count of sources that will need embedding work."""
        return len(self.new) + len(self.changed)


def plan_library_refresh(
    library_root: Path,
    manifest: IngestionManifest,
    *,
    current_config_fingerprint: str,
    extensions: Iterable[str],
) -> RefreshPlan:
    """Compare source files with a manifest without changing either index or manifest."""
    root = library_root.expanduser().resolve()
    recorded_sources = {record.canonical_path: record for record in manifest.list_sources()}
    recorded_fingerprint = manifest.state_value("config_fingerprint")
    configuration_changed = (
        recorded_fingerprint is not None and recorded_fingerprint != current_config_fingerprint
    )
    plan = RefreshPlan(configuration_changed=configuration_changed)
    seen_paths: set[str] = set()

    for source_file in discover_files(str(root), extensions):
        canonical_path = str(source_file.resolve())
        seen_paths.add(canonical_path)
        try:
            source_metadata = build_source_metadata(source_file)
        except OSError as exc:
            plan.failed.append(RefreshFailure(source_file, str(exc)))
            continue

        previous = recorded_sources.get(canonical_path)
        if previous is None:
            plan.new.append(source_file)
        elif configuration_changed or previous.source_id != source_metadata["source_id"]:
            plan.changed.append(source_file)
        elif previous.status != "indexed":
            plan.changed.append(source_file)
        else:
            plan.unchanged.append(source_file)

    plan.deleted.extend(Path(path) for path in sorted(set(recorded_sources).difference(seen_paths)))
    return plan
