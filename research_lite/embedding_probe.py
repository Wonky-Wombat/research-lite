#
# embedding_probe.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-17.
#

"""Detection of a changed embedding space for a persisted library."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


class EmbedsQueries(Protocol):
    """The one method this check needs, so any embedding service satisfies it."""

    def embed_query(self, text: str) -> list[float]: ...


STATE_KEY = "embedding_probe"
PROBE_TEXT = "ResearchLite records this sentence to detect a changed embedding space."
# two vectors from one model agree to far closer than this; two models never do
TOLERANCE = 1e-3


def encode(vector: Sequence[float]) -> str:
    """Serialize a probe vector for the manifest's key-value state."""
    return json.dumps([round(float(value), 7) for value in vector], separators=(",", ":"))


def decode(payload: str | None) -> list[float] | None:
    """Read a recorded probe vector back, treating anything unreadable as absent."""
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    try:
        return [float(value) for value in parsed]
    except (TypeError, ValueError):
        return None


def drifted(recorded: Sequence[float] | None, current: Sequence[float]) -> bool:
    """Whether the two probe vectors came from different embedding spaces."""
    if recorded is None:
        return False
    if len(recorded) != len(current):
        return True
    return any(
        not math.isclose(left, right, abs_tol=TOLERANCE)
        for left, right in zip(recorded, current, strict=True)
    )


WARNING = (
    "The embedding model now produces different vectors than when this library was indexed, "
    "so search results will be unreliable. Run `researchlite library sync` to rebuild it."
)


def drift_warning(recorded: str | None, embedding_service: EmbedsQueries) -> str | None:
    """The notice to show before querying, or None when the spaces still agree."""
    if decode(recorded) is None:
        return None
    try:
        current = embedding_service.embed_query(PROBE_TEXT)
    except Exception:  # noqa: BLE001 - the query path reports its own failure
        return None
    return WARNING if drifted(decode(recorded), current) else None
