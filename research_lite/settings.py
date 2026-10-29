#
# settings.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Where ResearchLite keeps its user-owned configuration."""

from __future__ import annotations

import os
from pathlib import Path


def settings_path() -> Path:
    """Return the user-owned settings location, with a testable override."""
    override = os.environ.get("RESEARCHLITE_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".researchlite" / "config.toml"
