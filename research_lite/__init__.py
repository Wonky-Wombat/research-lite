#
# __init__.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["Document"]
