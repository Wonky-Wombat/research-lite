#
# model_loading.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Presentation controls for locally loaded transformer models."""

from __future__ import annotations

import logging


def silence_transformers_progress() -> None:
    """Hide dependency status noise while preserving model-loading errors."""
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
