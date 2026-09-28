"""Presentation controls for locally loaded transformer models."""

from __future__ import annotations


def silence_transformers_progress() -> None:
    """Disable Transformers' low-level progress bars in the terminal interface."""
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
