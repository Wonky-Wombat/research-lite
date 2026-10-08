#
# model_loading.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Shared settings for locally loaded ONNX models."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def silence_model_downloads() -> None:
    """Hide dependency status noise while preserving model-loading errors."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def model_cache_dir() -> str:
    return str(Path.home() / ".cache" / "researchlite" / "models")


def use_cuda(device: str) -> bool:
    if device not in {"cpu", "cuda"}:
        msg = f"Unsupported device {device!r}; use 'cpu' or 'cuda'."
        raise ValueError(msg)
    return device == "cuda"
