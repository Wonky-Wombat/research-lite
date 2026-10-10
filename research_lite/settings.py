#
# settings.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""Small, local-only defaults for the ResearchLite command line."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from research_lite.defaults import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL

DEFAULT_LLM_PROVIDER = "ollama"


@dataclass(frozen=True)
class LLMSettings:
    """Non-secret LLM settings retained between one-click launches."""

    provider: str = DEFAULT_LLM_PROVIDER
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL


def load_env_file(start: Path | None = None) -> None:
    """Load KEY=VALUE pairs from the nearest .env file without overriding the environment."""
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        path = candidate / ".env"
        if path.is_file():
            break
    else:
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        entry = line.strip().removeprefix("export ").lstrip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        key = key.strip()
        if not key or any(character.isspace() for character in key):
            continue
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            quote, value = value[0], value[1:-1]
            if quote == '"':
                value = value.encode("utf-8").decode("unicode_escape")
        os.environ.setdefault(key, value)


def settings_path() -> Path:
    """Return the user-owned settings location, with a testable override."""
    override = os.environ.get("RESEARCHLITE_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".researchlite" / "config.toml"


def _load_payload() -> dict[str, object]:
    """Load the small TOML settings document, validating its top-level shape."""
    path = settings_path()
    if not path.is_file():
        return {}

    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg = f"Could not read ResearchLite settings at '{path}'."
        raise RuntimeError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Invalid ResearchLite settings at '{path}'."
        raise RuntimeError(msg)
    return payload


def load_llm_settings() -> LLMSettings:
    """Load saved local defaults, or return the local-first defaults."""
    path = settings_path()
    payload = _load_payload()

    llm = payload.get("llm", {})
    if not isinstance(llm, dict):
        msg = f"Invalid [llm] section in ResearchLite settings at '{path}'."
        raise RuntimeError(msg)
    provider = llm.get("provider", DEFAULT_LLM_PROVIDER)
    model = llm.get("model", DEFAULT_OLLAMA_MODEL)
    base_url = llm.get("base_url", DEFAULT_OLLAMA_BASE_URL)
    if provider not in {"openai", "ollama"} or not all(
        isinstance(value, str) and value for value in (model, base_url)
    ):
        msg = f"Invalid LLM settings at '{path}'."
        raise RuntimeError(msg)
    return LLMSettings(provider=provider, model=model, base_url=base_url)


def save_llm_settings(settings: LLMSettings) -> Path:
    """Persist non-secret defaults so local startup needs no repeated flags."""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[llm]\n"
        f'provider = "{settings.provider}"\n'
        f'model = "{settings.model}"\n'
        f'base_url = "{settings.base_url}"\n',
        encoding="utf-8",
    )
    return path
