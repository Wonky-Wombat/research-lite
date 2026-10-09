#
# llm_service.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from research_lite import Document
from research_lite.citations import citation_number_by_source, page_label
from research_lite.defaults import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are ResearchLite, a helpful assistant powered by a lightweight RAG system.\n"
    "Use the following pieces of retrieved context to answer the user's question.\n"
    "If the answer is not in the context, say that you don't know. Keep the answer concise.\n\n"
    "Each context item begins with a source citation like [1]. Cite factual claims using only "
    "these numbers, for example [1]. Do not write raw source labels, file paths, chunk IDs, "
    "or metadata.\n\n"
    "Context:\n{context}"
)
REQUEST_TIMEOUT_SECONDS = 120


class OllamaUnavailableError(RuntimeError):
    """Raised when the selected Ollama service or model cannot be used."""


class LLMRequestError(RuntimeError):
    pass


def _ollama_base_url(base_url: str | None) -> str:
    """Return the root URL expected by Ollama's native API."""
    return (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def _validate_ollama(model_name: str, base_url: str) -> None:
    """Check that Ollama is reachable and has the requested model installed."""
    try:
        request = Request(f"{base_url}/api/tags", method="GET")
        with urlopen(request, timeout=3) as response:  # noqa: S310 - user-selected local endpoint.
            payload = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {base_url}. Start it with `ollama serve`, "
            "or pass its root URL with --llm-base-url."
        ) from exc

    installed_models = {
        model.get("name") for model in payload.get("models", []) if isinstance(model, dict)
    }
    installed_aliases = {
        installed_name.removesuffix(":latest")
        for installed_name in installed_models
        if isinstance(installed_name, str)
    }
    if model_name not in installed_models and model_name not in installed_aliases:
        raise OllamaUnavailableError(
            f"Ollama is running, but model {model_name!r} is not installed. "
            f"Run `ollama pull {model_name}` and try again."
        )


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            body: dict[str, Any] = json.loads(response.read())
            return body
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LLMRequestError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise LLMRequestError(f"Could not reach the LLM service at {url}.") from exc
    except json.JSONDecodeError as exc:
        raise LLMRequestError(f"LLM service at {url} returned invalid JSON.") from exc


def _supports_temperature(model_name: str) -> bool:
    model = model_name.lower()
    return not (model.startswith("gpt-5") and "chat" not in model)


class RAGGenerator:
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        provider: str = "openai",
    ) -> None:
        if provider not in {"openai", "ollama"}:
            raise ValueError(f"Unsupported LLM provider: {provider!r}")

        self._provider = provider
        self._temperature = temperature
        self._headers: dict[str, str] = {}
        if provider == "ollama":
            self._model = model_name or DEFAULT_OLLAMA_MODEL
            self._base_url = _ollama_base_url(base_url)
            _validate_ollama(self._model, self._base_url)
        else:
            if api_key is None and base_url is None:
                raise ValueError(
                    "An OpenAI API key is required. Set OPENAI_API_KEY or pass --llm-api-key."
                )
            self._model = model_name or DEFAULT_OPENAI_MODEL
            self._base_url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
            if api_key:
                self._headers["Authorization"] = f"Bearer {api_key}"

    def generate_answer(self, query: str, context_documents: Iterable[Document]) -> str | None:
        """Generate an answer based on the query and retrieved documents."""
        documents = list(context_documents)
        citation_numbers = citation_number_by_source(documents)
        context_items: list[str] = []
        for document in documents:
            source_path = str(document.metadata.get("source_path", ""))
            title = str(document.metadata.get("title", "Unknown"))
            citation_number = citation_numbers.get(source_path or title, 0)
            location = page_label(document.metadata)
            context_items.append(f"[{citation_number}] {title}{location}\n{document.page_content}")
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT.format(context="\n\n".join(context_items)),
            },
            {"role": "user", "content": query},
        ]

        try:
            if self._provider == "ollama":
                response = _post_json(
                    f"{self._base_url}/api/chat",
                    {
                        "model": self._model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": self._temperature},
                    },
                    self._headers,
                )
                content = response["message"]["content"]
            else:
                payload: dict[str, Any] = {"model": self._model, "messages": messages}
                if _supports_temperature(self._model):
                    payload["temperature"] = self._temperature
                response = _post_json(f"{self._base_url}/chat/completions", payload, self._headers)
                content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("LLM service returned an unexpected response.") from exc
        return content if isinstance(content, str) else None
