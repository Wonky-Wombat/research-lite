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

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_core.pydantic_v1 import SecretStr
except ImportError:
    from pydantic import SecretStr

from langchain_openai import ChatOpenAI

from research_lite.citations import citation_number_by_source, page_label
from research_lite.defaults import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
)

ChatOllama: Any = None
try:
    from langchain_ollama import ChatOllama as _chat_ollama

    ChatOllama = _chat_ollama
except ImportError:  # pragma: no cover - exercised only in incomplete installations.
    pass


DEFAULT_SYSTEM_PROMPT = (
    "You are ResearchLite, a helpful assistant powered by a lightweight RAG system.\n"
    "Use the following pieces of retrieved context to answer the user's question.\n"
    "If the answer is not in the context, say that you don't know. Keep the answer concise.\n\n"
    "Each context item begins with a source citation like [1]. Cite factual claims using only "
    "these numbers, for example [1]. Do not write raw source labels, file paths, chunk IDs, "
    "or metadata.\n\n"
    "Context:\n{context}"
)


class OllamaUnavailableError(RuntimeError):
    """Raised when the selected Ollama service or model cannot be used."""


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

        final_api_key: SecretStr | None = None
        if provider == "openai" and api_key is not None:
            final_api_key = SecretStr(api_key)

        if provider == "ollama":
            if ChatOllama is None:
                raise RuntimeError(
                    "Ollama support is not installed. Run `pip install -r requirements.txt`."
                )
            final_model_name = model_name or DEFAULT_OLLAMA_MODEL
            ollama_url = _ollama_base_url(base_url)
            _validate_ollama(final_model_name, ollama_url)
            self._llm: BaseChatModel = ChatOllama(
                model=final_model_name,
                base_url=ollama_url,
                temperature=temperature,
            )
        else:
            self._llm = ChatOpenAI(
                model=model_name or DEFAULT_OPENAI_MODEL,
                api_key=final_api_key,
                base_url=base_url,
                temperature=temperature,
            )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DEFAULT_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

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
        context_text = "\n\n".join(context_items)

        result = self._chain.invoke({"question": query, "context": context_text})
        if isinstance(result, str):
            return result
        return None
