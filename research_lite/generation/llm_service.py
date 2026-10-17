#
# llm_service.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
    "Earlier turns of the conversation may be shown before the current question. Use them only "
    "to resolve what the question refers to; every fact in your answer must come from the "
    "retrieved context.\n\n"
    "Each context item begins with a source citation like [1]. Cite factual claims using only "
    "these numbers, for example [1]. Do not write raw source labels, file paths, chunk IDs, "
    "or metadata.\n\n"
    "Some context items will not bear on the question at all. Ignore those completely rather "
    "than working them into the answer, and cite at most five sources.\n\n"
    "Context:\n{context}"
)
TOOL_FACTS = (
    "\n\nAbout this tool\n"
    "You are ResearchLite, a local command-line tool that searches the user's own document "
    "folders. A question about the tool itself is answered from here, never from the "
    "documents.\n"
    "- add a folder as a library: researchlite library add PATH --name NAME\n"
    "- pick up new or changed files: researchlite library sync NAME\n"
    "- open a library and chat with it: researchlite library use NAME\n"
    "- list the registered libraries: researchlite library list\n"
    "- in this chat: /library statistics, /sources the last question's evidence, /reset, "
    "/help, /exit\n"
    "- no publication dates, authors or venues are recorded, only each document's title and "
    "text\n"
    "- everything runs on this machine; no document is sent anywhere\n"
    "If the tool cannot do what is asked, say so rather than naming another tool."
)
DOCUMENT_LIST = (
    "\n\nThe documents these passages came from\n"
    "{titles}\n"
    "A question about what this library holds is answered from this list only. The passages "
    "mention other people's work; those are not documents in this library, so never name one "
    "in answer to such a question."
)
CHOOSE_MODE = """A tool searches one folder of research papers. It can do two things.

PASSAGES  find the passages that answer a question about what the papers say
OVERVIEW  describe the folder as a whole: how many documents it holds, which titles, how many \
contain a given word, and which fields it does not record at all

Reply with one line and nothing else, in one of these two shapes:

  PASSAGES
  OVERVIEW: the subject words to count, or - when none

Examples:
  Message: what is agentic pressure
  PASSAGES
  Message: what dataset does Better Call Reward use
  PASSAGES
  Message: how many papers do i have on deep learning
  OVERVIEW: deep learning
  Message: which papers are about computer vision
  OVERVIEW: computer vision
  Message: how many papers are there in total
  OVERVIEW: -
  Message: what is the newest paper in the collection
  OVERVIEW: -
  Message: how do i add a folder
  OVERVIEW: -

A message asking what a paper says, reports, uses or proposes needs PASSAGES, even when it \
mentions a number. A message asking how many documents, which documents exist, when one was \
published, or how to use the tool needs OVERVIEW.

Message: {question}"""

OVERVIEW_ANSWER = """You are ResearchLite, a tool that searches one folder of research papers.

The reader asked about the folder itself, so below is what the folder records, taken by \
counting rather than by reading passages. Answer from it alone.

- A count here is exact. Say it plainly.
- `not_recorded` lists what the folder does not hold at all. If the question needs one of \
those, say it is not recorded and stop; do not estimate it.
- `documents_containing_all_terms` counts documents whose text holds every one of `terms`. \
Containing a word is not the same as being about the subject, so say which it is.
- Keep it to two or three sentences. No lists unless the question asks which documents.

{overview}
{tool_facts}"""

_TERM = re.compile(r"[a-z0-9][a-z0-9-]*")
_TERM_STOP = frozenset(
    """the a an of in on for to and or with that this these those is are was were be been
    it its their they them we our you your which who what when where how why not no than
    then there do does did can could may might must will would have has had as by from at
    about into within concerning regarding related relating using involving mentioning
    papers paper documents document pdfs pdf files file library collection folder""".split()
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


def _post_lines(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Iterator[bytes]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            yield from response
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LLMRequestError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise LLMRequestError(f"Could not reach the LLM service at {url}.") from exc


def _event(text: str, url: str) -> dict[str, Any]:
    try:
        payload: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMRequestError(f"LLM service at {url} returned invalid JSON.") from exc
    if payload.get("error"):
        raise LLMRequestError(f"LLM service reported an error: {payload['error']}")
    return payload


def _ollama_deltas(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Iterator[str]:
    """Yield the text of each message in Ollama's newline-delimited JSON stream."""
    for line in _post_lines(url, payload, headers):
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        event = _event(text, url)
        delta = (event.get("message") or {}).get("content")
        if isinstance(delta, str) and delta:
            yield delta
        if event.get("done"):
            return


def _openai_deltas(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Iterator[str]:
    """Yield the text of each delta in an OpenAI-compatible server-sent event stream."""
    for line in _post_lines(url, payload, headers):
        text = line.decode("utf-8", errors="replace").strip()
        if not text.startswith("data:"):
            continue
        data = text[len("data:") :].strip()
        if data == "[DONE]":
            return
        choices = _event(data, url).get("choices") or [{}]
        delta = (choices[0].get("delta") or {}).get("content")
        if isinstance(delta, str) and delta:
            yield delta


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

    def _messages(
        self,
        query: str,
        context_documents: Iterable[Document],
        history: Sequence[tuple[str, str]],
    ) -> list[dict[str, str]]:
        documents = list(context_documents)
        citation_numbers = citation_number_by_source(documents)
        context_items: list[str] = []
        for document in documents:
            source_path = str(document.metadata.get("source_path", ""))
            title = str(document.metadata.get("title", "Unknown"))
            citation_number = citation_numbers.get(source_path or title, 0)
            location = page_label(document.metadata)
            context_items.append(f"[{citation_number}] {title}{location}\n{document.page_content}")
        titles = "\n".join(
            f"- {title}"
            for title in dict.fromkeys(
                str(document.metadata.get("title", "")) for document in documents
            )
            if title
        )
        prompt = DEFAULT_SYSTEM_PROMPT.format(context="\n\n".join(context_items))
        if titles:
            prompt += DOCUMENT_LIST.format(titles=titles)
        messages = [{"role": "system", "content": prompt}]
        for past_question, past_answer in history:
            messages.append({"role": "user", "content": past_question})
            messages.append({"role": "assistant", "content": past_answer})
        messages.append({"role": "user", "content": query})
        return messages

    def _complete(self, messages: list[dict[str, str]], *, limit: int | None = None) -> str | None:
        try:
            if self._provider == "ollama":
                options: dict[str, Any] = {"temperature": self._temperature}
                if limit is not None:
                    options["num_predict"] = limit
                response = _post_json(
                    f"{self._base_url}/api/chat",
                    {
                        "model": self._model,
                        "messages": messages,
                        "stream": False,
                        "options": options,
                    },
                    self._headers,
                )
                content = response["message"]["content"]
            else:
                payload: dict[str, Any] = {"model": self._model, "messages": messages}
                if _supports_temperature(self._model):
                    payload["temperature"] = self._temperature
                if limit is not None:
                    payload["max_tokens"] = limit
                response = _post_json(f"{self._base_url}/chat/completions", payload, self._headers)
                content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("LLM service returned an unexpected response.") from exc
        return content if isinstance(content, str) else None

    def choose_mode(self, question: str) -> tuple[str, tuple[str, ...]]:
        """Pick retrieval or the whole-library overview, and the words the overview counts.

        Retrieval is the fallback for anything unparsed: a whole-library question answered
        from passages is wrong but recoverable, while a content question answered from the
        overview reaches the model with no passages in it at all.
        """
        try:
            reply = self._complete(
                [{"role": "user", "content": CHOOSE_MODE.format(question=question)}], limit=24
            )
        except LLMRequestError:
            return "passages", ()
        head = (reply or "").strip()
        if not head.upper().startswith("OVERVIEW"):
            return "passages", ()
        rest = head[len("OVERVIEW") :].strip(" :\n\t.")
        words = tuple(dict.fromkeys(w for w in _TERM.findall(rest.lower()) if w not in _TERM_STOP))
        return "overview", words

    def answer_overview(self, question: str, overview: Mapping[str, object]) -> str | None:
        """Answer from counted library facts rather than from retrieved passages."""
        prompt = OVERVIEW_ANSWER.format(
            overview=json.dumps(overview, ensure_ascii=False, indent=1), tool_facts=TOOL_FACTS
        )
        return self._complete(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ]
        )

    def generate_answer(
        self,
        query: str,
        context_documents: Iterable[Document],
        history: Sequence[tuple[str, str]] = (),
    ) -> str | None:
        """Generate an answer from the retrieved documents and any earlier turns."""
        return self._complete(self._messages(query, context_documents, history))

    def stream_answer(
        self,
        query: str,
        context_documents: Iterable[Document],
        history: Sequence[tuple[str, str]] = (),
    ) -> Iterator[str]:
        """Yield the answer in the pieces the service sends, so it can be shown as it arrives."""
        messages = self._messages(query, context_documents, history)
        if self._provider == "ollama":
            return _ollama_deltas(
                f"{self._base_url}/api/chat",
                {
                    "model": self._model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": self._temperature},
                },
                self._headers,
            )
        payload: dict[str, Any] = {"model": self._model, "messages": messages, "stream": True}
        if _supports_temperature(self._model):
            payload["temperature"] = self._temperature
        return _openai_deltas(f"{self._base_url}/chat/completions", payload, self._headers)
