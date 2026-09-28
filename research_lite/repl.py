"""A small interactive terminal interface for ResearchLite libraries."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_core.documents import Document

from research_lite.query_session import LibraryQuerySession, QueryOutcome, format_evidence

AnswerGenerator = Callable[[str, Sequence[Document]], str | None]
Output = Callable[[str], None]

_HELP_TEXT = """Commands:
  /help       Show this help message.
  /sources    Show the full retrieved evidence from the last question.
  /exit       Leave the chat. Ctrl-D also exits.

Ask a question to search the loaded library. Each question is retrieved independently."""


def _source_summary(documents: Sequence[Document]) -> str:
    """Return compact citations for an answer printed in the terminal."""
    sources = []
    for index, document in enumerate(documents, start=1):
        title = document.metadata.get("title", "Unknown")
        source_path = document.metadata.get("source_path", "Unknown")
        sources.append(f"[{index}] {title} ({source_path})")
    return "\n".join(sources)


class TerminalChat:
    """Run a read-only question loop against a preloaded library session."""

    def __init__(
        self,
        query_session: LibraryQuerySession,
        *,
        generate_answer: AnswerGenerator | None,
        output: Output = print,
    ) -> None:
        self._query_session = query_session
        self._generate_answer = generate_answer
        self._output = output
        self._last_documents: list[Document] = []

    def run(self) -> None:
        """Start the prompt loop until the user exits or closes standard input."""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import InMemoryHistory
        except ImportError as exc:
            msg = "Interactive chat requires prompt_toolkit. Run `pip install -r requirements.txt`."
            raise RuntimeError(msg) from exc

        prompt: PromptSession[str] = PromptSession(history=InMemoryHistory())
        self._output("ResearchLite chat is ready. Type /help for commands.")
        while True:
            try:
                question = prompt.prompt("You> ").strip()
            except EOFError:
                self._output("Goodbye.")
                return
            except KeyboardInterrupt:
                self._output("Use /exit to leave the chat.")
                continue

            if not question:
                continue
            if question in {"/exit", "/quit"}:
                self._output("Goodbye.")
                return
            if question == "/help":
                self._output(_HELP_TEXT)
                continue
            if question == "/sources":
                self._show_sources()
                continue
            if question.startswith("/"):
                self._output("Unknown command. Type /help for available commands.")
                continue

            self._answer(question)

    def _answer(self, question: str) -> None:
        outcome = self._query_session.query(question)
        self._last_documents = outcome.documents
        self._print_retrieval_notice(outcome)
        if not outcome.documents:
            self._output("No results returned from retrieval.")
            return

        if self._generate_answer is None:
            self._output("Generation is disabled. Type /sources to inspect retrieved evidence.")
        else:
            try:
                answer = self._generate_answer(question, outcome.documents)
            except Exception as exc:
                self._output(f"Failed to generate answer: {exc}")
            else:
                self._output(f"\nAnswer:\n{answer or 'No answer was generated.'}")
        self._output(f"\nSources:\n{_source_summary(outcome.documents)}")
        self._output("Type /sources for excerpts and full metadata.")

    def _print_retrieval_notice(self, outcome: QueryOutcome) -> None:
        if outcome.reranker_unavailable:
            self._output("Reranker unavailable; using hybrid retrieval instead.")

    def _show_sources(self) -> None:
        if not self._last_documents:
            self._output("No sources yet. Ask a question first.")
            return
        self._output("\nRetrieved evidence:")
        for evidence in format_evidence(self._last_documents):
            self._output(evidence)
