#
# repl.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-01.
#

"""A small interactive terminal interface for ResearchLite libraries."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

from research_lite import Document
from research_lite.citations import citation_sources
from research_lite.library_stats import LibraryStats
from research_lite.query_session import (
    HISTORY_TURNS,
    LibraryQuerySession,
    QueryOutcome,
)

HISTORY_ANSWER_CHARS = 500
WAITING_NOTICE = "Generating..."

AnswerGenerator = Callable[[str, Sequence[Document], Sequence[tuple[str, str]]], str | None]
ModeChooser = Callable[[str], tuple[str, tuple[str, ...]]]
OverviewReader = Callable[[Sequence[str]], Mapping[str, object]]
OverviewAnswerer = Callable[[str, Mapping[str, object]], str | None]
AnswerStreamer = Callable[[str, Sequence[Document], Sequence[tuple[str, str]]], Iterator[str]]
Output = Callable[[str], None]
Writer = Callable[[str], None]


def _write_without_newline(text: str) -> None:
    print(text, end="", flush=True)


_HELP_TEXT = """Commands:
  /help       Show this help message.
  /library    Show local library statistics.
  /sources    Show the full retrieved evidence from the last question.
  /reset      Forget the conversation so far and start a fresh topic.
  /exit       Leave the chat. Ctrl-D also exits.

Ask a question to search the loaded library. Follow-up questions use the recent turns to work
out what they refer to; use /reset when you change topic."""

GREETING = "greeting"
ACKNOWLEDGEMENT = "acknowledgement"
FAREWELL = "farewell"
ABOUT = "about"

_SMALLTALK_VOCABULARY: dict[str, str | None] = {
    "hi": GREETING,
    "hii": GREETING,
    "hey": GREETING,
    "heya": GREETING,
    "hiya": GREETING,
    "hello": GREETING,
    "helo": GREETING,
    "yo": GREETING,
    "sup": GREETING,
    "morning": GREETING,
    "afternoon": GREETING,
    "evening": GREETING,
    "day": GREETING,
    "thanks": ACKNOWLEDGEMENT,
    "thank": ACKNOWLEDGEMENT,
    "thx": ACKNOWLEDGEMENT,
    "ty": ACKNOWLEDGEMENT,
    "cheers": ACKNOWLEDGEMENT,
    "ok": ACKNOWLEDGEMENT,
    "okay": ACKNOWLEDGEMENT,
    "cool": ACKNOWLEDGEMENT,
    "nice": ACKNOWLEDGEMENT,
    "great": ACKNOWLEDGEMENT,
    "perfect": ACKNOWLEDGEMENT,
    "awesome": ACKNOWLEDGEMENT,
    "sure": ACKNOWLEDGEMENT,
    "alright": ACKNOWLEDGEMENT,
    "right": ACKNOWLEDGEMENT,
    "yep": ACKNOWLEDGEMENT,
    "yeah": ACKNOWLEDGEMENT,
    "understood": ACKNOWLEDGEMENT,
    "oops": ACKNOWLEDGEMENT,
    "hm": ACKNOWLEDGEMENT,
    "hmm": ACKNOWLEDGEMENT,
    "i": ACKNOWLEDGEMENT,
    "makes": ACKNOWLEDGEMENT,
    "sense": ACKNOWLEDGEMENT,
    "got": ACKNOWLEDGEMENT,
    "no": ACKNOWLEDGEMENT,
    "problem": ACKNOWLEDGEMENT,
    "worries": ACKNOWLEDGEMENT,
    "bye": FAREWELL,
    "goodbye": FAREWELL,
    "see": FAREWELL,
    "ya": FAREWELL,
    "later": FAREWELL,
    "night": FAREWELL,
    "good": None,
    "there": None,
    "you": None,
    "very": None,
    "much": None,
    "so": None,
    "a": None,
    "lot": None,
    "it": None,
}

_ABOUT_PHRASES = frozenset(
    {
        "who are you",
        "what are you",
        "what is this",
        "what can you do",
        "what do you do",
    }
)

_SMALLTALK_REPLIES = {
    GREETING: "Hello. Ask a question about the library, or type /help for commands.",
    ACKNOWLEDGEMENT: "Ask another question whenever you are ready.",
    FAREWELL: "Type /exit to leave the chat.",
    ABOUT: (
        "I answer questions using only the documents in this library, and cite the passages "
        "I used. Type /library to see what is indexed."
    ),
}

_SMALLTALK_WORDS = 4
_PUNCTUATION = ".,!?;:~'\"-"
_BARE_COMMANDS = {"exit": "/exit", "quit": "/exit", "help": "/help"}


def smalltalk_category(question: str) -> str | None:
    """Classify a greeting or an acknowledgement, which needs no search."""
    words = [word.strip(_PUNCTUATION) for word in question.lower().split()]
    words = [word for word in words if word]
    if not words or len(words) > _SMALLTALK_WORDS:
        return None
    if " ".join(words) in _ABOUT_PHRASES:
        return ABOUT
    if any(word not in _SMALLTALK_VOCABULARY for word in words):
        return None
    categories = (_SMALLTALK_VOCABULARY[word] for word in words)
    return next((category for category in categories if category), None)


def smalltalk_reply(question: str) -> str | None:
    """Answer a greeting or an acknowledgement without searching the library."""
    category = smalltalk_category(question)
    return None if category is None else _SMALLTALK_REPLIES[category]


def bare_command(question: str) -> str | None:
    """Map a command typed without its leading slash, such as `exit`, to the command."""
    return _BARE_COMMANDS.get(question.lower().strip(_PUNCTUATION).strip())


def _source_summary(documents: Sequence[Document]) -> str:
    """Return compact citations for an answer printed in the terminal."""
    return "\n".join(
        f"[{source.number}] {source.title} ({source.filename})"
        for source in citation_sources(documents)
    )


class TerminalChat:
    """Run a read-only question loop against a preloaded library session."""

    def __init__(
        self,
        query_session: LibraryQuerySession,
        *,
        generate_answer: AnswerGenerator | None,
        stream_answer: AnswerStreamer | None = None,
        library_stats: LibraryStats | None = None,
        choose_mode: ModeChooser | None = None,
        read_overview: OverviewReader | None = None,
        answer_overview: OverviewAnswerer | None = None,
        output: Output = print,
        write: Writer = _write_without_newline,
    ) -> None:
        self._query_session = query_session
        self._generate_answer = generate_answer
        self._stream_answer = stream_answer
        self._library_stats = library_stats
        self._choose_mode = choose_mode
        self._read_overview = read_overview
        self._answer_overview = answer_overview
        self._output = output
        self._write = write
        self._last_outcome: QueryOutcome | None = None
        self._history: list[tuple[str, str]] = []

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
        if self._library_stats is not None:
            self._output(self._library_stats.display())
        while True:
            try:
                question = prompt.prompt("ResearchLite> ").strip()
            except EOFError:
                self._output("Goodbye.")
                return
            except KeyboardInterrupt:
                self._output("Use /exit to leave the chat.")
                continue

            if not question:
                continue
            question = bare_command(question) or question
            if question in {"/exit", "/quit"}:
                self._output("Goodbye.")
                return
            if question == "/help":
                self._output(_HELP_TEXT)
                continue
            if question == "/library":
                self._show_library()
                continue
            if question == "/sources":
                self._show_sources()
                continue
            if question == "/reset":
                self._history.clear()
                self._last_outcome = None
                self._output("Conversation cleared.")
                continue
            if question.startswith("/"):
                self._output("Unknown command. Type /help for available commands.")
                continue

            reply = smalltalk_reply(question)
            if reply is not None:
                self._output(reply)
            elif not self._route_to_overview(question):
                self._answer(question)

    def _route_to_overview(self, question: str) -> bool:
        """Answer a whole-library question from counted facts, or report that we did not.

        Retrieval returns the best few passages and cannot say how many documents a word
        appears in, so a question about the collection has to be answered from the manifest
        instead. Anything the chooser does not recognise falls through to retrieval.
        """
        if self._choose_mode is None or self._read_overview is None:
            return False
        if self._answer_overview is None:
            return False
        try:
            mode, terms = self._choose_mode(question)
        except Exception:
            return False
        if mode != "overview":
            return False
        try:
            overview = self._read_overview(terms)
            answer = self._answer_overview(question, overview)
        except Exception as exc:
            self._output(f"Library overview failed: {exc}")
            return True
        if not answer:
            return False
        self._last_outcome = None
        self._output(answer.strip())
        self._remember(question, answer.strip())
        return True

    def _answer(self, question: str) -> None:
        try:
            outcome = self._query_session.query(question, self._history)
        except Exception as exc:
            self._output(f"Search failed: {exc}")
            self._output("Leave the chat and start it again if the library was rebuilt meanwhile.")
            return
        self._last_outcome = outcome
        self._print_retrieval_notice(outcome)
        if not outcome.documents:
            self._output("No results returned from retrieval.")
            return

        if self._stream_answer is not None:
            self._stream(question, outcome.documents)
        elif self._generate_answer is None:
            self._output("Generation is disabled. Type /sources to inspect retrieved evidence.")
        else:
            try:
                answer = self._generate_answer(question, outcome.documents, self._history)
            except Exception as exc:
                self._output(f"Failed to generate answer: {exc}")
            else:
                self._output(f"\nAnswer:\n{answer or 'No answer was generated.'}")
                if answer:
                    self._remember(question, answer)
        self._output(f"\nSources:\n{_source_summary(outcome.documents)}")
        self._output("Type /sources for excerpts and full metadata.")

    def _stream(self, question: str, documents: Sequence[Document]) -> None:
        """Print the answer as the model produces it, so a slow model still shows progress."""
        assert self._stream_answer is not None
        self._output("\nAnswer:")
        self._write(WAITING_NOTICE)
        waiting = True
        parts: list[str] = []
        failure: str | None = None
        try:
            for delta in self._stream_answer(question, documents, self._history):
                if waiting:
                    self._clear_notice()
                    waiting = False
                parts.append(delta)
                self._write(delta)
        except Exception as exc:
            failure = str(exc)
        if waiting:
            self._clear_notice()
        elif parts:
            self._write("\n")
        if failure is not None:
            self._output(f"Failed to generate answer: {failure}")
            return
        answer = "".join(parts).strip()
        if answer:
            self._remember(question, answer)
        else:
            self._output("No answer was generated.")

    def _clear_notice(self) -> None:
        self._write(f"\r{'':<{len(WAITING_NOTICE)}}\r")

    def _print_retrieval_notice(self, outcome: QueryOutcome) -> None:
        for warning in outcome.warnings():
            self._output(warning)

    def _remember(self, question: str, answer: str) -> None:
        """Keep the recent turns, trimming each answer so evidence keeps most of the context."""
        self._history.append((question, answer[:HISTORY_ANSWER_CHARS]))
        del self._history[:-HISTORY_TURNS]

    def _show_sources(self) -> None:
        if self._last_outcome is None:
            self._output("No sources yet. Ask a question first.")
            return
        self._output("\nRetrieved evidence:")
        for evidence in self._last_outcome.as_evidence_lines():
            self._output(evidence)

    def _show_library(self) -> None:
        if self._library_stats is None:
            self._output("Library statistics are unavailable for this session.")
            return
        self._output(self._library_stats.display())
