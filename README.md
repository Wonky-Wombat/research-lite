# ResearchLite

A lightweight, local-first research assistant for evidence-grounded literature workflows.

Point it at a folder of papers and ask questions about them. Indexing and search run entirely
on your machine: no API key, no Docker, no database server, and no document ever leaves the
computer. Answers carry numbered citations back to the passages they came from.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Needs Python 3.12 or newer. The embedding and reranker models download on first use and are
cached; after that, indexing and search work offline. Answer generation needs either a local
[Ollama](https://ollama.com) server or an OpenAI-compatible endpoint.

## Use

```bash
researchlite library add ~/papers --name work   # index a folder and open it
researchlite                                    # chat with the active library
researchlite library sync work                  # pick up new or changed files
researchlite library list                       # list local libraries
```

In the chat: `/sources` shows the evidence behind the last answer, `/library` the current
counts, `/reset` clears the conversation, `/help` lists the rest.

Non-interactive, for scripting or for another tool to consume:

```bash
researchlite --query "what does agentic pressure measure" --json   # evidence as JSON
researchlite --query "..." --no-generation                         # retrieval only
researchlite --overview "deep learning"                            # whole-library counts
researchlite --context CHUNK_ID --context-radius 1                 # a passage in context
```

`--overview` answers questions about the collection rather than about what the papers say:
how many documents it holds, which titles, and how many contain a given set of words. The
counts are exact, taken by SQL over the manifest, and no model is loaded.

## From Claude Code, Codex or another MCP host

```bash
claude mcp add researchlite -- researchlite mcp ~/papers
```

The host's own model writes the answer; this serves it the passages. Six tools: `search`,
`overview`, `read_context`, `status`, `sync_start` and `sync_status`. There is deliberately
no tool that returns a whole document — a host that can ask for one will, and the point is
to keep papers out of its context.

Measured against the same host answering the same questions by opening the files itself,
across a library of 100 papers:

| | with these tools | reading the files |
|---|---|---|
| tokens | **170k** | 523k |
| context per turn | **8.5k** (24 turns before a 200k window fills) | 22.3k (9 turns) |
| wall clock | **28s** | 62s |
| right answer | **12 of 12** | 10 of 12 |

The saving is in not knowing which paper holds the answer. Told which file to read, a host
is slightly better off reading it.

Speaks the current protocol revision and the older `initialize` handshake, over stdin and
stdout, with no protocol library: the official SDK's sixteen required dependencies exist
mostly for an HTTP transport a local server never opens.

## What it does differently

**References and page furniture are demoted.** Every chunk is classified as body text,
references, boilerplate or running text. A reference list that happens to mention your search
terms will not outrank the paper that actually discusses them, and the names in it are not
treated as documents you own.

**Citations map to documents, not to chunks.** The model is given numbered sources and cites
`[1]`, `[2]`; those numbers resolve to papers, so an answer drawn from three passages of one
paper cites it once.

**It says what it does not know.** The library records each document's title, text and page
numbers — not authors, publication dates or venues. Asked for one of those, it says so rather
than inferring it from a passage.

## How it works

Retrieval is dense vectors (`sentence-transformers/all-MiniLM-L6-v2` via fastembed/ONNX)
fused with SQLite FTS5 keyword search by reciprocal rank fusion, then reranked by a
cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`). On a 40-paper holdout set that took no part
in tuning, each stage earns its place:

| mode | right paper in context | ranked first | per query |
|---|---|---|---|
| `dense` | 65.2% | 52.9% | 0.003 s |
| `hybrid` | 68.9% | 55.3% | 0.010 s |
| `hybrid-rerank` (default) | 73.4% | 60.7% | 0.337 s |

Pick one with `--retrieval-mode`. The reranker is most of the query time; `dense` is nearly
free if you would rather have the latency back.

Generation is a single LLM call over the retrieved passages — not one call per chunk — which
is why an answer takes seconds rather than most of a minute.

## Layout

The retrieval layer offers modes and does not decide which to use; a host agent, or the
built-in chat, chooses. `research_lite/query_session.py` and `retrieval.py` are the search
core, `manifest.py` the SQLite index, `ingestion/` the per-format loaders, `generation/` the
LLM client, `repl.py` and `app.py` the interactive and command-line shells.

Supported formats: PDF, DOCX, PPTX, XLSX, XLS, CSV, EPUB, HTML, Markdown, JSON, TXT.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
