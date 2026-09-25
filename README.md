# ResearchLite

A lightweight, local-first research assistant for evidence-grounded literature workflows.

## Incremental document libraries

Initialize a library once, then refresh it as its source files change:

```bash
python -m research_lite.app ./papers --init-library
python -m research_lite.app ./papers --refresh-library
```

The refresh command stores its FAISS index and SQLite manifest under
`./papers/.researchlite`. It embeds only new or changed files, removes chunks
for deleted files, and leaves an existing file's indexed chunks intact if its
replacement cannot be loaded or embedded. Use the same model and chunking
settings for successive refreshes; changing either triggers a full re-index.
