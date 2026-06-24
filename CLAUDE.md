# CLAUDE.md

Production code here is **deliberate, minimal, and verified before it ships**.

---

## Critical rules

**1. No dead code.**
Replacing an approach means deleting the old one in the same change. After every edit, sweep the blast radius: orphaned imports, unused helpers, unreachable branches. Verify with grep + `python -m ruff check` + `python -m pytest`.

**2. Surgical diffs only.**
Every changed line must trace directly to the request. Don't refactor, reformat, or "improve" code you weren't asked to touch.

**3. Show the algorithm before coding.**
For any non-trivial feature: present the exact flow first (steps, input/output data, edge cases) and wait for approval before writing a single line.

**4. Verify before declaring done.**
A task is finished when `python -m pytest` is green — not when the code "looks right."

**5. Ask before assuming.**
Multiple interpretations? Surface them. Simpler path exists? Say so. Something unclear? Stop, name it, ask.

**6. Simplicity first.**
The minimum code that solves the problem. No abstractions for single-use code, no unrequested flexibility, no error handling for impossible cases.

**7. Research-backed recommendations.**
For any design or tooling question: check official docs, return options with tradeoffs, then a clear recommendation with reasoning.

**8. Goal-driven execution.**
Define the success criterion before starting. For multi-step work:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

---

## Architectural invariants (do not break)

- `RAGSystem` is the single entry point — the UI never calls the retriever or agent directly.
- Qdrant + `parent_store/` must always come from the same indexing run (consistent UUIDs). Any re-ingestion or chunker change requires `--recreate`.
- The RRF score is **not** a relevance score. The quality gate is the cross-encoder only.
- `search_documents` queries must be written in **English** (docs are in English; dense + BM25 sparse both require the same language as the indexed chunks).
- API keys come from `.env` — never from committed code.
