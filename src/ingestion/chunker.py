"""Hierarchical (parent / child) chunking of Markdown documents.

Code-fence aware: fenced blocks (```json, ```curl, ```python, ...) are treated
as ATOMIC units and are never split in the middle. This guarantees that the
agent can return exact, valid code / JSON examples without corruption.

Chunk sizes are measured in real tokens (tiktoken cl100k_base), matching the
token budget logic used in the agent nodes.
"""

from pathlib import Path
from dataclasses import dataclass, field
import re
import uuid
import logging

import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import (
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

_enc = tiktoken.get_encoding("cl100k_base")

# Matches a fenced code block:  ```lang\n ... \n```
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _tok_len(text: str) -> int:
    return len(_enc.encode(text))


def _segment_by_code(text: str) -> list[tuple[str, str]]:
    """
    Split text into ordered segments: ("code", block) for fenced code blocks
    (kept whole), ("text", prose) for everything in between.
    """
    segments: list[tuple[str, str]] = []
    last = 0
    for m in _FENCE_RE.finditer(text):
        if m.start() > last:
            segments.append(("text", text[last : m.start()]))
        segments.append(("code", m.group()))
        last = m.end()
    if last < len(text):
        segments.append(("text", text[last:]))
    return segments


def _split_section(text: str, size: int, overlap: int) -> list[str]:
    """
    Split a section into chunks of ~`size` tokens WITHOUT ever breaking a
    fenced code block. A code block larger than `size` becomes its own chunk,
    kept intact (we never corrupt code / JSON).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=_tok_len,
    )

    # 1. Build a list of atomic units (prose pieces + whole code blocks)
    units: list[str] = []
    for kind, content in _segment_by_code(text):
        if not content.strip():
            continue
        if kind == "code":
            units.append(content)  # atomic — never split, even if oversized
        else:
            units.extend(text_splitter.split_text(content))

    # 2. Greedily pack units into chunks up to `size` tokens
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit_len = _tok_len(unit)
        if current and current_len + unit_len > size:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(unit)
        current_len += unit_len
    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_markdown_document(
    md_path: Path,
    source_name: str | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Split a Markdown file into (parent_chunks, child_chunks).

    Parent chunks hold the full section context (intact code blocks).
    Child chunks are the retrieval targets stored in Qdrant.
    Each child chunk carries a `parent_id` pointing to its parent.
    Fenced code blocks are never split across chunks.
    """
    source = source_name or md_path.stem
    text = md_path.read_text(encoding="utf-8")

    # 1. Split on Markdown headers to get logical (often tiny) sections.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    header_sections = header_splitter.split_text(text)

    # 2. Turn each header section into one or more "blocks" no bigger than
    #    PARENT_CHUNK_SIZE. Small sections stay whole; oversized ones are split
    #    code-fence-aware. Each block keeps its section title for citations.
    blocks: list[tuple[str, str]] = []
    for section in header_sections:
        content = section.page_content
        if not content.strip():
            continue
        section_title = (
            " > ".join(v for k, v in section.metadata.items() if v) or source
        )
        if _tok_len(content) > PARENT_CHUNK_SIZE:
            for piece in _split_section(content, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP):
                blocks.append((section_title, piece))
        else:
            blocks.append((section_title, content))

    parents: list[Chunk] = []
    children: list[Chunk] = []

    # 3. Pack CONSECUTIVE blocks into parents up to ~PARENT_CHUNK_SIZE tokens.
    #    This is the key fix: Stripe docs are densely headered, so without this
    #    every tiny "### Install the library" became its own ~90-token parent
    #    and the small-to-big retrieval gave the LLM almost no context.
    cur_texts: list[str] = []
    cur_title: str | None = None
    cur_len = 0

    def _flush() -> None:
        nonlocal cur_texts, cur_title, cur_len
        if not cur_texts:
            return
        parent_text = "\n\n".join(cur_texts)
        parent = Chunk(
            text=parent_text,
            metadata={
                "source": source,
                "section": cur_title or source,
                "chunk_type": "parent",
            },
        )
        parents.append(parent)
        # Child chunks (retrieval targets) derived from the rich parent.
        for child_text in _split_section(parent_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP):
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        "source": source,
                        "section": cur_title or source,
                        "chunk_type": "child",
                        "parent_id": parent.chunk_id,
                    },
                )
            )
        cur_texts, cur_title, cur_len = [], None, 0

    for title, block in blocks:
        blen = _tok_len(block)
        if cur_texts and cur_len + blen > PARENT_CHUNK_SIZE:
            _flush()
        if cur_title is None:
            cur_title = title  # first (most general) title of the group
        cur_texts.append(block)
        cur_len += blen
    _flush()

    logger.info(
        f"{source}: {len(parents)} parent chunks, {len(children)} child chunks"
    )
    return parents, children
