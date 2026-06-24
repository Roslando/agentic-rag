"""CLI script to ingest PDFs into the RAG system.

Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --docs-dir ./docs --recreate
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DOCS_DIR, MARKDOWN_DIR
from src.ingestion.pdf_converter import convert_all_pdfs
from src.ingestion.chunker import chunk_markdown_document
from src.ingestion.indexer import index_children, save_parents, reset_index_stats
from src.retrieval.vector_store import get_qdrant_client, ensure_collection, get_collection_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection")
    args = parser.parse_args()

    docs_dir: Path = args.docs_dir

    # 1. PDF → Markdown
    logger.info(f"Step 1/3 — Converting PDFs in {docs_dir}")
    md_paths = convert_all_pdfs(docs_dir, MARKDOWN_DIR)
    if not md_paths:
        logger.error("No PDFs converted. Add PDF files to the docs/ folder and retry.")
        sys.exit(1)

    # 2. Init Qdrant
    logger.info("Step 2/3 — Initializing Qdrant collection")
    client = get_qdrant_client()
    ensure_collection(client, recreate=args.recreate)
    if args.recreate:
        reset_index_stats()

    # 3. Chunk + Index
    logger.info("Step 3/3 — Chunking and indexing documents")
    total_children = 0
    for md_path in md_paths:
        parents, children = chunk_markdown_document(md_path)
        save_parents(parents, source=md_path.stem)
        indexed = index_children(client, children, source_name=md_path.stem)
        total_children += indexed

    info = get_collection_info(client)
    logger.info(
        f"\nIngestion complete!\n"
        f"  Documents processed : {len(md_paths)}\n"
        f"  Child chunks indexed: {total_children}\n"
        f"  Total points in DB  : {info['points_count']}"
    )


if __name__ == "__main__":
    main()
