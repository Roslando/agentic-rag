"""
Index pre-existing Markdown files directly into Qdrant (bypasses PDF conversion).

Usage:
    # Index Stripe docs (after running fetch_stripe_docs.py)
    python scripts/index_markdown_files.py

    # Custom directory
    python scripts/index_markdown_files.py --markdown-dir data/markdown_docs/stripe

    # Recreate the collection (wipe + rebuild)
    python scripts/index_markdown_files.py --recreate
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MD_DIR = Path(__file__).parent.parent / "data" / "markdown_docs" / "stripe"


def load_meta(md_path: Path) -> dict:
    """Load companion .meta.json file if it exists."""
    meta_path = md_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Markdown files into Qdrant")
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=DEFAULT_MD_DIR,
        help="Directory containing .md files to index",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing",
    )
    args = parser.parse_args()

    md_dir: Path = args.markdown_dir

    if not md_dir.exists():
        logger.error(f"Directory not found: {md_dir}")
        logger.error("Run scripts/fetch_stripe_docs.py first.")
        sys.exit(1)

    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        logger.error(f"No .md files found in {md_dir}")
        sys.exit(1)

    logger.info(f"Found {len(md_files)} Markdown files in {md_dir}")

    # Imports here to give a fast --help
    from src.retrieval.vector_store import get_qdrant_client, ensure_collection, get_collection_info
    from src.ingestion.chunker import chunk_markdown_document
    from src.ingestion.indexer import save_parents, index_children, reset_index_stats

    client = get_qdrant_client()
    ensure_collection(client, recreate=args.recreate)
    if args.recreate:
        reset_index_stats()

    total_files = 0
    total_children = 0
    failed = []

    for i, md_path in enumerate(md_files, 1):
        try:
            logger.info(f"[{i}/{len(md_files)}] Processing: {md_path.name}")

            # Load companion metadata (url, category)
            meta = load_meta(md_path)

            # Chunk the Markdown file (reuses existing chunker, unchanged)
            parents, children = chunk_markdown_document(md_path)

            if not children:
                logger.warning(f"  No chunks produced for {md_path.name} — skipping")
                continue

            # Inject url + category into every chunk's metadata
            for chunk in parents + children:
                if meta.get("url"):
                    chunk.metadata["url"] = meta["url"]
                if meta.get("category"):
                    chunk.metadata["category"] = meta["category"]

            # Persist parents (for context retrieval) + index children in Qdrant
            save_parents(parents, source=md_path.stem)
            n = index_children(client, children, source_name=md_path.stem)

            total_files += 1
            total_children += n
            logger.info(f"  OK {len(parents)} parents, {n} children indexed")

        except Exception as e:
            logger.error(f"  [FAIL] {md_path.name} - {e}")
            failed.append(md_path.name)

    info = get_collection_info(client)

    print(
        f"\nIndexation complete!\n"
        f"  Files processed       : {total_files}/{len(md_files)}\n"
        f"  Child chunks indexed  : {total_children}\n"
        f"  Total points in Qdrant: {info['points_count']}\n"
        + (f"  [FAIL] files          : {', '.join(failed)}\n" if failed else "")
    )


if __name__ == "__main__":
    main()
