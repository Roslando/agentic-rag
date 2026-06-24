"""Embed child chunks and store them in Qdrant. Persist parent chunks as JSON."""

import json
import logging
from datetime import datetime

import tiktoken
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from src.config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    PARENT_STORE_DIR,
    QDRANT_COLLECTION,
)
from src.ingestion.chunker import Chunk

INDEX_STATS_PATH = DATA_DIR / "index_stats.json"
_tok = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tok.encode(text))

logger = logging.getLogger(__name__)

_dense_model: HuggingFaceEmbeddings | None = None
_sparse_model: SparseTextEmbedding | None = None


def _get_dense_model() -> HuggingFaceEmbeddings:
    global _dense_model
    if _dense_model is None:
        logger.info(f"Loading dense embedding model: {EMBEDDING_MODEL}")
        _dense_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        logger.info("Loading sparse BM25 model")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def save_parents(parents: list[Chunk], source: str) -> None:
    """Persist parent chunks as a JSON file keyed by chunk_id."""
    PARENT_STORE_DIR.mkdir(parents=True, exist_ok=True)
    store_path = PARENT_STORE_DIR / f"{source}.json"
    data = {p.chunk_id: {"text": p.text, "metadata": p.metadata} for p in parents}
    store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(parents)} parent chunks → {store_path}")


def load_all_parents(source: str) -> dict:
    store_path = PARENT_STORE_DIR / f"{source}.json"
    if not store_path.exists():
        return {}
    return json.loads(store_path.read_text(encoding="utf-8"))


def load_index_stats() -> dict:
    """Load persistent indexation statistics."""
    if INDEX_STATS_PATH.exists():
        return json.loads(INDEX_STATS_PATH.read_text(encoding="utf-8"))
    return {"total_tokens": 0, "total_chunks": 0, "total_documents": 0, "sources": {}}


def save_index_stats(stats: dict) -> None:
    INDEX_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats["last_updated"] = datetime.now().isoformat(timespec="seconds")
    INDEX_STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_index_stats() -> None:
    """Wipe stats (call this when recreating the collection)."""
    save_index_stats({"total_tokens": 0, "total_chunks": 0, "total_documents": 0, "sources": {}})


def index_children(
    client: QdrantClient,
    children: list[Chunk],
    batch_size: int = 32,
    source_name: str = "",
) -> int:
    """Embed and upsert child chunks into Qdrant. Returns number of indexed points."""
    if not children:
        return 0

    dense_model = _get_dense_model()
    sparse_model = _get_sparse_model()

    texts = [c.text for c in children]
    total = 0
    chunk_tokens = sum(_count_tokens(t) for t in texts)

    for i in range(0, len(texts), batch_size):
        batch_chunks = children[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        dense_vecs = dense_model.embed_documents(batch_texts)
        sparse_results = list(sparse_model.embed(batch_texts))

        points = []
        for chunk, dense_vec, sparse_result in zip(batch_chunks, dense_vecs, sparse_results):
            sparse_vec = SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist(),
            )
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector={"dense": dense_vec, "sparse": sparse_vec},
                    payload={**chunk.metadata, "text": chunk.text},
                )
            )

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        total += len(points)
        logger.info(f"Indexed batch {i // batch_size + 1}: {len(points)} points")

    # Persist stats
    stats = load_index_stats()
    stats["total_tokens"] += chunk_tokens
    stats["total_chunks"] += total
    if source_name:
        stats["sources"][source_name] = {
            "chunks": total,
            "tokens": chunk_tokens,
        }
        stats["total_documents"] = len(stats["sources"])
    save_index_stats(stats)
    logger.info(f"Stats updated: +{chunk_tokens:,} tokens, +{total} chunks")

    return total
