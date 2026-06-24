"""Hybrid retrieval (dense + sparse BM25) from Qdrant, with cross-encoder rerank."""

import logging
import math
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector,
    FusionQuery,
    Prefetch,
    Fusion,
)
from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    RETRIEVAL_TOP_K,
    RERANK_MODEL,
    RERANK_CANDIDATES,
    RERANK_THRESHOLD,
    RERANK_KEEP,
)
from src.ingestion.indexer import load_all_parents

logger = logging.getLogger(__name__)

_dense_model: HuggingFaceEmbeddings | None = None
_sparse_model: SparseTextEmbedding | None = None
_reranker = None


def _get_dense_model() -> HuggingFaceEmbeddings:
    global _dense_model
    if _dense_model is None:
        _dense_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _rerank(query: str, children: list[dict]) -> list[dict]:
    """Re-score (query, child) pairs with the cross-encoder, keep only those
    above RERANK_THRESHOLD, sorted by true relevance. This is the real quality
    gate — unlike the RRF score, the cross-encoder score IS a 0-1 relevance."""
    if not children:
        return []
    try:
        reranker = _get_reranker()
        raw = reranker.predict([(query, c["text"]) for c in children])
    except Exception as e:  # if the model can't load, don't break retrieval
        logger.warning(f"Reranker unavailable ({e}) — using fusion order as-is")
        return children

    for c, logit in zip(children, raw):
        c["rerank_score"] = 1.0 / (1.0 + math.exp(-float(logit)))  # sigmoid → 0-1

    kept = [c for c in children if c["rerank_score"] >= RERANK_THRESHOLD]
    kept.sort(key=lambda c: c["rerank_score"], reverse=True)
    logger.info(
        f"Rerank: {len(kept)}/{len(children)} children passed "
        f"threshold {RERANK_THRESHOLD} for {query!r}"
    )
    return kept


def warm_up() -> None:
    """Preload the embedding models AND the reranker (and run tiny encodes) so the
    FIRST real search isn't penalized by cold-start model loading. Called once at
    RAGSystem startup."""
    logger.info("Warming up models (dense + BM25 + reranker)...")
    _get_dense_model().embed_query("warm up")
    list(_get_sparse_model().embed(["warm up"]))
    try:
        _get_reranker().predict([("warm up", "warm up")])
    except Exception as e:
        logger.warning(f"Reranker warm-up failed ({e})")
    logger.info("Models ready.")


def hybrid_search(
    client: QdrantClient,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Perform hybrid search (dense cosine + BM25 sparse) with RRF fusion.
    Returns a list of dicts: {text, source, section, url, parent_id, score}.

    Note: no score_threshold is applied — RRF fusion scores are on a tiny
    reciprocal-rank scale (~0.03), not cosine 0-1, so a 0.4 threshold would be
    meaningless (it would drop everything). We rank by fusion and keep top_k.
    """
    dense_model = _get_dense_model()
    sparse_model = _get_sparse_model()

    dense_vec = dense_model.embed_query(query)
    sparse_result = list(sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k * 2,
            ),
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "section": payload.get("section", ""),
                "url": payload.get("url", ""),
                "parent_id": payload.get("parent_id", ""),
                "score": point.score,
            }
        )

    logger.info(f"Retrieved {len(chunks)} chunks for {query!r}")
    return chunks


def retrieve_with_parents(
    client: QdrantClient,
    query: str,
    candidates: int = RERANK_CANDIDATES,
    keep: int = RERANK_KEEP,
) -> tuple[list[dict], list[dict]]:
    """
    Hybrid search (large candidate pool) → cross-encoder rerank + threshold →
    expand the survivors to parent chunks. Returns (reranked_children, parents).

    Only parents whose best child cleared the relevance bar are returned, capped
    at `keep`. If nothing clears the bar, both lists are empty (the agent will
    then re-search or answer honestly that it found nothing).
    """
    children = hybrid_search(client, query, candidates)
    children = _rerank(query, children)  # filtered + sorted by true relevance
    if not children:
        return [], []

    # Group by source to load parent stores efficiently
    parents_by_source: dict[str, dict] = {}
    for child in children:
        src = child["source"]
        if src not in parents_by_source:
            parents_by_source[src] = load_all_parents(src)

    parent_chunks = []
    seen_parent_ids: set[str] = set()

    for child in children:  # already in best-relevance-first order
        pid = child.get("parent_id", "")
        if pid and pid not in seen_parent_ids:
            src = child["source"]
            parent_data = parents_by_source.get(src, {}).get(pid)
            if parent_data:
                parent_chunks.append(
                    {
                        "text": parent_data["text"],
                        "source": src,
                        "section": parent_data["metadata"].get("section", ""),
                        "url": parent_data["metadata"].get("url", ""),
                        "parent_id": pid,
                        "score": child.get("rerank_score", child.get("score", 0)),
                    }
                )
                seen_parent_ids.add(pid)
        if len(parent_chunks) >= keep:  # cap on the most relevant parents
            break

    return children, parent_chunks
