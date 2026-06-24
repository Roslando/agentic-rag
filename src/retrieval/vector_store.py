"""Qdrant local vector store — collection creation and hybrid search setup."""

from pathlib import Path
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

from src.config import QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Return a persistent on-disk Qdrant client."""
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the Qdrant collection with hybrid search vectors if it doesn't exist."""
    exists = client.collection_exists(QDRANT_COLLECTION)

    if exists and recreate:
        logger.info(f"Deleting existing collection '{QDRANT_COLLECTION}'")
        client.delete_collection(QDRANT_COLLECTION)
        exists = False

    if not exists:
        logger.info(f"Creating collection '{QDRANT_COLLECTION}' with hybrid vectors")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("Collection created.")
    else:
        logger.info(f"Collection '{QDRANT_COLLECTION}' already exists.")


def get_collection_info(client: QdrantClient) -> dict:
    info = client.get_collection(QDRANT_COLLECTION)
    # points_count is the reliable field across qdrant-client versions;
    # vectors_count was removed in newer releases.
    points = getattr(info, "points_count", None)
    if points is None:
        points = client.count(QDRANT_COLLECTION).count
    return {
        "name": QDRANT_COLLECTION,
        "points_count": points or 0,
    }
