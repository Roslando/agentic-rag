"""RAGSystem — top-level orchestrator that wires everything together."""

from __future__ import annotations
import logging
import uuid
from typing import Iterator

from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient

from src.config import RECURSION_LIMIT
from src.llm.llm_factory import get_llm
from src.retrieval.vector_store import get_qdrant_client, ensure_collection
from src.agent.graph import build_graph
from src.core.observability import get_observability

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Top-level entry point for the RAG agent.

    Usage:
        rag = RAGSystem()

        # Streaming (token-by-token, consumed by the UI)
        for chunk in rag.stream_chat("What is the refund policy?"):
            print(chunk, end="", flush=True)
    """

    def __init__(self) -> None:
        logger.info("Initializing RAG system...")
        self.llm = get_llm()
        self.qdrant: QdrantClient = get_qdrant_client()
        ensure_collection(self.qdrant)
        self.graph = build_graph(self.llm, self.qdrant)
        self.obs = get_observability()
        self._thread_id = str(uuid.uuid4())   # default single-user thread
        # Preload embedding models now so the first user search isn't hit with
        # ~20s of cold-start model loading.
        from src.retrieval.retriever import warm_up
        warm_up()
        logger.info("RAG system ready.")

    def _config(self, thread_id: str | None = None) -> dict:
        tid = thread_id or self._thread_id
        config: dict = {
            "configurable": {"thread_id": tid},
            "recursion_limit": RECURSION_LIMIT,
        }
        # Attach Langfuse callback → full node-by-node trace of the graph.
        if self.obs.handler is not None:
            config["callbacks"] = [self.obs.handler]
            config["metadata"] = {"langfuse_session_id": tid}
        return config

    def _fresh_input(self, query: str) -> dict:
        """Build the per-turn input. `messages` appends the new question to the
        running history (via the add_messages reducer); `answer` and
        `iteration_count` are reset so neither the previous answer nor a stale
        search counter bleeds into this turn."""
        return {
            "messages": [HumanMessage(content=query)],
            "answer": "",
            "iteration_count": 0,
        }

    def stream_chat(self, query: str, thread_id: str | None = None) -> Iterator[str]:
        """
        Stream the final answer token-by-token for the UI.

        Uses stream_mode="messages": LLM tokens are emitted as they are
        generated. We only forward tokens from the `agent` / `finalize` nodes
        that carry actual text — the agent's tool-decision turns produce empty
        content (just tool calls), so the search-loop never leaks to the user.
        """
        config = self._config(thread_id)

        for chunk, meta in self.graph.stream(
            self._fresh_input(query), config=config, stream_mode="messages"
        ):
            if meta.get("langgraph_node") not in ("agent", "finalize"):
                continue
            text = getattr(chunk, "content", "")
            if isinstance(text, list):  # some providers return content blocks
                text = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in text
                )
            if text:
                yield text
        # Full graph trace is captured automatically via the Langfuse callback.
        self.obs.flush()

    def reset_conversation(self) -> None:
        """Start a fresh conversation thread."""
        self._thread_id = str(uuid.uuid4())
        logger.info(f"New conversation thread: {self._thread_id}")

    def get_collection_stats(self) -> dict:
        from src.retrieval.vector_store import get_collection_info
        from src.ingestion.indexer import load_index_stats
        from src.llm.llm_factory import get_active_model
        try:
            info = get_collection_info(self.qdrant)
            idx = load_index_stats()
            info["total_tokens"] = idx.get("total_tokens", 0)
            info["total_documents"] = idx.get("total_documents", 0)
            info["last_updated"] = idx.get("last_updated", "")
            info["active_model"] = get_active_model()
            return info
        except Exception:
            return {"error": "Collection not found — run ingest_documents.py first"}
