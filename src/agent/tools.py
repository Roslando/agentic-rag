"""LangChain tool wrapping hybrid retrieval, exposed to the agent.

The agent (orchestrator LLM) calls `search_documents` itself, as many times as
it needs (bounded by the graph), refining the query until it has enough to
answer. The Qdrant client is injected once at startup via `set_retrieval_client`.
"""

import tiktoken
from langchain_core.tools import tool
from qdrant_client import QdrantClient

from src.config import SEARCH_CONTEXT_TOKEN_BUDGET

# Token counter (cl100k_base) used to cap the context returned per search so the
# accumulated prompt never exceeds the model/provider prompt limit.
_enc = tiktoken.get_encoding("cl100k_base")

# Injected at runtime so the tool stays a plain function the LLM can call.
_client: QdrantClient | None = None


def set_retrieval_client(client: QdrantClient) -> None:
    global _client
    _client = client


def get_retrieval_tools() -> list:
    """Return the retrieval tools bound to the active Qdrant client."""
    from src.retrieval.retriever import retrieve_with_parents

    @tool
    def search_documents(query: str) -> str:
        """Search the Stripe documentation knowledge base.

        Call this whenever you need factual information to answer the user
        (APIs, fees, webhooks, Checkout, Connect, etc.). You may call it several
        times with refined queries until the results are sufficient.

        Args:
            query: A focused search query in ENGLISH (the docs are in English).
                Include the EXACT API/object/product/version names implied by the
                question (e.g. "Accounts v2 v2/core/accounts", "PaymentIntent
                capture_method") rather than generic words — exact terms surface
                the precise page instead of overview pages.
        """
        if _client is None:
            return "RETRIEVAL_ERROR: vector store not initialized."
        try:
            children, parents = retrieve_with_parents(_client, query)
        except Exception as e:  # never crash the agent loop
            return f"RETRIEVAL_ERROR: {e}"

        docs = parents or children  # already reranked, filtered and capped
        if not docs:
            return (
                "NO_RESULTS: nothing in the documentation cleared the relevance "
                "bar for this query. Try a different query, or tell the user it "
                "is not covered."
            )

        # Build doc blocks, including WHOLE docs (best-ranked first) until the
        # token budget is reached — never truncate mid-document, so code/JSON
        # blocks stay intact. This caps the prompt size and prevents provider
        # "prompt tokens limit exceeded" (402) errors when several searches stack.
        parts: list[str] = []
        used = 0
        for i, doc in enumerate(docs, 1):
            url = doc.get("url", "")
            header = f"[Doc {i}] Source: {doc['source']} | Section: {doc['section']}"
            if url:
                header += f" | URL: {url}"
            block = f"{header}\n{doc['text']}"
            n = len(_enc.encode(block))
            if parts and used + n > SEARCH_CONTEXT_TOKEN_BUDGET:
                break  # keep at least the top doc; stop before overflowing
            parts.append(block)
            used += n
        return "\n\n---\n\n".join(parts)

    return [search_documents]
