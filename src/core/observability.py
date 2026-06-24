"""Langfuse observability integration (optional)."""

import logging

from src.config import (
    LANGFUSE_SECRET_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_HOST,
    ENABLE_LANGFUSE,
)

logger = logging.getLogger(__name__)


class ObservabilityClient:
    """Thin wrapper around Langfuse — silently disabled if keys are missing."""

    def __init__(self) -> None:
        self._client = None
        self._handler = None
        if ENABLE_LANGFUSE:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    secret_key=LANGFUSE_SECRET_KEY,
                    public_key=LANGFUSE_PUBLIC_KEY,
                    host=LANGFUSE_HOST,
                )
                logger.info("Langfuse observability enabled")
                # CallbackHandler traces the FULL LangGraph run (every node +
                # every LLM call) automatically when attached to the run config.
                try:
                    from langfuse.langchain import CallbackHandler
                    self._handler = CallbackHandler()
                    logger.info("Langfuse LangGraph callback handler ready")
                except Exception as e:
                    logger.warning(f"Langfuse CallbackHandler unavailable: {e}")
            except Exception as e:
                logger.warning(f"Langfuse init failed: {e} — tracing disabled")

    @property
    def handler(self):
        """LangChain/LangGraph callback handler, or None if disabled."""
        return self._handler

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass


# Singleton
_obs: ObservabilityClient | None = None


def get_observability() -> ObservabilityClient:
    global _obs
    if _obs is None:
        _obs = ObservabilityClient()
    return _obs
