"""LLM factory: Ollama (local, primary) with OpenRouter fallback."""

import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_llm_instance: BaseChatModel | None = None
# Human-readable label of the LLM actually serving requests ("Ollama · qwen3:4b"
# or "OpenRouter · xiaomi/mimo-v2.5"). Surfaced in logs and the UI stats panel so
# you always KNOW which model answers — Ollama is tried first, so a running Ollama
# silently wins over the OpenRouter model you configured.
_active_model: str | None = None


def get_active_model() -> str:
    """Label of the LLM currently in use (empty until get_llm() has run)."""
    return _active_model or ""


def _try_ollama() -> BaseChatModel | None:
    global _active_model
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=LLM_TEMPERATURE,
        )
        llm.invoke([HumanMessage(content="ping")])
        _active_model = f"Ollama · {OLLAMA_MODEL}"
        logger.info(f"Ollama connected: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        return llm
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e})")
        return None


def _try_openrouter() -> BaseChatModel | None:
    global _active_model
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set — OpenRouter fallback disabled")
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "http://localhost:7860",
                "X-Title": "Corporate RAG Agent",
            },
            # OpenRouter-specific: turn OFF reasoning so the answer comes back
            # in `content` (not consumed by hidden reasoning tokens) and so the
            # classifier returns clean JSON. Goes via extra_body to stay on the
            # Chat Completions API (not the Responses API).
            extra_body={"reasoning": {"enabled": False}},
        )
        _active_model = f"OpenRouter · {OPENROUTER_MODEL}"
        logger.info(f"OpenRouter active: {OPENROUTER_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"OpenRouter unavailable ({e})")
        return None


def get_llm(force_reload: bool = False) -> BaseChatModel:
    """Return the active LLM. Prefers Ollama, falls back to OpenRouter."""
    global _llm_instance
    if _llm_instance is None or force_reload:
        _llm_instance = _try_ollama() or _try_openrouter()
        if _llm_instance is None:
            raise RuntimeError(
                "No LLM available. Start Ollama or set OPENROUTER_API_KEY in .env"
            )
        # Prominent banner so the active model is never a surprise (Ollama wins
        # over OpenRouter when both are available — this tells you which served).
        logger.warning("=" * 60)
        logger.warning(f"  ACTIVE LLM → {_active_model}")
        logger.warning("=" * 60)
    return _llm_instance
