"""LangGraph node implementations for the agentic RAG loop."""

from __future__ import annotations
import logging
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.prompts import ORCHESTRATOR_PROMPT, FINALIZE_PROMPT

logger = logging.getLogger(__name__)


def _llm_text(response) -> str:
    """Extract assistant text robustly across providers (handles list-content
    and reasoning models that leave `content` empty)."""
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    content = (content or "").strip()
    if not content:
        ak = getattr(response, "additional_kwargs", {}) or {}
        content = (ak.get("reasoning_content") or ak.get("reasoning") or "").strip()
    # Strip <thinking>...</thinking> blocks (Amazon Nova, some open models leak these)
    content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()
    return content


def agent(state: AgentState, llm_with_tools) -> dict:
    """Orchestrator: the LLM decides whether to search (and how), or to answer.

    - Greeting / small talk → answers directly (no tool call).
    - Factual question → emits tool calls to `search_documents`.
    The graph routes tool calls to the ToolNode and loops back here with the
    results, until the LLM stops calling tools or the search cap is reached.
    """
    messages = [SystemMessage(content=ORCHESTRATOR_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)

    tool_calls = getattr(response, "tool_calls", None) or []
    out: dict = {"messages": [response]}
    if tool_calls:
        # A real search round happened.
        out["iteration_count"] = state.get("iteration_count", 0) + 1
        logger.info(
            f"Agent search round {out['iteration_count']}: "
            + ", ".join(repr(tc["args"].get("query", "")) for tc in tool_calls)
        )
    else:
        # No tool call → this message IS the final answer.
        out["answer"] = _llm_text(response)
    return out


def finalize(state: AgentState, llm) -> dict:
    """Forced answer when the search cap is hit: synthesize from what was found,
    using the plain LLM (no tools) so it cannot keep searching."""
    force = HumanMessage(
        content="Provide your final answer now, using only the information "
        "gathered above."
    )
    response = llm.invoke(
        [SystemMessage(content=FINALIZE_PROMPT)] + list(state["messages"]) + [force]
    )
    answer = _llm_text(response)
    if not answer:
        answer = "I couldn't find this information in the documentation."
    return {"messages": [AIMessage(content=answer)], "answer": answer}
