"""Conditional routing for the agentic loop."""

from langgraph.graph import END

from src.agent.state import AgentState
from src.config import MAX_SEARCH_ITERATIONS


def route_after_agent(state: AgentState) -> str:
    """After the orchestrator runs:
    - it asked to search and we still have budget  → run the tools, then loop
    - it asked to search but the cap is reached     → force a final answer
    - it produced an answer (no tool call)          → done
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        if state.get("iteration_count", 0) >= MAX_SEARCH_ITERATIONS:
            return "finalize"
        return "tools"
    return END
