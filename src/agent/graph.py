"""LangGraph compilation — the agentic RAG loop.

Flow:
    START → agent ⇄ tools        (the LLM searches, up to MAX_SEARCH_ITERATIONS)
              │
              ├─ tool calls + budget left → tools → agent
              ├─ tool calls + cap reached → finalize → END
              └─ no tool call (answer)    → END
"""

from functools import partial
from langchain_core.language_models import BaseChatModel
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import agent, finalize
from src.agent.edges import route_after_agent
from src.agent.tools import get_retrieval_tools, set_retrieval_client


def build_graph(llm: BaseChatModel, qdrant_client: QdrantClient):
    """Build and compile the agentic RAG graph."""
    set_retrieval_client(qdrant_client)
    tools = get_retrieval_tools()
    llm_with_tools = llm.bind_tools(tools)

    p_agent = partial(agent, llm_with_tools=llm_with_tools)
    p_finalize = partial(finalize, llm=llm)

    graph = StateGraph(AgentState)
    graph.add_node("agent", p_agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("finalize", p_finalize)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
