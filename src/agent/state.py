"""LangGraph agent state.

Deliberately minimal: the orchestrator LLM drives everything through the
conversation `messages`, so we only track the search-loop counter and the
final answer surfaced to the UI.
"""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Full running conversation (user, assistant, tool messages). The
    # add_messages reducer appends; the checkpointer persists it across turns.
    messages: Annotated[list[BaseMessage], add_messages]

    # Number of search rounds done THIS turn. Plain int (no additive reducer):
    # the loop is sequential, and it is reset to 0 at the start of each turn.
    iteration_count: int

    # Final answer text, surfaced to the UI for streaming.
    answer: str
