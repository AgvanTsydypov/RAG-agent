from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools import AGENT_TOOLS
from app.config import settings

MAX_STEPS = 6  # safety valve against infinite tool loops

SYSTEM_PROMPT = """You are a financial research agent answering questions about \
SEC 10-K filings for the companies in the database (currently AAPL and MSFT).

Tools:
- retrieve_documents: semantic search over filing TEXT (business, risks, MD&A). Use for qualitative questions.
- query_financials: exact figures from income statements. Use for quantitative questions (revenue, net income...).
- calculator: arithmetic. ALWAYS use it for any math (growth rates, ratios); never compute in your head.
- web_search: the live web, for recent or external facts NOT in the filings.

Process:
1. Pick the tool the question needs. Numbers -> query_financials. Concepts -> retrieve_documents. Math -> calculator. Recent/outside info -> web_search.
2. After a tool returns, judge whether it actually answers the question. If the result is insufficient or off-target, REFORMULATE and search again (different wording or a different tool) before answering.
3. Ground the answer ONLY in tool results. If the tools cannot answer, say so plainly instead of guessing.
4. Cite what you used (ticker, section or line item, period). Be concise and factual."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


def _make_llm(with_tools: bool) -> ChatOpenAI:
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        reasoning_effort=settings.reasoning_effort,  # "none": tools need this
        use_responses_api=True,                      # route via /v1/responses
    )
    return llm.bind_tools(AGENT_TOOLS) if with_tools else llm


_llm_with_tools = _make_llm(with_tools=True)
_llm_final = _make_llm(with_tools=False)


def agent_node(state: AgentState) -> dict:
    """Reason step: the LLM decides to call tools or to answer."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


def finalize_node(state: AgentState) -> dict:
    """Forced answer when the step limit is reached (no tools available)."""
    guard = SystemMessage(
        content=SYSTEM_PROMPT
        + "\n\nYou have reached the tool-use limit. Answer now using only what "
        "has already been gathered."
    )
    response = _llm_final.invoke([guard, *state["messages"]])
    return {"messages": [response]}


def route(state: AgentState) -> str:
    """Conditional edge: run tools, force a final answer, or finish."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        if state.get("steps", 0) >= MAX_STEPS:
            return "finalize"
        return "tools"
    return END


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(AGENT_TOOLS))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "finalize": "finalize", END: END}
    )
    graph.add_edge("tools", "agent")  # the loop: observe, then reason again
    graph.add_edge("finalize", END)

    # MemorySaver = in-memory checkpointer (dev). Swap for PostgresSaver in prod.
    return graph.compile(checkpointer=MemorySaver())


agent_app = build_agent()