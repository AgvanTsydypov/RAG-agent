import time
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph import SYSTEM_PROMPT, _llm_final, _llm_with_tools, agent_app
from agent.tools import AGENT_TOOLS
from app.streaming import _chunk_text
from rag.pipeline import answer as baseline_answer


@dataclass
class Trace:
    """What one system did for the final (graded) turn of one item."""

    answer: str
    contexts: list[str] = field(default_factory=list)  # every tool output in the thread
    tools: list[str] = field(default_factory=list)  # tool names, in call order
    steps: int = 0  # LLM reasoning calls
    latency_s: float = 0.0


def _last_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _chunk_text(message)
            if text:
                return text
    return ""


def _trace_from(thread, turn_start: int, latency_s: float) -> Trace:
    """Tools and steps count only the final turn; contexts span the whole
    thread, since a follow-up may legitimately reuse an earlier lookup."""
    turn = thread[turn_start:]
    return Trace(
        answer=_last_answer(turn),
        contexts=[str(m.content) for m in thread if isinstance(m, ToolMessage)],
        tools=[c["name"] for m in turn if isinstance(m, AIMessage) for c in m.tool_calls],
        steps=sum(isinstance(m, AIMessage) for m in turn),
        latency_s=latency_s,
    )


def collect_baseline(question: str, turns: list[str] | None = None) -> Trace:
    """Linear RAG. It has no memory, so earlier turns are not visible to it."""
    start = time.perf_counter()
    result = baseline_answer(question)
    return Trace(
        answer=result.answer,
        contexts=[chunk.content for chunk in result.sources],
        steps=1,
        latency_s=time.perf_counter() - start,
    )


def collect_agent(question: str, turns: list[str] | None = None) -> Trace:
    """Full LangGraph agent; earlier turns share the thread (checkpointer)."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    # `steps` is checkpointed too, so reset it per question like the API does.
    for turn in turns or []:
        agent_app.invoke({"messages": [HumanMessage(content=turn)], "steps": 0}, config)
    start = time.perf_counter()
    before = len(agent_app.get_state(config).values.get("messages", []))
    state = agent_app.invoke(
        {"messages": [HumanMessage(content=question)], "steps": 0}, config
    )
    return _trace_from(state["messages"], before, time.perf_counter() - start)


# --- Ablation: same model, prompt and tools, but no loop -------------------
# One round of (possibly parallel) tool calls, then a forced answer. The gap
# between this and the agent is what the reason-act-observe loop is worth.
SINGLE_PASS_PROMPT = (
    SYSTEM_PROMPT
    + "\n\nYou get exactly ONE round of tool calls (several tools in parallel is "
    "allowed). After the results come back you must answer; there is no second round."
)
_tools_by_name = {t.name: t for t in AGENT_TOOLS}


def _single_pass_turn(history: list, question: str) -> list:
    messages = [*history, HumanMessage(content=question)]
    system = SystemMessage(content=SINGLE_PASS_PROMPT)
    response = _llm_with_tools.invoke([system, *messages])
    messages.append(response)
    if response.tool_calls:
        for call in response.tool_calls:
            output = _tools_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        messages.append(_llm_final.invoke([system, *messages]))
    return messages


def collect_single_pass(question: str, turns: list[str] | None = None) -> Trace:
    history: list = []
    for turn in turns or []:
        history = _single_pass_turn(history, turn)
    start = time.perf_counter()
    messages = _single_pass_turn(history, question)
    return _trace_from(messages, len(history), time.perf_counter() - start)


SYSTEMS = {
    "baseline": collect_baseline,
    "single_pass": collect_single_pass,
    "agent": collect_agent,
}
