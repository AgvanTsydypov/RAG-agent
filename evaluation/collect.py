import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import agent_app
from app.streaming import _chunk_text
from rag.pipeline import answer as baseline_answer


def collect_baseline(question: str) -> tuple[str, list[str]]:
    """Baseline contexts are the retrieved chunks themselves."""
    result = baseline_answer(question)
    contexts = [chunk.content for chunk in result.sources]
    return result.answer, contexts


def collect_agent(question: str) -> tuple[str, list[str]]:
    """Agent contexts are its tool outputs; answer is the final AI message."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = agent_app.invoke(
        {"messages": [HumanMessage(content=question)], "steps": 0}, config
    )
    messages = state["messages"]
    contexts = [str(m.content) for m in messages if isinstance(m, ToolMessage)]

    answer_text = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _chunk_text(message)
            if text:
                answer_text = text
                break
    return answer_text, contexts