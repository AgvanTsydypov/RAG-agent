import sys
import uuid

from langchain_core.messages import HumanMessage

from agent.graph import agent_app


def ask(question: str, thread_id: str | None = None) -> None:
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    for event in agent_app.stream(
        {"messages": [HumanMessage(content=question)], "steps": 0},
        config,
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "How did Apple's net sales change from FY2023 to FY2024, in percent?"
    )
    ask(question)