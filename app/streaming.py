import json

from langchain_core.messages import HumanMessage

from agent.graph import agent_app


def _chunk_text(msg) -> str:
    """Extract token text from a streamed message chunk.

    With the Responses API, `.content` may be a list of blocks rather than a
    plain string, so we prefer `.text` and fall back to parsing content.
    """
    text = getattr(msg, "text", None)
    if callable(text):
        text = text()
    if text:
        return text
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def _sse(payload: dict) -> str:
    """Format a payload as one SSE event."""
    return f"data: {json.dumps(payload)}\n\n"


async def stream_agent(question: str, thread_id: str):
    """Yield SSE events: 'step' when a tool is chosen, 'token' for answer text."""
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)], "steps": 0}
    try:
        async for mode, chunk in agent_app.astream(
            inputs, config, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                msg, _metadata = chunk
                text = _chunk_text(msg)
                if text:
                    yield _sse({"type": "token", "text": text})
            elif mode == "updates":
                agent_update = chunk.get("agent")
                if agent_update:
                    last = agent_update["messages"][-1]
                    for call in getattr(last, "tool_calls", None) or []:
                        yield _sse({"type": "step", "tool": call["name"]})
    except Exception as exc:  # surface errors to the client instead of hanging
        yield _sse({"type": "error", "message": str(exc)})
    finally:
        yield _sse({"type": "done"})