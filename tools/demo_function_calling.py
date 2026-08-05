"""Minimal OpenAI function-calling demo with our calculator tool.

The loop the agent will rely on in Stage 4:
1. We describe the tool to the model as a JSON schema.
2. The model returns a structured tool call (it does NOT run anything).
3. We execute the tool and feed the result back.
4. The model writes the final answer using that result.
"""
import json

from openai import OpenAI

from app.config import settings
from tools.calculator import calculator

_client = OpenAI(api_key=settings.openai_api_key)

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression on numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "e.g. '(391035 - 383285) / 383285 * 100'",
                }
            },
            "required": ["expression"],
        },
    },
}


def run(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    first = _client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        tools=[CALCULATOR_SCHEMA],
        reasoning_effort=settings.reasoning_effort,
    )
    message = first.choices[0].message

    if not message.tool_calls:
        return message.content or ""

    messages.append(message)  # assistant turn that contains the tool call
    for call in message.tool_calls:
        args = json.loads(call.function.arguments)
        result = calculator(**args)
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result}
        )

    second = _client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        reasoning_effort=settings.reasoning_effort,
    )
    return second.choices[0].message.content or ""


if __name__ == "__main__":
    question = (
        "Revenue went from 383285 to 391035. "
        "What is the percent change? Use the calculator."
    )
    print(run(question))