from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def generate_answer(system_prompt: str, user_message: str) -> str:
    """Single-shot generation grounded in the provided user message."""
    response = _client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""