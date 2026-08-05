from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts, preserving input order.

    We batch requests to stay well within API limits and reduce round-trips.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = _client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors