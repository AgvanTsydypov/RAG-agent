import sys
from dataclasses import dataclass

from app.config import settings
from rag.generate import generate_answer
from rag.prompt import SYSTEM_PROMPT, build_user_message
from rag.retriever import RetrievedChunk, retrieve


@dataclass
class RagResult:
    answer: str
    sources: list[RetrievedChunk]


def answer(question: str, k: int | None = None) -> RagResult:
    """The full linear pipeline: retrieve context, then generate an answer."""
    k = k or settings.retrieval_top_k
    chunks = retrieve(question, k=k)
    user_message = build_user_message(question, chunks)
    text = generate_answer(SYSTEM_PROMPT, user_message)
    return RagResult(answer=text, sources=chunks)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What supply chain risks does Apple face?"
    result = answer(question)

    print(f"Q: {question}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for i, chunk in enumerate(result.sources, start=1):
        print(
            f"  [{i}] {chunk.ticker} | {chunk.section} | "
            f"{chunk.filing_date} | sim={chunk.similarity:.3f}"
        )