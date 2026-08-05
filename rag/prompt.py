from rag.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a financial research assistant answering questions about SEC "
    "filings. Answer using ONLY the provided context. Rules:\n"
    "- If the answer is not in the context, reply exactly: \"I don't have "
    "enough information in the provided filings to answer that.\"\n"
    "- Never use outside knowledge or make assumptions.\n"
    "- Cite the sources you rely on using their [number] labels.\n"
    "- Be concise and factual."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, labeled source blocks."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.ticker} | {chunk.section} | filed {chunk.filing_date}"
        blocks.append(f"{header}\n{chunk.content}")
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"