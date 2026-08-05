from rag.retriever import retrieve


def retrieve_documents(query: str, k: int = 5) -> str:
    """Search SEC 10-K filing text for passages relevant to the query.

    Use for QUALITATIVE questions about a company's business, risk factors, or
    management discussion. Returns numbered source passages with metadata.
    """
    chunks = retrieve(query, k=k)
    if not chunks:
        return "No relevant passages found."
    blocks = [
        f"[{i}] {c.ticker} | {c.section} | {c.filing_date} "
        f"(similarity {c.similarity:.2f})\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)