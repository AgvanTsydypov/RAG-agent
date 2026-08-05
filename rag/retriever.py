from dataclasses import dataclass

from ingestion.embed import embed_texts
from ingestion.store import connect, to_vector_literal


@dataclass
class RetrievedChunk:
    ticker: str
    section: str
    filing_date: str
    content: str
    similarity: float


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Embed the query and return the k most similar chunks from pgvector."""
    query_vector = to_vector_literal(embed_texts([query])[0])
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker,
                       section,
                       filing_date,
                       content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, k),
            )
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            ticker=ticker,
            section=section,
            filing_date=str(filing_date),
            content=content,
            similarity=float(similarity),
        )
        for ticker, section, filing_date, content, similarity in rows
    ]