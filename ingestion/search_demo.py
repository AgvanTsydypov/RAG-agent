import sys

from ingestion.embed import embed_texts
from ingestion.store import connect, to_vector_literal


def search(query: str, k: int = 5) -> list[tuple]:
    query_vector = to_vector_literal(embed_texts([query])[0])
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker,
                       section,
                       filing_date,
                       1 - (embedding <=> %s::vector) AS similarity,
                       left(content, 200)             AS preview
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, k),
            )
            return cur.fetchall()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What are the main risk factors?"
    print(f"Query: {query}\n")
    for ticker, section, filing_date, similarity, preview in search(query):
        print(f"{similarity:.3f} | {ticker} | {section} | {filing_date}")
        print(f"    {preview.strip()}...\n")