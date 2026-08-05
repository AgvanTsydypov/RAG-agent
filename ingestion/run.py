from ingestion.chunk import chunk_text
from ingestion.embed import embed_texts
from ingestion.fetch import fetch_tenk_sections
from ingestion.store import delete_filing, ensure_schema, insert_chunks

# Small, recognizable corpus. Swap or extend freely.
TICKERS = ["AAPL", "MSFT"]


def ingest_ticker(ticker: str) -> int:
    sections = fetch_tenk_sections(ticker)
    if not sections:
        print(f"[{ticker}] no 10-K sections found, skipping")
        return 0

    # All sections share one filing identity; clear it once before inserting.
    meta = sections[0]
    delete_filing(meta.ticker, meta.form_type, meta.filing_date)

    rows: list[tuple] = []
    for section in sections:
        chunks = chunk_text(section.text)
        embeddings = embed_texts(chunks)
        for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
            rows.append(
                (
                    section.ticker,
                    section.form_type,
                    section.filing_date,
                    section.section,
                    i,
                    content,
                    embedding,
                )
            )
        print(f"[{ticker}] {section.section}: {len(chunks)} chunks")

    insert_chunks(rows)
    return len(rows)


def main() -> None:
    ensure_schema()
    total = sum(ingest_ticker(ticker) for ticker in TICKERS)
    print(f"Done. Stored {total} chunks total.")


if __name__ == "__main__":
    main()