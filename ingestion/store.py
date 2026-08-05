import psycopg
from pgvector.psycopg import register_vector

from app.config import settings

# Idempotent DDL. We define the schema in code (not in db/init) because
# init scripts only run on a fresh volume — see the Stage 0 pitfall.
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    ticker       text         NOT NULL,
    form_type    text         NOT NULL,
    filing_date  date         NOT NULL,
    section      text         NOT NULL,
    chunk_index  int          NOT NULL,
    content      text         NOT NULL,
    embedding    vector(1536) NOT NULL,
    created_at   timestamptz  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_ticker_idx ON chunks (ticker);

CREATE TABLE IF NOT EXISTS financials (
    id          bigserial PRIMARY KEY,
    ticker      text    NOT NULL,
    statement   text    NOT NULL,
    line_item   text    NOT NULL,
    concept     text,
    period      text    NOT NULL,
    value       numeric NOT NULL,
    created_at  timestamptz DEFAULT now()
);

ALTER TABLE financials ADD COLUMN IF NOT EXISTS concept text;

CREATE INDEX IF NOT EXISTS financials_ticker_idx ON financials (ticker);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def ensure_schema() -> None:
    # DDL needs no vector adaptation, so we don't register the type here.
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


def delete_filing(ticker: str, form_type: str, filing_date: str) -> None:
    """Remove prior rows for a filing so re-ingestion stays idempotent."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks "
                "WHERE ticker = %s AND form_type = %s AND filing_date = %s",
                (ticker, form_type, filing_date),
            )
        conn.commit()


def insert_chunks(rows: list[tuple]) -> None:
    """Insert chunk rows. Each row:
    (ticker, form_type, filing_date, section, chunk_index, content, embedding).
    """
    with connect() as conn:
        register_vector(conn)  # lets psycopg adapt Python lists to vector()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks "
                "(ticker, form_type, filing_date, section, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
        conn.commit()

def to_vector_literal(vec: list[float]) -> str:
    """Render a float list into pgvector's text format: [x,y,z].

    We pass this with an explicit ::vector cast so Postgres treats the
    parameter as a vector instead of a plain float array.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"

def delete_financials(ticker: str) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM financials WHERE ticker = %s", (ticker,))
        conn.commit()


def insert_financials(rows: list[tuple]) -> None:
    """Rows: (ticker, statement, line_item, concept, period, value)."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO financials "
                "(ticker, statement, line_item, concept, period, value) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                rows,
            )
        conn.commit()


def query_financials_rows(ticker: str, metric: str, limit: int = 50) -> list[tuple]:
    """Read-only, parameterized lookup. Match either the human label or the
    XBRL concept, so 'revenue' hits even when the label reads 'Net sales'."""
    pattern = f"%{metric}%"
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT line_item, period, value FROM financials "
                "WHERE ticker = %s AND (line_item ILIKE %s OR concept ILIKE %s) "
                "ORDER BY line_item, period "
                "LIMIT %s",
                (ticker, pattern, pattern, limit),
            )
            return cur.fetchall()