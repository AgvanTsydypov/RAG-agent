from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI

from app.config import settings


def check_database() -> dict:
    """Open a short-lived connection and confirm pgvector is installed.

    Returns a status dict instead of raising, so /health can report a
    degraded state rather than crashing the whole app.
    """
    try:
        with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
                )
                row = cur.fetchone()
        if row is None:
            return {"connected": True, "pgvector": None}
        return {"connected": True, "pgvector": row[0]}
    except Exception as exc:  # surface any connection error as status text
        return {"connected": False, "error": str(exc)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify the database once so misconfiguration fails loudly early.
    status = check_database()
    if not status["connected"]:
        print(f"[startup] WARNING: database not reachable: {status.get('error')}")
    elif status["pgvector"] is None:
        print("[startup] WARNING: connected, but pgvector extension is missing")
    else:
        print(f"[startup] database OK, pgvector {status['pgvector']}")
    yield
    # Shutdown: nothing to clean up yet.


app = FastAPI(title="RAG Agent", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness plus dependency check, used locally and by Cloud Run later."""
    db = check_database()
    healthy = db["connected"] and db.get("pgvector") is not None
    return {"status": "ok" if healthy else "degraded", "database": db}