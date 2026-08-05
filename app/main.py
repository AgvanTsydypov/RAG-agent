from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI
import uuid

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.streaming import stream_agent

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

class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None


@app.post("/chat")
async def chat(request: ChatRequest):
    # A thread_id ties follow-up questions to the same agent memory.
    thread_id = request.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        stream_agent(request.question, thread_id),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id, "Cache-Control": "no-cache"},
    )