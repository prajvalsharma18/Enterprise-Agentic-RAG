"""
app/api.py
──────────
FastAPI production API — wraps the LangGraph agent.

Endpoints
─────────
  POST /ingest        upload a PDF and index it
  POST /query         ask a question, get an answer + sources
  GET  /health        liveness probe (required for cloud deploy)
  GET  /metrics       RAGAs evaluation scores (if run)

Run locally:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import ask
from app.runtime_info import public_runtime_info
from ingestion.ingest import load_pdfs
from vectorstore.store import build_index
from vectorstore.qdrant_store import check_qdrant_connection


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG Document Assistant",
    description=(
        "Production RAG API with hybrid retrieval, "
        "cross-encoder reranking, LangGraph agentic loop, "
        "and Corrective RAG (query rewriting + web fallback)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR    = Path("data")
METRICS_FILE = Path("evaluation/latest_scores.json")
DATA_DIR.mkdir(exist_ok=True)


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    """Small, serializable browser-session turn accepted by /query."""
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class QueryRequest(BaseModel):
    question:   str
    history:    Optional[List[ConversationMessage]] = None


class SourceItem(BaseModel):
    text:   str
    source: str
    page:   int
    chunk_id: Optional[int] = None
    chunk_index: Optional[int] = None
    rrf_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    reranker_rank: Optional[int] = None
    reranker_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer:        str
    sources:       List[SourceItem]
    rewrite_count: int
    used_web:      bool
    latency_ms:    int
    route:         Optional[str] = None
    evidence_source: Optional[str] = None
    generation_context: List[str] = Field(default_factory=list)
    diagnostics:   Dict[str, Any] = Field(default_factory=dict)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — required for Docker/cloud deploy."""
    return {
        "status": "ok",
        "version": app.version,
        "configuration": public_runtime_info(),
    }


@app.get("/health/qdrant")
def qdrant_health():
    """Read-only Qdrant connection, collection, and vector compatibility check."""
    try:
        return check_qdrant_connection()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and add it to the vector index.
    Existing index is rebuilt to include the new document.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = DATA_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    logger.info(f"Saved uploaded file: {dest}")

    try:
        chunks, metas = load_pdfs(DATA_DIR)
        build_index(chunks, metas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "status":    "indexed",
        "filename":  file.filename,
        "chunks":    len(chunks),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question. The LangGraph agent:
    1. Routes (documents vs LLM-only)
    2. Retrieves with hybrid FAISS+BM25
    3. Reranks with cross-encoder
    4. Grades relevance — rewrites query if needed (Corrective RAG)
    5. Falls back to web search if docs fail after max rewrites
    6. Generates a cited answer
    """
    t0 = time.time()
    try:
        history = [
            HumanMessage(content=message.content)
            if message.role == "user" else AIMessage(content=message.content)
            for message in (req.history or [])[-6:]
        ]
        result = ask(req.question, history=history) if history else ask(req.question)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No index found. Upload a PDF via POST /ingest first."
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = int((time.time() - t0) * 1000)
    logger.info(f"Query answered in {latency}ms | rewrites={result['rewrite_count']}")

    return QueryResponse(
        answer        = result["answer"],
        sources       = [SourceItem(**s) for s in result["sources"]],
        rewrite_count = result["rewrite_count"],
        used_web      = result["used_web"],
        latency_ms    = latency,
        route         = result.get("route"),
        evidence_source = result.get("evidence_source"),
        generation_context = result.get("generation_context", []),
        diagnostics   = result.get("diagnostics", {}),
    )


@app.get("/metrics")
def metrics():
    """Return latest RAGAs evaluation scores if available."""
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text())
    return {"detail": "No evaluation scores yet. Run: python -m evaluation.evaluate"}
