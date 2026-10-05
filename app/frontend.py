"""Small HTTP client and response adapters for the Streamlit presentation layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


class BackendError(Exception):
    """Safe, user-facing error raised when a backend request cannot complete."""


EVALUATION_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _evaluation_unavailable() -> Dict[str, Any]:
    return {"available": False, "message": "Evaluation results are currently unavailable."}


def parse_evaluation_artifacts(
    scores_payload: Any,
    results_payload: Any,
    updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and project the saved aggregate and per-question RAGAS artifacts."""
    if not isinstance(scores_payload, dict) or not isinstance(results_payload, list):
        return _evaluation_unavailable()
    if not results_payload:
        return _evaluation_unavailable()

    scores: Dict[str, float] = {}
    for metric in EVALUATION_METRICS:
        value = scores_payload.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            return _evaluation_unavailable()
        scores[metric] = float(value)

    questions = []
    for record in results_payload:
        if not isinstance(record, dict):
            return _evaluation_unavailable()
        if not isinstance(record.get("id"), str) or not isinstance(record.get("question"), str):
            return _evaluation_unavailable()
        metrics_payload = record.get("metrics")
        if not isinstance(metrics_payload, dict):
            return _evaluation_unavailable()
        metrics = {}
        for metric in EVALUATION_METRICS:
            value = metrics_payload.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                return _evaluation_unavailable()
            metrics[metric] = float(value)
        if record.get("evaluation_status") != "completed":
            return _evaluation_unavailable()

        agent = record.get("agent") if isinstance(record.get("agent"), dict) else {}
        questions.append({
            "id": record["id"],
            "question": record["question"],
            "category": record.get("category"),
            "difficulty": record.get("difficulty"),
            "answer": record.get("answer") if isinstance(record.get("answer"), str) else "",
            "metrics": metrics,
            "route": record.get("route", agent.get("route")),
            "evidence_source": record.get("evidence_source"),
        })

    return {
        "available": True,
        "scores": scores,
        "average": round(sum(scores.values()) / len(EVALUATION_METRICS), 4),
        "question_count": len(questions),
        "status": "completed",
        "updated_at": updated_at,
        "questions": questions,
    }


def load_evaluation_artifacts(
    scores_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read the two existing evaluation files without running the evaluator."""
    project_root = Path(__file__).resolve().parents[1]
    scores_file = scores_path or project_root / "evaluation" / "latest_scores.json"
    results_file = results_path or project_root / "evaluation" / "latest_results.json"
    try:
        scores_payload = json.loads(scores_file.read_text(encoding="utf-8"))
        results_payload = json.loads(results_file.read_text(encoding="utf-8"))
        modified = results_file.stat().st_mtime
        from datetime import datetime

        updated_at = datetime.fromtimestamp(modified).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except (OSError, json.JSONDecodeError, ValueError):
        return _evaluation_unavailable()
    return parse_evaluation_artifacts(scores_payload, results_payload, updated_at)


def knowledge_base_state(status: Any) -> str:
    """Return an app-level KB label using only existing health/count information."""
    if not isinstance(status, dict):
        return "Unknown"
    if not status.get("api_healthy") or not status.get("qdrant_healthy"):
        return "Unavailable"
    chunks = status.get("indexed_chunks")
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        return "Ready" if chunks > 0 else "Not indexed"
    return "Unknown"


def parse_query_response(payload: Any) -> Dict[str, Any]:
    """Normalize the existing /query response while tolerating optional fields."""
    if not isinstance(payload, dict):
        raise BackendError("The backend returned an invalid response.")

    answer = payload.get("answer")
    if not isinstance(answer, str):
        raise BackendError("The backend response did not include a valid answer.")

    raw_sources = payload.get("sources")
    sources = (
        [item for item in raw_sources if isinstance(item, dict)]
        if isinstance(raw_sources, list) else []
    )
    raw_diagnostics = payload.get("diagnostics")
    diagnostics = raw_diagnostics if isinstance(raw_diagnostics, dict) else {}
    raw_context = payload.get("generation_context")
    generation_context = (
        [item for item in raw_context if isinstance(item, str)]
        if isinstance(raw_context, list) else []
    )
    rewrite_count = payload.get("rewrite_count")
    if not isinstance(rewrite_count, int) or isinstance(rewrite_count, bool):
        rewrite_count = None
    used_web = payload.get("used_web")
    if not isinstance(used_web, bool):
        used_web = None

    return {
        "answer": answer,
        "sources": sources,
        "route": payload.get("route") if isinstance(payload.get("route"), str) else None,
        "evidence_source": (
            payload.get("evidence_source")
            if isinstance(payload.get("evidence_source"), str) else None
        ),
        "generation_context": generation_context,
        "diagnostics": diagnostics,
        "rewrite_count": rewrite_count,
        "used_web": used_web,
        "latency_ms": (
            payload.get("latency_ms")
            if isinstance(payload.get("latency_ms"), int)
            and not isinstance(payload.get("latency_ms"), bool) else None
        ),
    }


def document_evidence_details(item: Dict[str, Any], index: int) -> list[str]:
    """Format only backend-provided document evidence metadata for the UI."""
    details = [f"Evidence {index}"]
    page = item.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        details.append(f"Page {page}")
    chunk_id = item.get("chunk_id")
    if isinstance(chunk_id, (str, int)) and not isinstance(chunk_id, bool):
        details.append(f"Chunk ID {chunk_id}")
    for key, label in (
        ("rrf_rank", "RRF rank"),
        ("reranker_rank", "Cross-Encoder rank"),
    ):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            details.append(f"{label} {value}")
    score = item.get("reranker_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        details.append(f"Cross-Encoder score {score:.4f}")
    return details


def retrieval_attempt_counts(attempt: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize candidate lists and completed grading decisions from one attempt."""
    counts = {}
    for key, label in (
        ("dense_candidates", "Dense"),
        ("bm25_candidates", "BM25"),
        ("retrieval_candidates", "RRF"),
        ("reranker_candidates", "Cross-Encoder"),
    ):
        value = attempt.get(key)
        if isinstance(value, list):
            counts[label] = len(value)

    graded = attempt.get("reranked_candidates")
    if isinstance(graded, list):
        counts["Selected / graded"] = len(graded)
    decisions = [
        item.get("relevance_decision")
        for item in graded
        if isinstance(item, dict)
        and item.get("relevance_decision") in ("accepted", "rejected")
    ] if isinstance(graded, list) else []
    if decisions:
        accepted = decisions.count("accepted")
        counts["Relevant"] = f"{accepted} / {len(decisions)} graded"
    return counts


def _raise_query_status_error(status_code: int) -> None:
    if status_code == 404:
        raise BackendError(
            "No document index is available yet. Ask the administrator to index the corpus."
        )
    if status_code in (401, 403):
        raise BackendError(
            "The backend provider is not authenticated. Check its server-side configuration."
        )
    if status_code == 503:
        raise BackendError(
            "A required backend service is unavailable. Check the API and Qdrant status."
        )
    if status_code >= 400:
        raise BackendError(
            "The backend could not complete this request. Check its logs and configuration."
        )


def serialize_history(messages: Any) -> list[Dict[str, str]]:
    """Return only the six most recent role/content messages, capped per message."""
    if not isinstance(messages, (list, tuple)):
        return []
    clean = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        clean.append({"role": role, "content": content[:2000]})
    return clean[-6:]


def query_backend(
    base_url: str,
    question: str,
    history: Any = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Call the existing /query endpoint; never surface server exception details."""
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/query",
            json={"question": question, "history": serialize_history(history)},
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise BackendError("The request timed out. Please try again.") from exc
    except httpx.RequestError as exc:
        raise BackendError(
            "The RAG backend is unavailable. Check that the API is running."
        ) from exc

    _raise_query_status_error(response.status_code)
    try:
        payload = response.json()
    except ValueError as exc:
        raise BackendError("The backend returned an invalid response.") from exc
    return parse_query_response(payload)


def parse_ingest_response(payload: Any) -> Dict[str, Any]:
    """Validate the existing /ingest success response without inventing metadata."""
    if not isinstance(payload, dict):
        raise BackendError("The indexing service returned an invalid response.")
    filename = payload.get("filename")
    chunks = payload.get("chunks")
    if payload.get("status") != "indexed" or not isinstance(filename, str):
        raise BackendError("The indexing service did not confirm a successful index.")
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 0:
        raise BackendError("The indexing service returned invalid indexing information.")
    return {"filename": filename, "chunks": chunks}


def ingest_pdf(
    base_url: str,
    filename: str,
    content: bytes,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Upload one PDF through the existing FastAPI /ingest endpoint."""
    if not isinstance(filename, str) or not filename.endswith(".pdf"):
        raise BackendError("Select a PDF file with a .pdf extension.")
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise BackendError("The selected filename is invalid.")
    if not isinstance(content, bytes) or not content:
        raise BackendError("The selected PDF is empty or unreadable.")

    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/ingest",
            files={"file": (filename, content, "application/pdf")},
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise BackendError(
            "PDF indexing timed out. Check the backend and try again."
        ) from exc
    except httpx.RequestError as exc:
        raise BackendError(
            "The indexing service is unavailable. Check that the API is running."
        ) from exc

    if response.status_code == 400:
        raise BackendError("The backend only accepts PDF files. Select a valid .pdf file.")
    if response.status_code == 503:
        raise BackendError(
            "A required indexing service is unavailable. Check API and Qdrant status."
        )
    if response.status_code >= 400:
        raise BackendError(
            "PDF indexing failed. Check backend logs and service configuration."
        )
    try:
        return parse_ingest_response(response.json())
    except ValueError as exc:
        raise BackendError(
            "The indexing service returned an invalid response."
        ) from exc


def get_system_status(base_url: str, timeout: float = 2.0) -> Dict[str, Any]:
    """Read existing health endpoints and return safe application-level status."""
    result: Dict[str, Any] = {
        "api_healthy": False,
        "qdrant_healthy": False,
        "indexed_chunks": None,
        "configuration": None,
    }
    try:
        api_response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        if api_response.is_success:
            api_payload = api_response.json()
            result["api_healthy"] = (
                isinstance(api_payload, dict) and api_payload.get("status") == "ok"
            )
            if isinstance(api_payload, dict) and isinstance(api_payload.get("configuration"), dict):
                allowed = {
                    "llm_provider", "llm_model", "llm_credential_status",
                    "embedding_model", "vector_store", "sparse_retrieval", "fusion",
                    "reranking", "orchestration", "web_fallback", "api", "frontend",
                }
                result["configuration"] = {
                    key: value for key, value in api_payload["configuration"].items()
                    if key in allowed and isinstance(value, str)
                }
    except (httpx.RequestError, ValueError):
        pass

    try:
        qdrant_response = httpx.get(
            f"{base_url.rstrip('/')}/health/qdrant", timeout=timeout
        )
        if qdrant_response.is_success:
            qdrant_payload = qdrant_response.json()
            if isinstance(qdrant_payload, dict) and qdrant_payload.get("status") == "ok":
                result["qdrant_healthy"] = True
                point_count = qdrant_payload.get("points_count")
                if isinstance(point_count, int) and not isinstance(point_count, bool):
                    result["indexed_chunks"] = point_count
    except (httpx.RequestError, ValueError):
        pass
    return result
