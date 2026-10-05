"""Safe, non-secret runtime information shared by the API and frontend."""

from __future__ import annotations

import os
from typing import Mapping, Optional


def public_runtime_info(environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Describe selected runtime components without returning credential values."""
    config = os.environ if environ is None else environ
    provider = config.get("LLM_PROVIDER", "openai").strip().lower() or "openai"

    if provider == "openai":
        llm_provider = "OpenAI"
        llm_model = config.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        credential_status = (
            "Configured" if config.get("OPENAI_API_KEY", "").strip()
            else "Not configured"
        )
    elif provider == "ollama":
        llm_provider = "Ollama"
        llm_model = config.get("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
        credential_status = "Not required by this provider"
    else:
        llm_provider = "Unknown"
        llm_model = "Not available"
        credential_status = "Provider configuration unknown"

    return {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_credential_status": credential_status,
        "embedding_model": "BAAI/bge-m3",
        "vector_store": config.get("VECTOR_STORE", "qdrant").strip().upper() or "QDRANT",
        "sparse_retrieval": "BM25",
        "fusion": "RRF",
        "reranking": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "orchestration": "LangGraph",
        "web_fallback": (
            "Tavily (configured)" if config.get("TAVILY_API_KEY", "").strip()
            else "Tavily (not configured)"
        ),
        "api": "FastAPI",
        "frontend": "Streamlit",
    }
