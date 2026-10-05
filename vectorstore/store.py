"""
vectorstore/store.py
────────────────────
Hybrid retrieval: dense Qdrant/FAISS  +  sparse BM25
Results fused with Reciprocal Rank Fusion (RRF).

This replaces your original FAISS-only search and gives significantly
better recall — especially for keyword-heavy queries.
"""

from __future__ import annotations
import json
import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from embeddings.embedder import embed_texts, embed_query, rerank
from vectorstore.qdrant_store import (
    search_chunks as qdrant_search_chunks,
    upsert_chunks as qdrant_upsert_chunks,
)


STORE_DIR   = Path("vectorstore/index")
FAISS_PATH  = STORE_DIR / "faiss.index"
META_PATH   = STORE_DIR / "metadata.pkl"
BM25_PATH   = STORE_DIR / "bm25.pkl"

EMBED_DIM   = 1024   # BAAI/bge-m3 output dimension
TOP_K_FETCH = 20     # fetch more, rerank down to fewer
TOP_K_FINAL = 5      # chunks returned to the LLM
RRF_K       = 60     # standard RRF constant
_ROUTING_STOP_WORDS = {
    "about", "after", "also", "and", "are", "being", "does", "doesn", "for",
    "from", "have", "into", "that", "the", "their", "then", "there", "these",
    "they", "this", "those", "through", "what", "when", "where", "which", "while",
    "with", "would", "could", "should", "your", "data", "information", "year",
}


def _vector_store_backend() -> str:
    backend = os.getenv("VECTOR_STORE", "qdrant").strip().lower()
    if backend not in {"qdrant", "faiss"}:
        raise ValueError("VECTOR_STORE must be either 'qdrant' or 'faiss'.")
    return backend


def _metadata_identity(metadata: Dict[str, Any]):
    return (
        metadata.get("source", "unknown"),
        metadata.get("page", 0),
        metadata.get("chunk_index"),
    )


# ── Build / persist ──────────────────────────────────────────────────────────

def build_index(chunks: List[str], metadatas: List[Dict[str, Any]]) -> None:
    """
    Embed chunks, build FAISS + BM25 indexes, save to disk.
    metadatas: list of dicts like {"source": "file.pdf", "page": 3}
    """
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building index for {len(chunks)} chunks …")
    vectors = np.array(embed_texts(chunks), dtype="float32")

    # Store the same BGE-M3 vectors in Qdrant alongside the FAISS index.
    qdrant_upsert_chunks(chunks, vectors.tolist(), metadatas)

    # FAISS flat L2 index (exact, no approximation — fine up to ~100k chunks)
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_PATH))

    # BM25 sparse index
    tokenised = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenised)

    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    logger.success(f"Index saved → {STORE_DIR}")


# ── Load ─────────────────────────────────────────────────────────────────────

def _load_index(include_faiss: bool = True):
    if include_faiss and not FAISS_PATH.exists():
        raise FileNotFoundError("No FAISS index found. Run ingestion first.")
    index = faiss.read_index(str(FAISS_PATH)) if include_faiss else None
    with open(META_PATH, "rb") as f:
        store = pickle.load(f)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    return index, store["chunks"], store["metadatas"], bm25


def query_matches_indexed_corpus(query: str) -> bool:
    """Return whether a query has multiple informative terms in indexed chunks.

    This is a routing guard used only when the LLM suggests ``llm_only``. It
    does not perform retrieval or change the retrieval ranking/parameters.
    """
    query_terms = {
        term for term in re.findall(r"[a-z0-9]+", query.casefold())
        if len(term) >= 4 and term not in _ROUTING_STOP_WORDS and not term.isdigit()
    }
    if len(query_terms) < 2:
        return False

    try:
        _, chunks, _, _ = _load_index(include_faiss=False)
    except FileNotFoundError:
        # Keep general LLM-only questions usable before local indexing exists.
        return False

    corpus_terms = {
        term
        for chunk in chunks
        for term in re.findall(r"[a-z0-9]+", str(chunk).casefold())
    }
    return len(query_terms & corpus_terms) >= 2


# ── Hybrid search ────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = TOP_K_FINAL,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    1. Dense FAISS search     → ranked list A
    2. Sparse BM25 search     → ranked list B
    3. Reciprocal Rank Fusion → merged list
    4. Cross-encoder rerank   → final top_k
    Returns list of {"text": ..., "source": ..., "page": ..., "score": ...}
    """
    backend = _vector_store_backend()
    index, chunks, metadatas, bm25 = _load_index(include_faiss=backend == "faiss")

    # ── Dense retrieval ──────────────────────────────────────
    q_vec = np.array([embed_query(query)], dtype="float32")
    dense_hit_scores: List[Optional[float]] = []
    if backend == "faiss":
        distances, idxs = index.search(q_vec, TOP_K_FETCH)
        dense_ids = idxs[0].tolist()
        dense_hit_scores = [float(score) for score in distances[0].tolist()]
    else:
        dense_hits = qdrant_search_chunks(q_vec[0].tolist(), TOP_K_FETCH)
        corpus_ids = {
            _metadata_identity(metadata): corpus_index
            for corpus_index, metadata in enumerate(metadatas)
        }
        content_ids = {
            (
                metadata.get("source", "unknown"),
                metadata.get("page", 0),
                chunks[corpus_index],
            ): corpus_index
            for corpus_index, metadata in enumerate(metadatas)
        }
        dense_ids = []
        dense_hit_scores = []
        for hit in dense_hits:
            identity = (hit["source"], hit["page"], hit["chunk_index"])
            local_id = corpus_ids.get(identity)
            if local_id is None:
                # Also support older points whose payload predates chunk_index.
                local_id = content_ids.get((hit["source"], hit["page"], hit["text"]))
            # Ignore stale Qdrant points that are no longer in the local BM25 corpus.
            if local_id is not None:
                dense_ids.append(local_id)
                dense_hit_scores.append(
                    float(hit["score"]) if hit.get("score") is not None else None
                )

    def stage_candidate(
        corpus_index: int, rank: int, score: Optional[float], score_type: str
    ):
        return {
            "chunk_id": corpus_index,
            "source": str(metadatas[corpus_index].get("source", "unknown")),
            "page": metadatas[corpus_index].get("page"),
            "rank": rank,
            "score": float(score) if score is not None else None,
            "score_type": score_type,
            "text_preview": str(chunks[corpus_index])[:300],
        }

    if diagnostics is not None:
        dense_score_type = "l2_distance" if backend == "faiss" else "cosine_similarity"
        diagnostics["dense_candidates"] = [
            stage_candidate(doc_id, rank, score, dense_score_type)
            for rank, (doc_id, score) in enumerate(zip(dense_ids, dense_hit_scores), start=1)
            if 0 <= doc_id < len(chunks)
        ]

    # ── Sparse BM25 retrieval ────────────────────────────────
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_ids  = np.argsort(bm25_scores)[::-1][:TOP_K_FETCH].tolist()
    if diagnostics is not None:
        diagnostics["bm25_candidates"] = [
            stage_candidate(doc_id, rank, float(bm25_scores[doc_id]), "bm25_score")
            for rank, doc_id in enumerate(sparse_ids, start=1)
            if 0 <= doc_id < len(chunks)
        ]

    # ── RRF fusion ───────────────────────────────────────────
    rrf: Dict[int, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

    fused_ids = sorted(rrf, key=rrf.get, reverse=True)[:TOP_K_FETCH]

    # ── Collect candidate chunks ─────────────────────────────
    candidate_chunks = [chunks[i]     for i in fused_ids]
    candidate_metas  = [metadatas[i]  for i in fused_ids]
    if diagnostics is not None:
        diagnostics["rrf_candidates"] = [
            {
                "chunk_id": corpus_index,
                "source": str(metadatas[corpus_index].get("source", "unknown")),
                "page": metadatas[corpus_index].get("page"),
                "rrf_rank": rank,
                "rrf_score": float(rrf[corpus_index]),
                "text_preview": str(chunks[corpus_index])[:300],
            }
            for rank, corpus_index in enumerate(fused_ids, start=1)
        ]

    # ── Cross-encoder rerank ─────────────────────────────────
    reranker_diagnostics: Dict[str, Any] = {}
    if diagnostics is not None:
        reranked_texts = rerank(
            query, candidate_chunks, top_n=top_k, diagnostics=reranker_diagnostics
        )
        diagnostics["reranker_candidates"] = [
            {
                **stage_candidate(
                    fused_ids[item["candidate_index"]],
                    item["reranker_rank"],
                    item["reranker_score"],
                    "cross_encoder_score",
                ),
                "rrf_rank": item["candidate_index"] + 1,
                "reranker_rank": item["reranker_rank"],
                "reranker_score": item["reranker_score"],
                "selected": item["selected"],
            }
            for item in reranker_diagnostics["reranker_candidates"]
        ]
        reranker_scores_by_candidate = {
            item["candidate_index"]: item["reranker_score"]
            for item in reranker_diagnostics["reranker_candidates"]
        }
    else:
        reranked_texts = rerank(query, candidate_chunks, top_n=top_k)
        reranker_scores_by_candidate = {}

    results = []
    for reranker_rank, text in enumerate(reranked_texts, start=1):
        idx  = candidate_chunks.index(text)
        corpus_index = fused_ids[idx]
        meta = candidate_metas[idx]
        results.append({
            "text":   text,
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page", 0),
            "chunk_id": corpus_index,
            "rrf_rank": idx + 1,
            "reranker_rank": reranker_rank,
            "rrf_score": float(rrf[corpus_index]),
            **(
                {"reranker_score": float(reranker_scores_by_candidate[idx])}
                if idx in reranker_scores_by_candidate else {}
            ),
        })

    if diagnostics is not None:
        diagnostics["reranker_score_available"] = True

    logger.debug(f"Hybrid search returned {len(results)} chunks for: '{query[:60]}'")
    return results
