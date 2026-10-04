"""Qdrant collection and batch storage for existing BGE-M3 document vectors."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient, models

from vectorstore.qdrant_client import QDRANT_COLLECTION_NAME, get_qdrant_client


UPSERT_BATCH_SIZE = 64
EXPECTED_VECTOR_SIZE = 1024


def _validate_collection(collection_info: Any, expected_vector_size: int) -> None:
    vectors = collection_info.config.params.vectors
    if isinstance(vectors, dict):
        raise ValueError("Qdrant collection uses named vectors; expected one unnamed vector.")
    if vectors.size != expected_vector_size:
        raise ValueError(
            f"Qdrant collection has vector size {vectors.size}; "
            f"expected {expected_vector_size}."
        )
    if vectors.distance != models.Distance.COSINE:
        raise ValueError(
            f"Qdrant collection uses distance {vectors.distance}; expected COSINE."
        )


def ensure_collection(
    vector_size: int,
    client: Optional[QdrantClient] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> None:
    """Create the cosine collection when missing; reject incompatible dimensions."""
    qdrant = client or get_qdrant_client()
    if vector_size <= 0:
        raise ValueError("Vector size must be positive.")

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        return

    _validate_collection(qdrant.get_collection(collection_name), vector_size)


def check_qdrant_connection(
    client: Optional[QdrantClient] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    expected_vector_size: int = EXPECTED_VECTOR_SIZE,
) -> Dict[str, Any]:
    """Read-only connectivity, collection access, and vector configuration check."""
    qdrant = client or get_qdrant_client()
    try:
        collection_info = qdrant.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to access Qdrant collection {collection_name!r}; "
            "check QDRANT_URL, QDRANT_API_KEY, and collection existence."
        ) from exc

    _validate_collection(collection_info, expected_vector_size)
    return {
        "status": "ok",
        "collection_name": collection_name,
        "vector_size": expected_vector_size,
        "distance": "COSINE",
        "points_count": collection_info.points_count,
    }


def _point_id(chunk: str, metadata: Dict[str, Any], position: int) -> str:
    """Generate a stable UUID from the existing source/page/chunk identity."""
    identity = {
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page", 0),
        "chunk_index": metadata.get("chunk_index", position),
        "text_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
    }
    return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(identity, sort_keys=True)))


def upsert_chunks(
    chunks: Sequence[str],
    vectors: Sequence[Sequence[float]],
    metadatas: Sequence[Dict[str, Any]],
    client: Optional[QdrantClient] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> int:
    """Store precomputed BGE-M3 vectors and chunk payloads in batches."""
    if not (len(chunks) == len(vectors) == len(metadatas)):
        raise ValueError("chunks, vectors, and metadatas must have matching lengths.")
    if not chunks:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("All embedding vectors must have the same positive dimension.")
    vector_size = dimensions.pop()
    qdrant = client or get_qdrant_client()
    ensure_collection(vector_size, client=qdrant, collection_name=collection_name)

    points = [
        models.PointStruct(
            id=_point_id(chunk, metadata, index),
            vector=list(vector),
            payload={"text": chunk, **metadata},
        )
        for index, (chunk, vector, metadata) in enumerate(zip(chunks, vectors, metadatas))
    ]
    for start in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=collection_name,
            points=points[start : start + batch_size],
            wait=True,
        )
    return len(points)


def search_chunks(
    query_vector: Sequence[float],
    top_k: int,
    client: Optional[QdrantClient] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """Search Qdrant and convert hits to the app's plain chunk dictionary format."""
    if not query_vector:
        raise ValueError("Query embedding must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    try:
        qdrant = client or get_qdrant_client()
        response = qdrant.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            limit=top_k,
            with_payload=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"VECTOR_STORE=qdrant but Qdrant retrieval failed for collection "
            f"{collection_name!r}. Check QDRANT_URL and ensure ingestion created "
            "the collection and the service is reachable."
        ) from exc

    results: List[Dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        if "text" not in payload:
            raise RuntimeError(
                f"Qdrant point {point.id!r} in {collection_name!r} has no text payload. "
                "Re-ingest documents using the current Qdrant storage layer."
            )
        results.append(
            {
                "text": payload["text"],
                "source": payload.get("source", "unknown"),
                "page": payload.get("page", 0),
                "chunk_index": payload.get("chunk_index"),
                "score": point.score,
            }
        )
    return results
