"""Qdrant connection configuration shared by Qdrant storage operations."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return the process-wide Qdrant client using environment configuration."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


client = get_qdrant_client()
