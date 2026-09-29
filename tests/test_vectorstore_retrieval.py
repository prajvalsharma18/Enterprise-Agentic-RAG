import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from types import SimpleNamespace

from vectorstore import store
from vectorstore.qdrant_store import search_chunks


class QdrantDenseSearchTests(unittest.TestCase):
    def test_query_embedding_qdrant_search_and_hit_conversion(self):
        client = Mock()
        client.query_points.return_value.points = [
            SimpleNamespace(
                id="point-1",
                score=0.91,
                payload={
                    "text": "matching chunk",
                    "source": "manual.pdf",
                    "page": 3,
                    "chunk_index": 1,
                },
            )
        ]

        hits = search_chunks([0.1, 0.2], 20, client=client, collection_name="docs")

        client.query_points.assert_called_once_with(
            collection_name="docs", query=[0.1, 0.2], limit=20, with_payload=True
        )
        self.assertEqual(
            hits,
            [{
                "text": "matching chunk",
                "source": "manual.pdf",
                "page": 3,
                "chunk_index": 1,
                "score": 0.91,
            }],
        )

    def test_qdrant_connection_errors_are_explicit(self):
        client = Mock()
        client.query_points.side_effect = ConnectionError("offline")

        with self.assertRaisesRegex(RuntimeError, "VECTOR_STORE=qdrant.*QDRANT_URL"):
            search_chunks([0.1], 5, client=client)


class HybridBackendTests(unittest.TestCase):
    def setUp(self):
        self.chunks = ["alpha chunk", "beta chunk", "gamma chunk"]
        self.metadata = [
            {"source": "doc.pdf", "page": 1, "chunk_index": i}
            for i in range(3)
        ]
        self.index = Mock()
        self.bm25 = Mock()
        self.bm25.get_scores.return_value = np.array([3.0, 1.0, 0.0])
        self.embed = patch("vectorstore.store.embed_query", return_value=[0.1, 0.2]).start()
        self.load = patch(
            "vectorstore.store._load_index",
            return_value=(self.index, self.chunks, self.metadata, self.bm25),
        ).start()
        self.rerank = patch(
            "vectorstore.store.rerank", side_effect=lambda query, chunks, top_n: chunks[:top_n]
        ).start()
        self.addCleanup(patch.stopall)

    def test_qdrant_and_bm25_candidates_enter_existing_reranker(self):
        self.index.search.return_value = (np.array([[0.0]]), np.array([[2]]))
        qdrant_search = patch(
            "vectorstore.store.qdrant_search_chunks",
            return_value=[
                {"text": "gamma chunk", "source": "doc.pdf", "page": 1, "chunk_index": 2}
            ],
        ).start()
        self.addCleanup(patch.stopall)
        with patch.dict(os.environ, {"VECTOR_STORE": "qdrant"}):
            results = store.hybrid_search("question")

        self.embed.assert_called_once_with("question")
        qdrant_search.assert_called_once_with([0.1, 0.2], store.TOP_K_FETCH)
        self.index.search.assert_not_called()
        self.bm25.get_scores.assert_called_once_with(["question"])
        candidate_texts = self.rerank.call_args.args[1]
        self.assertIn("gamma chunk", candidate_texts)
        self.assertIn("alpha chunk", candidate_texts)
        self.load.assert_called_once_with(include_faiss=False)
        self.assertTrue(results)

    def test_explicit_faiss_backend_uses_faiss_dense_search(self):
        self.index.search.return_value = (
            np.array([[0.0, 1.0]]), np.array([[1, 0]])
        )
        qdrant_search = patch("vectorstore.store.qdrant_search_chunks").start()
        self.addCleanup(patch.stopall)
        with patch.dict(os.environ, {"VECTOR_STORE": "faiss"}):
            results = store.hybrid_search("question")

        self.index.search.assert_called_once()
        qdrant_search.assert_not_called()
        self.load.assert_called_once_with(include_faiss=True)
        self.assertTrue(results)

    def test_default_backend_is_qdrant(self):
        self.index.search.return_value = (np.array([[0.0]]), np.array([[0]]))
        qdrant_search = patch(
            "vectorstore.store.qdrant_search_chunks", return_value=[]
        ).start()
        self.addCleanup(patch.stopall)
        with patch.dict(os.environ, {}, clear=True):
            store.hybrid_search("question")
        qdrant_search.assert_called_once()
        self.index.search.assert_not_called()


class CorrectiveRetrievalTests(unittest.TestCase):
    def test_rewritten_query_uses_selected_qdrant_backend(self):
        from app import agent

        chunks = ["rewritten-query result"]
        metadatas = [{"source": "doc.pdf", "page": 1, "chunk_index": 0}]
        bm25 = Mock()
        bm25.get_scores.return_value = np.array([1.0])
        with patch.dict(os.environ, {"VECTOR_STORE": "qdrant"}), patch(
            "vectorstore.store._load_index", return_value=(None, chunks, metadatas, bm25)
        ), patch("vectorstore.store.embed_query", return_value=[0.4, 0.5]), patch(
            "vectorstore.store.qdrant_search_chunks",
            return_value=[{
                "text": chunks[0], "source": "doc.pdf", "page": 1, "chunk_index": 0
            }],
        ) as qdrant_search, patch(
            "vectorstore.store.rerank", side_effect=lambda query, candidates, top_n: candidates
        ):
            state = {"query": "rewritten query", "context": []}
            result = agent.retrieve(state)

        qdrant_search.assert_called_once_with([0.4, 0.5], store.TOP_K_FETCH)
        self.assertEqual(result["context"][0]["text"], chunks[0])


if __name__ == "__main__":
    unittest.main()
