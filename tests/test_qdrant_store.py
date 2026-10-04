import unittest
from unittest.mock import Mock, patch

from qdrant_client import models

from vectorstore import qdrant_client as qdrant_config
from vectorstore.qdrant_store import (
    check_qdrant_connection,
    ensure_collection,
    upsert_chunks,
)


class QdrantStoreTests(unittest.TestCase):
    def test_client_uses_environment_configuration(self):
        with patch.object(
            qdrant_config, "QDRANT_URL", "https://cluster.cloud.qdrant.io"
        ), patch.object(
            qdrant_config, "QDRANT_API_KEY", "test-api-key"
        ), patch("vectorstore.qdrant_client.QdrantClient") as client_type:
            qdrant_config.get_qdrant_client.cache_clear()
            qdrant_config.get_qdrant_client()
            client_type.assert_called_once_with(
                url="https://cluster.cloud.qdrant.io", api_key="test-api-key"
            )
            qdrant_config.get_qdrant_client.cache_clear()

    def test_client_supports_cloud_url_without_api_key(self):
        with patch.object(
            qdrant_config, "QDRANT_URL", "https://cluster.cloud.qdrant.io"
        ), patch.object(qdrant_config, "QDRANT_API_KEY", None), patch(
            "vectorstore.qdrant_client.QdrantClient"
        ) as client_type:
            qdrant_config.get_qdrant_client.cache_clear()
            qdrant_config.get_qdrant_client()
            client_type.assert_called_once_with(
                url="https://cluster.cloud.qdrant.io", api_key=None
            )
            qdrant_config.get_qdrant_client.cache_clear()

    def test_creates_cosine_collection_with_embedding_dimension(self):
        client = Mock()
        client.collection_exists.return_value = False

        ensure_collection(1024, client=client, collection_name="docs")

        client.create_collection.assert_called_once()
        kwargs = client.create_collection.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], "docs")
        self.assertEqual(kwargs["vectors_config"].size, 1024)
        self.assertEqual(kwargs["vectors_config"].distance, models.Distance.COSINE)

    def test_existing_collection_dimension_must_match(self):
        client = Mock()
        client.collection_exists.return_value = True
        client.get_collection.return_value.config.params.vectors.size = 768

        with self.assertRaisesRegex(ValueError, "has vector size 768"):
            ensure_collection(1024, client=client, collection_name="docs")

    def test_existing_compatible_collection_is_reused(self):
        client = Mock()
        client.collection_exists.return_value = True
        vectors = client.get_collection.return_value.config.params.vectors
        vectors.size = 1024
        vectors.distance = models.Distance.COSINE

        ensure_collection(1024, client=client, collection_name="docs")

        client.create_collection.assert_not_called()

    def test_existing_collection_distance_must_be_cosine(self):
        client = Mock()
        client.collection_exists.return_value = True
        vectors = client.get_collection.return_value.config.params.vectors
        vectors.size = 1024
        vectors.distance = models.Distance.DOT

        with self.assertRaisesRegex(ValueError, "expected COSINE"):
            ensure_collection(1024, client=client, collection_name="docs")

    def test_read_only_connection_check_reports_collection_configuration(self):
        client = Mock()
        vectors = client.get_collection.return_value.config.params.vectors
        vectors.size = 1024
        vectors.distance = models.Distance.COSINE
        client.get_collection.return_value.points_count = 17

        result = check_qdrant_connection(client=client, collection_name="docs")

        client.get_collection.assert_called_once_with("docs")
        self.assertEqual(
            result,
            {
                "status": "ok",
                "collection_name": "docs",
                "vector_size": 1024,
                "distance": "COSINE",
                "points_count": 17,
            },
        )

    def test_read_only_connection_check_wraps_connectivity_auth_failure(self):
        client = Mock()
        client.get_collection.side_effect = RuntimeError("unauthorized")

        with self.assertRaisesRegex(RuntimeError, "check QDRANT_URL, QDRANT_API_KEY") as err:
            check_qdrant_connection(client=client, collection_name="docs")

        self.assertNotIn("unauthorized", str(err.exception))

    def test_batch_upsert_contains_text_and_existing_metadata(self):
        client = Mock()
        client.collection_exists.return_value = False
        chunks = ["first chunk", "second chunk", "third chunk"]
        vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        metadata = [
            {"source": "guide.pdf", "page": 1, "chunk_index": index}
            for index in range(3)
        ]

        count = upsert_chunks(
            chunks, vectors, metadata, client=client, collection_name="docs", batch_size=2
        )

        self.assertEqual(count, 3)
        calls = client.upsert.call_args_list
        self.assertEqual([len(call.kwargs["points"]) for call in calls], [2, 1])
        first_point = calls[0].kwargs["points"][0]
        self.assertEqual(first_point.vector, vectors[0])
        self.assertEqual(
            first_point.payload,
            {"text": chunks[0], "source": "guide.pdf", "page": 1, "chunk_index": 0},
        )
        upsert_chunks(
            chunks[:1], vectors[:1], metadata[:1], client=client, collection_name="docs"
        )
        self.assertEqual(client.upsert.call_args.kwargs["points"][0].id, first_point.id)


if __name__ == "__main__":
    unittest.main()
