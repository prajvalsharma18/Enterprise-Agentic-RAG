import unittest
from unittest.mock import Mock, patch

from qdrant_client import models

from vectorstore import qdrant_client as qdrant_config
from vectorstore.qdrant_store import ensure_collection, upsert_chunks


class QdrantStoreTests(unittest.TestCase):
    def test_client_uses_environment_configuration(self):
        with patch.object(qdrant_config, "QDRANT_URL", "http://qdrant:6333"), patch.object(
            qdrant_config, "QDRANT_API_KEY", "secret"
        ), patch("vectorstore.qdrant_client.QdrantClient") as client_type:
            qdrant_config.get_qdrant_client.cache_clear()
            qdrant_config.get_qdrant_client()
            client_type.assert_called_once_with(url="http://qdrant:6333", api_key="secret")
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
