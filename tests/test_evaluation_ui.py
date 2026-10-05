import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.frontend import (
    get_system_status,
    knowledge_base_state,
    load_evaluation_artifacts,
    parse_evaluation_artifacts,
)
from app.runtime_info import public_runtime_info


class EvaluationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.scores = {
            "faithfulness": 0.8531,
            "answer_relevancy": 0.8659,
            "context_precision": 0.8925,
            "context_recall": 0.9056,
        }
        self.results = [{
            "id": "enc-01",
            "question": "Which technologies are used?",
            "category": "terminology_hybrid",
            "difficulty": "easy",
            "answer": "TLS and BoringSSL.",
            "metrics": {metric: score for metric, score in self.scores.items()},
            "route": "documents",
            "evidence_source": "documents",
            "evaluation_status": "completed",
            "diagnostics": {"internal": "not needed by the page"},
        }]

    def test_saved_aggregate_and_question_results_are_parsed(self):
        result = parse_evaluation_artifacts(self.scores, self.results, "2026-10-05 12:00 UTC")

        self.assertTrue(result["available"])
        self.assertEqual(result["question_count"], 1)
        self.assertEqual(result["scores"], self.scores)
        self.assertEqual(result["average"], 0.8793)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["updated_at"], "2026-10-05 12:00 UTC")
        self.assertEqual(result["questions"][0]["route"], "documents")
        self.assertNotIn("diagnostics", result["questions"][0])

    def test_missing_artifacts_return_unavailable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            result = load_evaluation_artifacts(
                Path(directory) / "missing_scores.json",
                Path(directory) / "missing_results.json",
            )

        self.assertFalse(result["available"])
        self.assertIn("unavailable", result["message"].lower())

    def test_malformed_or_incomplete_artifacts_return_unavailable_state(self):
        malformed_scores = dict(self.scores)
        malformed_scores["faithfulness"] = float("nan")
        incomplete_question = dict(self.results[0], evaluation_status="pending")

        self.assertFalse(parse_evaluation_artifacts(malformed_scores, self.results)["available"])
        self.assertFalse(parse_evaluation_artifacts(self.scores, [incomplete_question])["available"])
        self.assertFalse(parse_evaluation_artifacts(self.scores, {"items": self.results})["available"])

    def test_malformed_json_file_is_handled_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            scores_path = Path(directory) / "scores.json"
            results_path = Path(directory) / "results.json"
            scores_path.write_text("{broken", encoding="utf-8")
            results_path.write_text(json.dumps(self.results), encoding="utf-8")

            result = load_evaluation_artifacts(scores_path, results_path)

        self.assertFalse(result["available"])


class RuntimeStatusTests(unittest.TestCase):
    def test_empty_knowledge_base_state_uses_existing_health_and_count(self):
        self.assertEqual(knowledge_base_state({
            "api_healthy": True, "qdrant_healthy": True, "indexed_chunks": 0,
        }), "Not indexed")
        self.assertEqual(knowledge_base_state({
            "api_healthy": True, "qdrant_healthy": True, "indexed_chunks": 8,
        }), "Ready")
        self.assertEqual(knowledge_base_state({
            "api_healthy": False, "qdrant_healthy": True, "indexed_chunks": 8,
        }), "Unavailable")

    def test_system_configuration_reports_secret_presence_without_secret_values(self):
        secret = "sk-this-value-must-not-be-returned"
        config = public_runtime_info({
            "LLM_PROVIDER": "openai",
            "OPENAI_MODEL": "configured-model",
            "OPENAI_API_KEY": secret,
            "VECTOR_STORE": "qdrant",
            "TAVILY_API_KEY": "tavily-secret",
        })

        self.assertEqual(config["llm_provider"], "OpenAI")
        self.assertEqual(config["llm_model"], "configured-model")
        self.assertEqual(config["llm_credential_status"], "Configured")
        self.assertEqual(config["vector_store"], "QDRANT")
        self.assertIn("configured", config["web_fallback"].lower())
        self.assertNotIn(secret, str(config))
        self.assertNotIn("tavily-secret", str(config))

    def test_ollama_runtime_info_does_not_require_credentials(self):
        config = public_runtime_info({
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "llama3.2-test",
        })

        self.assertEqual(config["llm_provider"], "Ollama")
        self.assertEqual(config["llm_model"], "llama3.2-test")
        self.assertEqual(config["llm_credential_status"], "Not required by this provider")

    def test_health_response_with_missing_optional_count_is_still_healthy(self):
        api_response = type("Response", (), {
            "is_success": True,
            "json": lambda self: {"status": "ok"},
        })()
        qdrant_response = type("Response", (), {
            "is_success": True,
            "json": lambda self: {"status": "ok"},
        })()
        with patch("app.frontend.httpx.get", side_effect=[api_response, qdrant_response]):
            status = get_system_status("http://api.test")

        self.assertTrue(status["api_healthy"])
        self.assertTrue(status["qdrant_healthy"])
        self.assertIsNone(status["indexed_chunks"])


if __name__ == "__main__":
    unittest.main()
