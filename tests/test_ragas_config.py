import unittest
from unittest.mock import patch

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from app.llm_provider import create_chat_model
from evaluation.ragas_config import (
    BgeM3EvaluationEmbeddings,
    build_ragas_llm,
)
from evaluation.evaluate import _aggregate_metric_scores


class RagasConfigurationTests(unittest.TestCase):
    def test_ragas_uses_configured_chat_completions_client(self):
        evaluator = build_ragas_llm({
            "OPENAI_API_KEY": "test-evaluator-key",
            "OPENAI_MODEL": "gpt-evaluator-test",
        })

        self.assertIsInstance(evaluator, LangchainLLMWrapper)
        self.assertIsInstance(evaluator.langchain_llm, ChatOpenAI)
        self.assertEqual(evaluator.langchain_llm.model_name, "gpt-evaluator-test")
        self.assertFalse(evaluator.langchain_llm.use_responses_api)
        self.assertIsNone(evaluator.langchain_llm.temperature)
        self.assertEqual(evaluator.get_temperature(n=1), 1.0)
        self.assertNotEqual(evaluator.get_temperature(n=1), 1e-8)

    def test_application_openai_client_keeps_responses_api_enabled(self):
        application_model = create_chat_model({
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-application-key",
            "OPENAI_MODEL": "gpt-application-test",
        })

        self.assertTrue(application_model.use_responses_api)
        self.assertEqual(application_model.model_name, "gpt-application-test")

    def test_ragas_embeddings_delegate_to_existing_bge_m3_helpers(self):
        embeddings = BgeM3EvaluationEmbeddings()
        with patch(
            "evaluation.ragas_config.embed_texts", return_value=[[0.1, 0.2]]
        ) as embed_texts, patch(
            "evaluation.ragas_config.embed_bge_query", return_value=[0.3, 0.4]
        ) as embed_query:
            self.assertEqual(embeddings.embed_documents(["chunk"]), [[0.1, 0.2]])
            self.assertEqual(embeddings.embed_query("query"), [0.3, 0.4])

        embed_texts.assert_called_once_with(["chunk"])
        embed_query.assert_called_once_with("query")

    def test_non_finite_or_missing_metric_values_fail_instead_of_becoming_zero(self):
        with self.assertRaisesRegex(RuntimeError, "No benchmark score was produced"):
            _aggregate_metric_scores(
                [{
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None,
                }],
                expected_question_count=1,
            )

    def test_real_numeric_zero_is_not_misclassified_as_a_failed_metric(self):
        scores = _aggregate_metric_scores(
            [{
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }],
            expected_question_count=1,
        )

        self.assertEqual(set(scores.values()), {0.0})


if __name__ == "__main__":
    unittest.main()
