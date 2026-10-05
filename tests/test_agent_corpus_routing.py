import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage


class CorpusRoutingGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            from app import agent
        cls.agent = agent

    def route_with_corpus(self, query, chunks):
        model = Mock()
        # Exercise the guard even if the initial LLM classification is wrong.
        model.invoke.return_value = SimpleNamespace(content="llm_only")
        state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }
        fake_index_data = (None, chunks, [{} for _ in chunks], None)
        with patch.object(self.agent, "llm", model), patch(
            "vectorstore.store._load_index", return_value=fake_index_data
        ):
            return self.agent.route_query(state)

    def test_specific_encryption_in_transit_question_uses_documents(self):
        query = "How does encryption in transit protect a communication if it is intercepted?"
        result = self.route_with_corpus(
            query,
            ["Encryption in transit protects communication and verifies integrity."],
        )

        self.assertEqual(result["_route"], "documents")
        self.assertTrue(result["diagnostics"]["corpus_match_override"])

    def test_confidential_computing_terminology_question_uses_documents(self):
        query = "How does Confidential Computing protect data while processed, and what does attestation establish?"
        result = self.route_with_corpus(
            query,
            ["Confidential Computing uses a TEE. Attestation verifies system state and code."],
        )

        self.assertEqual(result["_route"], "documents")

    def test_genuinely_general_question_remains_llm_only(self):
        result = self.route_with_corpus(
            "How do I boil an egg?",
            ["Google Cloud protects data with encryption in transit."],
        )

        self.assertEqual(result["_route"], "llm_only")
        self.assertFalse(result["diagnostics"]["corpus_match_override"])

    def test_unsupported_question_about_corpus_subject_uses_documents(self):
        query = "What total revenue did Google report for fiscal year 2025?"
        result = self.route_with_corpus(
            query,
            ["Google Cloud security overview.",
             "Microsoft fiscal year 2025 total revenue was reported in the annual report."],
        )

        self.assertEqual(result["_route"], "documents")
        self.assertTrue(result["diagnostics"]["corpus_match_override"])


if __name__ == "__main__":
    unittest.main()
