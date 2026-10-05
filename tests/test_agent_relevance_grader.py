import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage


class AgentRelevanceGraderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            from app import agent
        cls.agent = agent

    def test_partial_answer_evidence_is_relevant_but_topical_text_is_not(self):
        query = "Which non-TCP example and key property does the protocol specify?"
        partial_evidence = {
            "text": "It supports non-TCP protocols such as UDP and uses a unique key per connection.",
            "source": "technical.pdf",
            "page": 8,
        }
        topical_only = {
            "text": "The document describes encryption technologies for network traffic.",
            "source": "technical.pdf",
            "page": 1,
        }
        state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "rewrite_count": 0,
            "context": [partial_evidence, topical_only],
            "web_results": [],
            "answer": "",
            "diagnostics": {
                "retrieval_attempts": [{
                    "reranked_candidates": [
                        {"relevance_decision": "pending"},
                        {"relevance_decision": "pending"},
                    ]
                }]
            },
        }
        model = Mock()
        model.invoke.side_effect = [
            SimpleNamespace(content="yes"),
            SimpleNamespace(content="no"),
        ]

        with patch.object(self.agent, "llm", model):
            graded = self.agent.grade_documents(state)

        prompts = [call.args[0][0].content for call in model.invoke.call_args_list]
        self.assertTrue(all("partial evidence" in prompt for prompt in prompts))
        self.assertTrue(all("only topically related" in prompt for prompt in prompts))
        self.assertEqual(graded["_grade"], "generate")
        self.assertEqual(graded["context"], [partial_evidence])
        decisions = graded["diagnostics"]["retrieval_attempts"][0]["reranked_candidates"]
        self.assertEqual([item["relevance_decision"] for item in decisions], ["accepted", "rejected"])


if __name__ == "__main__":
    unittest.main()
