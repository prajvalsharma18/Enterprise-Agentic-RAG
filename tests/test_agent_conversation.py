import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage


class ConversationalQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            from app import agent
        cls.agent = agent

    def contextualize(self, current, history, response):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content=response)
        with patch.object(self.agent, "llm", model):
            query = self.agent._contextualize_query(current, history)
        return query, model

    def test_copilot_follow_up_keeps_microsoft_revenue_context(self):
        query, model = self.contextualize(
            "What about Copilot?",
            [HumanMessage(content="What are Microsoft's major revenue growth drivers?")],
            '{"uses_context": true, "query": "How does Microsoft Copilot contribute to Microsoft revenue growth?"}',
        )

        self.assertIn("Microsoft", query)
        self.assertIn("revenue growth", query)
        self.assertIn("Copilot", query)
        self.assertEqual(model.invoke.call_count, 1)

    def test_pronoun_follow_up_resolves_from_conversation(self):
        query, _ = self.contextualize(
            "How does it affect customers?",
            [HumanMessage(content="Explain Google's shared responsibility model.")],
            '{"uses_context": true, "query": "How does Google Cloud shared responsibility model affect customers?"}',
        )

        self.assertIn("Google", query)
        self.assertIn("shared responsibility", query)
        self.assertIn("customers", query)

    def test_standalone_question_is_preserved(self):
        question = "What is Google Cloud encryption at rest?"
        query, _ = self.contextualize(
            question,
            [HumanMessage(content="What are Microsoft's revenue drivers?")],
            '{"uses_context": false, "query": "An unrelated rewritten question"}',
        )

        self.assertEqual(query, question)

    def test_empty_history_preserves_single_turn_behavior_without_llm_call(self):
        model = Mock()
        with patch.object(self.agent, "llm", model):
            query = self.agent._contextualize_query("What is Azure?", [])

        self.assertEqual(query, "What is Azure?")
        model.invoke.assert_not_called()

    def test_new_chat_empty_history_does_not_reuse_previous_subject(self):
        model = Mock()
        with patch.object(self.agent, "llm", model):
            query = self.agent._contextualize_query("What about Copilot?", [])

        self.assertEqual(query, "What about Copilot?")
        model.invoke.assert_not_called()

    def test_contextualization_node_preserves_original_user_query_for_generation(self):
        state = {
            "messages": [
                HumanMessage(content="What are Microsoft's revenue growth drivers?"),
                AIMessage(content="Cloud and productivity products."),
                HumanMessage(content="What about Copilot?"),
            ],
            "query": "What about Copilot?",
        }
        response = SimpleNamespace(content=(
            '{"uses_context": true, "query": "How does Microsoft Copilot contribute to revenue growth?"}'
        ))
        with patch.object(self.agent, "llm", Mock(invoke=Mock(return_value=response))):
            result = self.agent.contextualize_query(state)

        self.assertIn("Microsoft", result["query"])
        self.assertEqual(result["original_query"], "What about Copilot?")
        self.assertTrue(result["diagnostics"]["query_contextualized"])


if __name__ == "__main__":
    unittest.main()
