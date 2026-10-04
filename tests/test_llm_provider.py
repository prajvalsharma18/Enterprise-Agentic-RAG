import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.llm_provider import create_chat_model, extract_text_content


class LLMContentNormalizationTests(unittest.TestCase):
    def test_plain_string_content_is_returned_unchanged(self):
        self.assertEqual(extract_text_content("web"), "web")

    def test_langchain_responses_api_text_blocks_are_joined(self):
        content = [
            {"type": "text", "text": "docu"},
            {"type": "text", "text": "ments"},
        ]

        self.assertEqual(extract_text_content(content), "documents")

    def test_langchain_text_block_objects_are_supported(self):
        block = SimpleNamespace(type="text", text="answer")

        self.assertEqual(extract_text_content([block]), "answer")

    def test_non_text_response_blocks_are_not_stringified(self):
        with self.assertRaisesRegex(TypeError, "no supported text blocks"):
            extract_text_content([{"type": "image", "image_url": "unused"}])


class LLMProviderFactoryTests(unittest.TestCase):
    @patch("langchain_openai.ChatOpenAI")
    def test_openai_provider_uses_responses_api_and_configured_model(self, openai_model):
        model = create_chat_model({
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-test",
        })

        self.assertIs(model, openai_model.return_value)
        openai_model.assert_called_once_with(
            model="gpt-test",
            api_key="test-key",
            temperature=0.1,
            use_responses_api=True,
        )

    def test_openai_provider_requires_api_key(self):
        with self.assertRaisesRegex(ValueError, "requires OPENAI_API_KEY"):
            create_chat_model({"LLM_PROVIDER": "openai"})

    @patch("langchain_ollama.ChatOllama")
    def test_ollama_provider_remains_selectable(self, ollama_model):
        model = create_chat_model({
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "llama3.2",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        })

        self.assertIs(model, ollama_model.return_value)
        ollama_model.assert_called_once_with(
            model="llama3.2",
            temperature=0.1,
            base_url="http://localhost:11434",
        )

    def test_openai_is_default_and_unknown_provider_is_rejected(self):
        with patch("langchain_openai.ChatOpenAI") as openai_model:
            create_chat_model({"OPENAI_API_KEY": "test-key"})
        self.assertEqual(openai_model.call_args.kwargs["model"], "gpt-4o-mini")

        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER must be"):
            create_chat_model({"LLM_PROVIDER": "other"})


class AgentLLMProviderTests(unittest.TestCase):
    def test_route_query_accepts_structured_responses_api_content(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            from app import agent
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [HumanMessage(content="What does the report say?")],
            "query": "What does the report say?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }
        model = Mock()
        model.invoke.return_value = SimpleNamespace(
            content=[{"type": "text", "text": "documents"}]
        )

        with patch.object(agent, "llm", model):
            result = agent.route_query(state)

        self.assertEqual(result["_route"], "documents")

    def test_routing_grading_rewriting_and_generation_share_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            from app import agent
        from langchain_core.messages import HumanMessage

        model = Mock()
        model.invoke.side_effect = [
            SimpleNamespace(content="documents"),
            SimpleNamespace(content="yes"),
            SimpleNamespace(content="specific rewritten query"),
            SimpleNamespace(content="grounded answer"),
        ]
        state = {
            "messages": [HumanMessage(content="original question")],
            "query": "original question",
            "rewrite_count": 0,
            "context": [{"text": "matching passage", "source": "guide.pdf", "page": 1}],
            "web_results": [],
            "answer": "",
        }

        with patch.object(agent, "llm", model):
            routed = agent.route_query(state)
            graded = agent.grade_documents({**state, **routed})
            rewritten = agent.rewrite_query({**state, **graded, "_grade": "rewrite"})
            generated = agent.generate({**state, **rewritten})

        self.assertEqual(model.invoke.call_count, 4)
        self.assertEqual(routed["_route"], "documents")
        self.assertEqual(graded["_grade"], "generate")
        self.assertEqual(rewritten["query"], "specific rewritten query")
        self.assertEqual(generated["answer"], "grounded answer")


if __name__ == "__main__":
    unittest.main()
