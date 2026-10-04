import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage


class AgentEvidenceDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            from app import agent
        cls.agent = agent

    def _state(self, **updates):
        state = {
            "messages": [HumanMessage(content="How does the supplied technical report explain this? ")],
            "query": "How does the supplied technical report explain this?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
            "diagnostics": {},
        }
        state.update(updates)
        return state

    def test_corpus_question_routing_prompt_directs_specific_facts_to_documents(self):
        corpus_questions = (
            "How does encryption in transit protect a communication if it is intercepted?",
            "How does Confidential Computing protect data while it is being processed?",
        )
        for question in corpus_questions:
            state = self._state(
                query=question, messages=[HumanMessage(content=question)]
            )
            model = Mock()
            model.invoke.return_value = SimpleNamespace(content="documents")
            with patch.object(self.agent, "llm", model):
                routed = self.agent.route_query(state)

            prompt = model.invoke.call_args.args[0][0].content
            self.assertIn("specific factual, technical, or business questions", prompt)
            self.assertIn("even if the user does not explicitly mention a document", prompt)
            self.assertEqual(routed["_route"], "documents")
            self.assertEqual(routed["diagnostics"]["initial_route"], "documents")
            self.assertEqual(routed["diagnostics"]["initial_query"], question)

    def test_retrieval_and_grader_record_compact_candidate_decisions(self):
        chunk = {"text": "evidence text", "source": "source.pdf", "page": 4}
        def fake_search(query, top_k, diagnostics):
            diagnostics["rrf_candidates"] = [{
                "chunk_id": 7, "source": "source.pdf", "page": 4,
                "rrf_rank": 1, "rrf_score": 0.03,
                "text_preview": "evidence text",
            }]
            diagnostics["reranker_score_available"] = False
            return [{**chunk, "chunk_id": 7, "rrf_rank": 1, "rrf_score": 0.03}]

        with patch.object(self.agent, "hybrid_search", side_effect=fake_search):
            retrieved = self.agent.retrieve(self._state())
        with patch.object(self.agent, "llm") as model:
            model.invoke.return_value = SimpleNamespace(content="no")
            graded = self.agent.grade_documents(retrieved)

        candidate = graded["diagnostics"]["retrieval_attempts"][0]["reranked_candidates"][0]
        retrieval_candidate = graded["diagnostics"]["retrieval_attempts"][0]["retrieval_candidates"][0]
        self.assertEqual(candidate["relevance_decision"], "rejected")
        self.assertEqual(retrieval_candidate["rrf_score"], 0.03)
        self.assertEqual(candidate["source"], "source.pdf")
        self.assertEqual(candidate["reranker_rank"], 1)
        self.assertNotIn("vector", candidate)
        self.assertEqual(graded["diagnostics"]["final_document_context"], [])

    def test_web_fallback_evidence_supersedes_nonempty_document_context(self):
        docs = [{"text": "stale document evidence", "source": "local.pdf", "page": 1}]
        # web_search normalizes Tavily's `content` field to this internal schema.
        web = [{"title": "Public page", "url": "https://example.test/a", "snippet": "web evidence"}]
        state = self._state(context=docs, web_search_invoked=True, web_results=web)
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content="answer from web")
        with patch.object(self.agent, "llm", model):
            generated = self.agent.generate(state)

        prompt = model.invoke.call_args.args[0][0].content
        self.assertIn("web evidence", prompt)
        self.assertNotIn("stale document evidence", prompt)
        self.assertEqual(generated["generation_evidence_source"], "web")
        self.assertIn("web evidence", generated["generation_context"][0])

    def test_document_evidence_remains_selected_without_fallback(self):
        state = self._state(context=[{
            "text": "document evidence", "source": "local.pdf", "page": 2
        }])
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content="answer from docs")
        with patch.object(self.agent, "llm", model):
            generated = self.agent.generate(state)

        self.assertIn("document evidence", model.invoke.call_args.args[0][0].content)
        self.assertEqual(generated["generation_evidence_source"], "documents")

    def test_empty_web_fallback_returns_insufficient_evidence_without_llm(self):
        model = Mock()
        state = self._state(context=[{"text": "stale", "source": "x", "page": 1}],
                            web_search_invoked=True, web_results=[])
        with patch.object(self.agent, "llm", model):
            generated = self.agent.generate(state)

        model.invoke.assert_not_called()
        self.assertEqual(generated["generation_evidence_source"], "insufficient_evidence")
        self.assertIn("could not find sufficient", generated["answer"])

    def test_web_diagnostics_are_bounded_and_remove_url_query_credentials(self):
        state = self._state()
        mock_tool = Mock()
        mock_tool.invoke.return_value = [{
            "title": "Title", "url": "https://example.test/result?api_key=secret",
            "content": "snippet",
        }]
        with patch.object(self.agent, "web_search_tool", mock_tool):
            result = self.agent.web_search(state)

        diagnostic = result["diagnostics"]["web_results"][0]
        self.assertEqual(diagnostic["url"], "https://example.test/result")
        self.assertNotIn("secret", str(result["diagnostics"]))
        self.assertTrue(result["diagnostics"]["web_fallback_invoked"])

    def test_tavily_exception_becomes_empty_fallback_for_transparent_generation(self):
        mock_tool = Mock()
        mock_tool.invoke.side_effect = RuntimeError("private detail")
        with patch.object(self.agent, "web_search_tool", mock_tool):
            searched = self.agent.web_search(self._state())
        model = Mock()
        with patch.object(self.agent, "llm", model):
            generated = self.agent.generate(searched)

        self.assertEqual(searched["diagnostics"]["web_fallback_error"], "RuntimeError")
        self.assertNotIn("private detail", str(searched["diagnostics"]))
        self.assertEqual(generated["generation_evidence_source"], "insufficient_evidence")
        model.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
