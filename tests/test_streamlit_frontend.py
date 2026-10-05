import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

from app.frontend import (
    BackendError,
    document_evidence_details,
    get_system_status,
    ingest_pdf,
    parse_ingest_response,
    parse_query_response,
    query_backend,
    retrieval_attempt_counts,
    serialize_history,
)


class FrontendResponseTests(unittest.TestCase):
    def test_document_evidence_metadata_is_preserved_for_display(self):
        details = document_evidence_details({
            "source": "annual_report.pdf",
            "page": 23,
            "chunk_id": 91,
            "rrf_rank": 16,
            "reranker_rank": 4,
            "reranker_score": 0.058,
        }, 1)

        self.assertEqual(details, [
            "Evidence 1", "Page 23", "Chunk ID 91", "RRF rank 16",
            "Cross-Encoder rank 4", "Cross-Encoder score 0.0580",
        ])

    def test_missing_evidence_metadata_is_omitted(self):
        self.assertEqual(document_evidence_details({"text": "Evidence only"}, 2), ["Evidence 2"])

    def test_candidate_counts_and_relevance_use_each_attempt_diagnostics(self):
        first_attempt = {
            "dense_candidates": [{}, {}],
            "bm25_candidates": [{}],
            "retrieval_candidates": [{}, {}],
            "reranker_candidates": [{}, {}],
            "reranked_candidates": [
                {"relevance_decision": "accepted"},
                {"relevance_decision": "rejected"},
            ],
        }
        second_attempt = {"dense_candidates": [{}], "retrieval_candidates": []}

        self.assertEqual(retrieval_attempt_counts(first_attempt), {
            "Dense": 2, "BM25": 1, "RRF": 2, "Cross-Encoder": 2,
            "Selected / graded": 2, "Relevant": "1 / 2 graded",
        })
        self.assertEqual(retrieval_attempt_counts(second_attempt), {
            "Dense": 1, "RRF": 0,
        })

    def test_history_serialization_is_bounded_and_keeps_only_role_and_content(self):
        history = serialize_history([
            {"role": "user", "content": "old"},
            *[
                {"role": "assistant", "content": f"turn {index}", "result": {"secret": True}}
                for index in range(7)
            ],
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": 42},
        ])

        self.assertEqual(len(history), 6)
        self.assertEqual(history[0], {"role": "assistant", "content": "turn 1"})
        self.assertTrue(all(set(item) == {"role", "content"} for item in history))

    def test_query_request_sends_history_to_existing_endpoint(self):
        response = type("Response", (), {
            "status_code": 200,
            "json": lambda self: {"answer": "Follow-up answer"},
        })()
        history = [{"role": "user", "content": "Previous question"}]
        with patch("app.frontend.httpx.post", return_value=response) as post:
            query_backend("http://api.test", "What about it?", history=history)

        self.assertEqual(post.call_args.args[0], "http://api.test/query")
        self.assertEqual(post.call_args.kwargs["json"], {
            "question": "What about it?", "history": history,
        })

    def test_parses_backend_response_and_preserves_source_metadata(self):
        result = parse_query_response({
            "answer": "Grounded answer",
            "sources": [{
                "source": "report.pdf",
                "page": 4,
                "text": "Evidence",
                "chunk_id": 12,
                "rrf_rank": 3,
                "reranker_rank": 1,
                "reranker_score": 0.76,
            }],
            "route": "documents",
            "evidence_source": "documents",
            "diagnostics": {"retrieval_attempts": []},
            "used_web": False,
            "rewrite_count": 0,
            "latency_ms": 125,
        })

        self.assertEqual(result["answer"], "Grounded answer")
        self.assertEqual(result["sources"][0]["reranker_rank"], 1)
        self.assertEqual(result["sources"][0]["source"], "report.pdf")
        self.assertEqual(result["sources"][0]["page"], 4)
        self.assertEqual(result["sources"][0]["rrf_rank"], 3)
        self.assertEqual(result["sources"][0]["reranker_score"], 0.76)
        self.assertEqual(result["route"], "documents")
        self.assertEqual(result["diagnostics"], {"retrieval_attempts": []})

    def test_missing_optional_fields_are_handled_safely(self):
        result = parse_query_response({"answer": "Answer only"})

        self.assertEqual(result["sources"], [])
        self.assertEqual(result["diagnostics"], {})
        self.assertIsNone(result["route"])
        self.assertIsNone(result["used_web"])
        self.assertIsNone(result["rewrite_count"])

    def test_multiple_attempts_rewrites_and_web_diagnostics_are_preserved(self):
        diagnostics = {
            "retrieval_attempts": [
                {"query": "initial", "dense_candidates": [{"rank": 1}]},
                {"query": "rewritten", "dense_candidates": [{"rank": 1}, {"rank": 2}]},
            ],
            "rewritten_queries": ["rewritten"],
            "web_fallback_invoked": True,
            "web_results": [{
                "title": "Cloud security",
                "url": "https://example.test/security",
                "snippet": "Web evidence snippet",
            }],
            "final_evidence_source": "web",
        }
        result = parse_query_response({
            "answer": "Answer", "diagnostics": diagnostics,
            "evidence_source": "web", "used_web": True,
        })

        self.assertEqual(len(result["diagnostics"]["retrieval_attempts"]), 2)
        self.assertEqual(result["diagnostics"]["rewritten_queries"], ["rewritten"])
        self.assertEqual(result["diagnostics"]["web_results"][0]["snippet"], "Web evidence snippet")
        self.assertEqual(result["evidence_source"], "web")

    def test_conversation_history_is_not_parsed_as_evidence(self):
        claim = "Previous assistant claim: revenue was exactly $999 billion."
        result = parse_query_response({
            "answer": "Answer",
            "history": [{"role": "assistant", "content": claim}],
            "sources": [{"source": "report.pdf", "page": 2, "text": "Retrieved evidence"}],
            "evidence_source": "documents",
        })

        self.assertNotIn("history", result)
        self.assertEqual(result["sources"][0]["text"], "Retrieved evidence")
        self.assertNotIn(claim, str(result["sources"]))

    def test_malformed_response_is_rejected(self):
        with self.assertRaisesRegex(BackendError, "valid answer"):
            parse_query_response({"sources": "not-a-list"})

    def test_backend_connection_failure_is_safely_reported(self):
        with patch("app.frontend.httpx.post", side_effect=httpx.ConnectError("secret host detail")):
            with self.assertRaises(BackendError) as raised:
                query_backend("http://api.test", "question")

        self.assertIn("backend is unavailable", str(raised.exception))
        self.assertNotIn("secret host detail", str(raised.exception))

    def test_backend_timeout_is_safely_reported(self):
        with patch("app.frontend.httpx.post", side_effect=httpx.ReadTimeout("private timeout detail")):
            with self.assertRaises(BackendError) as raised:
                query_backend("http://api.test", "question")

        self.assertIn("timed out", str(raised.exception))
        self.assertNotIn("private timeout detail", str(raised.exception))

    def test_health_failure_is_reported_without_backend_details(self):
        with patch("app.frontend.httpx.get", side_effect=httpx.ConnectError("internal host")):
            status = get_system_status("http://api.test")

        self.assertFalse(status["api_healthy"])
        self.assertFalse(status["qdrant_healthy"])
        self.assertIsNone(status["indexed_chunks"])

    def test_health_response_exposes_only_app_level_chunk_count(self):
        api_response = type("Response", (), {
            "is_success": True,
            "json": lambda self: {"status": "ok", "version": "test"},
        })()
        qdrant_response = type("Response", (), {
            "is_success": True,
            "json": lambda self: {
                "status": "ok", "points_count": 12,
                "collection_name": "internal-name", "vector_size": 1024,
            },
        })()
        with patch("app.frontend.httpx.get", side_effect=[api_response, qdrant_response]):
            status = get_system_status("http://api.test")

        self.assertTrue(status["api_healthy"])
        self.assertTrue(status["qdrant_healthy"])
        self.assertEqual(status["indexed_chunks"], 12)
        self.assertNotIn("collection_name", status)

    def test_health_runtime_configuration_is_whitelisted_without_secrets(self):
        secret = "sk-never-return-this"
        api_response = type("Response", (), {
            "is_success": True,
            "json": lambda self: {
                "status": "ok",
                "configuration": {
                    "llm_provider": "OpenAI",
                    "llm_model": "gpt-test",
                    "OPENAI_API_KEY": secret,
                },
            },
        })()
        qdrant_response = type("Response", (), {
            "is_success": True,
            "json": lambda self: {"status": "ok", "points_count": 3},
        })()
        with patch("app.frontend.httpx.get", side_effect=[api_response, qdrant_response]):
            status = get_system_status("http://api.test")

        self.assertEqual(status["configuration"], {
            "llm_provider": "OpenAI", "llm_model": "gpt-test",
        })
        self.assertNotIn(secret, str(status))

    def test_ingest_response_parsing_uses_existing_contract(self):
        self.assertEqual(
            parse_ingest_response({"status": "indexed", "filename": "guide.pdf", "chunks": 9}),
            {"filename": "guide.pdf", "chunks": 9},
        )
        with self.assertRaises(BackendError):
            parse_ingest_response({"status": "indexed", "filename": "guide.pdf", "chunks": "many"})

    def test_ingest_upload_sends_pdf_to_existing_endpoint(self):
        response = type("Response", (), {
            "status_code": 200,
            "json": lambda self: {"status": "indexed", "filename": "guide.pdf", "chunks": 9},
        })()
        with patch("app.frontend.httpx.post", return_value=response) as post:
            result = ingest_pdf("http://api.test/", "guide.pdf", b"%PDF-test")

        self.assertEqual(result, {"filename": "guide.pdf", "chunks": 9})
        self.assertEqual(post.call_args.args[0], "http://api.test/ingest")
        self.assertEqual(
            post.call_args.kwargs["files"]["file"],
            ("guide.pdf", b"%PDF-test", "application/pdf"),
        )

    def test_invalid_pdf_upload_is_rejected_before_request(self):
        with patch("app.frontend.httpx.post") as post:
            with self.assertRaisesRegex(BackendError, r"\.pdf"):
                ingest_pdf("http://api.test", "notes.txt", b"not a pdf")
            with self.assertRaisesRegex(BackendError, "filename is invalid"):
                ingest_pdf("http://api.test", "..\\private.pdf", b"%PDF-test")
            with self.assertRaisesRegex(BackendError, "empty or unreadable"):
                ingest_pdf("http://api.test", "empty.pdf", b"")

        post.assert_not_called()

    def test_ingest_failure_hides_internal_error_details(self):
        response = type("Response", (), {
            "status_code": 500,
            "json": lambda self: {"detail": "secret path and stack trace"},
        })()
        with patch("app.frontend.httpx.post", return_value=response):
            with self.assertRaises(BackendError) as raised:
                ingest_pdf("http://api.test", "guide.pdf", b"%PDF-test")

        self.assertIn("indexing failed", str(raised.exception))
        self.assertNotIn("secret path", str(raised.exception))


class StreamlitAppTests(unittest.TestCase):
    def test_streamlit_module_runs_when_backend_is_unavailable(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        app = AppTest.from_file(str(app_path)).run()

        self.assertFalse(app.exception)
        self.assertTrue(any(title.value == "Enterprise Agentic RAG" for title in app.title))
        self.assertEqual(app.radio[0].value, "Ask")
        app.radio[0].set_value("Knowledge Base").run()
        self.assertTrue(any(title.value == "Knowledge Base" for title in app.title))

    def test_evaluation_page_reads_saved_benchmark_artifacts(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        app = AppTest.from_file(str(app_path)).run()
        app.radio[0].set_value("Evaluation").run()

        self.assertFalse(app.exception)
        self.assertTrue(any(title.value == "Evaluation" for title in app.title))
        self.assertTrue(any(
            metric.label == "Questions" and str(metric.value) == "30"
            for metric in app.metric
        ))

    def test_retrieval_trace_page_has_a_clear_empty_state(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        app = AppTest.from_file(str(app_path)).run()
        app.radio[0].set_value("Retrieval Trace").run()

        self.assertFalse(app.exception)
        self.assertTrue(any(title.value == "Retrieval Trace" for title in app.title))
        self.assertTrue(any("Ask a question first" in item.value for item in app.info))

    def test_system_page_renders_runtime_architecture(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        app = AppTest.from_file(str(app_path)).run()
        app.radio[0].set_value("System").run()

        self.assertFalse(app.exception)
        self.assertTrue(any(title.value == "System" for title in app.title))
        self.assertTrue(any(metric.label == "Qdrant" for metric in app.metric))

    def test_new_chat_clears_messages_without_clearing_indexed_documents(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        app = AppTest.from_file(str(app_path)).run()
        app.session_state["messages"] = [{"role": "user", "content": "prior turn"}]
        app.session_state["indexed_documents"] = [{"document": "guide.pdf"}]
        app.run()

        app.button[0].click().run()

        self.assertEqual(app.session_state["messages"], [])
        self.assertEqual(
            app.session_state["indexed_documents"], [{"document": "guide.pdf"}]
        )

    def test_multiple_chat_answers_render_each_evidence_text_area(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        app = AppTest.from_file(str(app_path)).run()
        app.session_state["messages"] = [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": "First answer",
                "result": {
                    "answer": "First answer",
                    "evidence_source": "documents",
                    "sources": [
                        {"source": "guide.pdf", "page": 1, "text": "First evidence"},
                        {"source": "guide.pdf", "page": 2, "text": "Second evidence"},
                    ],
                    "diagnostics": {},
                    "route": "documents",
                    "used_web": False,
                },
            },
            {"role": "user", "content": "Second question"},
            {
                "role": "assistant",
                "content": "Second answer",
                "result": {
                    "answer": "Second answer",
                    "evidence_source": "documents",
                    "sources": [
                        {"source": "guide.pdf", "page": 3, "text": "Third evidence"},
                        {"source": "guide.pdf", "page": 4, "text": "Fourth evidence"},
                    ],
                    "diagnostics": {},
                    "route": "documents",
                    "used_web": False,
                },
            },
        ]

        app.run()

        self.assertFalse(app.exception)
        evidence_widgets = [
            widget for widget in app.text_area
            if widget.value in {
                "First evidence", "Second evidence", "Third evidence", "Fourth evidence",
            }
        ]
        self.assertEqual(len(evidence_widgets), 4)


if __name__ == "__main__":
    unittest.main()
