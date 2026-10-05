"""Streamlit presentation layer for the existing FastAPI RAG backend."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, List

import streamlit as st
from dotenv import load_dotenv

from app.frontend import (
    BackendError, document_evidence_details, get_system_status, ingest_pdf,
    knowledge_base_state, load_evaluation_artifacts, query_backend,
    retrieval_attempt_counts, serialize_history,
)

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
EXAMPLE_QUESTIONS = (
    "How does Google Cloud protect data in transit?",
    "What security mechanisms protect data while it is being processed?",
    "What are Microsoft's major revenue growth drivers?",
    "How does Google use encryption between services?",
)
PAGES = ("Ask", "Knowledge Base", "Retrieval Trace", "Evaluation", "System")

st.set_page_config(page_title="Enterprise Agentic RAG", layout="wide")


@st.cache_data(ttl=10, show_spinner=False)
def _system_status(base_url: str) -> Dict[str, Any]:
    return get_system_status(base_url)


def _render_sidebar(status: Dict[str, Any]) -> str:
    with st.sidebar:
        st.title("Enterprise Agentic RAG")
        st.caption(
            "Hybrid enterprise document intelligence using dense retrieval, BM25, "
            "reranking, corrective RAG and grounded generation."
        )

        page = st.radio("Workspace", PAGES, label_visibility="visible")
        st.divider()
        st.subheader("Services")
        if status["api_healthy"]:
            st.success("API healthy")
        else:
            st.error("API unavailable")
        if status["qdrant_healthy"]:
            st.success("Qdrant connected")
        else:
            st.warning("Qdrant unavailable")

        st.subheader("Pipeline")
        configuration = status.get("configuration") or {}
        st.caption(f"Vector store · {configuration.get('vector_store', 'Unavailable')}")
        st.caption("Dense embeddings · BGE-M3")
        st.caption("Lexical retrieval · BM25")
        st.caption("Fusion · RRF")
        st.caption("Reranker · Cross-Encoder")
        st.caption("Orchestration · LangGraph")
        provider = configuration.get("llm_provider", "Unavailable")
        st.caption(f"LLM · {provider}")
        st.caption("Web fallback · Tavily")
    return page


def _candidate_rows(
    candidates: Iterable[Dict[str, Any]], stage: str
) -> List[Dict[str, Any]]:
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        rank = candidate.get(
            "reranker_rank", candidate.get("rrf_rank", candidate.get("rank"))
        )
        score = candidate.get(
            "reranker_score", candidate.get("rrf_score", candidate.get("score"))
        )
        rows.append({
            "Stage": stage,
            "Rank": rank if rank is not None else "—",
            "Source": candidate.get("source", "—"),
            "Page": candidate.get("page") if candidate.get("page") is not None else "—",
            "Chunk ID": candidate.get("chunk_id", "—"),
            "Score": (
                f"{score:.4f}"
                if isinstance(score, (int, float)) and not isinstance(score, bool)
                else "—"
            ),
            "Decision": candidate.get("relevance_decision", "—"),
            "Evidence preview": candidate.get("text_preview", ""),
        })
    return rows


def _render_evidence(result: Dict[str, Any], message_index: int) -> None:
    source_kind = result.get("evidence_source")
    diagnostics = result.get("diagnostics") or {}
    if result.get("used_web") or diagnostics.get("web_fallback_invoked"):
        st.warning(
            "Web fallback used. The document corpus did not provide sufficient evidence, "
            "so the agent searched the web."
        )

    with st.expander("Sources & Evidence", expanded=False):
        st.caption(
            "Conversation history can clarify your question; only retrieved documents or "
            "web results shown here are evidence."
        )
        if source_kind == "insufficient_evidence":
            st.warning("The backend could not find sufficient document or web evidence.")
            return

        if source_kind == "web":
            web_results = diagnostics.get("web_results") or []
            web_results = [item for item in web_results if isinstance(item, dict)]
            if not web_results:
                st.caption("The backend did not return web evidence details.")
            if web_results:
                tabs = st.tabs([
                    f"{index}. {item.get('title') or 'Web result'}"
                    for index, item in enumerate(web_results, start=1)
                ])
                for tab, item in zip(tabs, web_results):
                    with tab:
                        url = item.get("url")
                        if url:
                            st.markdown(f"[{url}]({url})")
                        if item.get("snippet"):
                            st.write(item["snippet"])
            return

        if source_kind == "documents":
            sources = result.get("sources") or []
            sources = [item for item in sources if isinstance(item, dict)]
            if not sources:
                st.caption("The backend reported document evidence but returned no source details.")
            else:
                labels = [
                    (item.get("source") or f"Document chunk {index}")
                    + (
                        f" · Page {item['page']}"
                        if isinstance(item.get("page"), int)
                        and not isinstance(item.get("page"), bool) else ""
                    )
                    + f" · Evidence {index}"
                    for index, item in enumerate(sources, start=1)
                ]
                tabs = st.tabs(labels)
                for index, (tab, item) in enumerate(zip(tabs, sources), start=1):
                    with tab:
                        st.caption(" · ".join(document_evidence_details(item, index)))
                        if isinstance(item.get("text"), str) and item["text"]:
                            st.text_area(
                                "Retrieved evidence",
                                item["text"],
                                height=240,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"evidence_text_area_{message_index}_{index}",
                            )
        elif source_kind == "llm_only":
            st.caption("No document or web evidence was used for this response.")
        elif source_kind not in ("web", "insufficient_evidence"):
            st.caption("The backend did not report a final evidence source.")


def _render_workflow(result: Dict[str, Any], expanded: bool = False) -> None:
    diagnostics = result.get("diagnostics") or {}
    route = diagnostics.get("initial_route") or result.get("route")
    attempts = diagnostics.get("retrieval_attempts") or []
    attempts = [attempt for attempt in attempts if isinstance(attempt, dict)]
    rewrites = diagnostics.get("rewritten_queries") or []
    rewrites = (
        [query for query in rewrites if isinstance(query, str)]
        if isinstance(rewrites, list) else []
    )
    with st.expander("Agent Workflow & Retrieval Trace", expanded=expanded):
        st.caption(
            "Conversation history may contextualize a follow-up; it is not evidence. "
            "Counts below come from the backend diagnostics."
        )
        if route:
            st.markdown(f"**Initial route:** `{route}`")
        if diagnostics.get("query_contextualized"):
            contextual_query = diagnostics.get("initial_query")
            if isinstance(contextual_query, str):
                st.markdown(f"**Contextualized query:** {contextual_query}")
        if diagnostics.get("corpus_match_override"):
            st.caption("Document route selected by the corpus-match guard.")
        if not attempts:
            if route == "documents":
                st.caption("No retrieval attempt details were returned by the backend.")
            elif route == "llm_only":
                st.markdown("**Document retrieval:** not run for this route.")
            else:
                st.caption("Document retrieval details were not reported.")
        for index, attempt in enumerate(attempts, start=1):
            query = attempt.get("query")
            title = f"Retrieval Attempt {index}"
            if isinstance(query, str) and query:
                title += f" · {query}"
            with st.container(border=True):
                st.markdown(f"**{title}**")
                counts = retrieval_attempt_counts(attempt)
                if counts:
                    metric_columns = st.columns(len(counts))
                    for column, (label, count) in zip(metric_columns, counts.items()):
                        column.metric(label, count)

                stage_fields = (
                    ("Dense", "dense_candidates"),
                    ("BM25", "bm25_candidates"),
                    ("RRF", "retrieval_candidates"),
                    ("Cross-Encoder", "reranker_candidates"),
                    ("Selected / graded", "reranked_candidates"),
                )
                candidate_stages = [
                    (stage, attempt.get(key))
                    for stage, key in stage_fields
                    if isinstance(attempt.get(key), list) and attempt.get(key)
                ]
                if candidate_stages:
                    candidate_tabs = st.tabs([
                        f"{stage} ({len(candidates)})"
                        for stage, candidates in candidate_stages
                    ])
                    for tab, (stage, candidates) in zip(candidate_tabs, candidate_stages):
                        with tab:
                            st.dataframe(
                                _candidate_rows(candidates, stage),
                                hide_index=True,
                                use_container_width=True,
                            )

        if rewrites:
            st.markdown(f"**Query Rewrite:** {len(rewrites)} attempt(s)")
            for index, rewritten_query in enumerate(rewrites, start=1):
                st.write(f"Attempt {index} → {rewritten_query}")
        elif route == "documents" and attempts:
            graded = any(
                isinstance(item, dict)
                and item.get("relevance_decision") == "accepted"
                for attempt in attempts
                for item in (attempt.get("reranked_candidates") or [])
                if isinstance(attempt.get("reranked_candidates"), list)
            )
            st.markdown(
                "**Query Rewrite:** not required; relevant document evidence was accepted."
                if graded else "**Query Rewrite:** no rewrite was reported."
            )
        elif route == "llm_only":
            st.markdown("**Query Rewrite:** not run (document retrieval was not selected).")
        else:
            st.markdown("**Query Rewrite:** not reported.")

        web_invoked = (
            diagnostics.get("web_fallback_invoked") is True
            or result.get("used_web") is True
        )
        web_results = diagnostics.get("web_results")
        if web_invoked:
            result_count = len(web_results) if isinstance(web_results, list) else 0
            st.markdown(f"**Web Fallback:** used · {result_count} result(s)")
            if diagnostics.get("web_fallback_error"):
                st.caption(f"Fallback error: {diagnostics['web_fallback_error']}")
        elif diagnostics.get("web_fallback_invoked") is False or result.get("used_web") is False:
            st.markdown("**Web Fallback:** not used")
        else:
            st.markdown("**Web Fallback:** not reported")

        evidence_source = result.get("evidence_source")
        st.markdown(f"**Final evidence source:** `{evidence_source or 'not reported'}`")
        if evidence_source == "insufficient_evidence":
            st.markdown("**Generation:** skipped because no evidence was available.")
        elif isinstance(result.get("answer"), str) and result["answer"]:
            st.markdown("**Generation:** answer returned.")


def _render_answer(result: Dict[str, Any], message_index: int) -> None:
    if result.get("evidence_source") == "insufficient_evidence":
        st.warning("Insufficient evidence")
    st.subheader("Answer")
    st.markdown(result["answer"])
    _render_evidence(result, message_index)
    _render_workflow(result)
    metadata = []
    if result.get("latency_ms") is not None:
        latency_ms = result["latency_ms"]
        metadata.append(
            f"{latency_ms / 1000:.1f}s" if latency_ms >= 1000 else f"{latency_ms} ms"
        )
    if result.get("rewrite_count") is not None:
        metadata.append(f"{result['rewrite_count']} rewrite(s)")
    if metadata:
        st.caption(" · ".join(metadata))


def _ask_page() -> None:
    st.title("Enterprise Agentic RAG")
    st.write("Ask questions about the indexed enterprise documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("New Chat", key="new_chat"):
        st.session_state.messages = []
        st.rerun()

    if not st.session_state.messages:
        st.info("Start with a question below, or choose an example to get going.")

    st.subheader("Try these questions")
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(
            example, key=f"example_{index}", use_container_width=True
        ):
            st.session_state.question_input = example

    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Ask a question about your documents...",
            key="question_input",
            height=110,
        )
        submitted = st.form_submit_button("Search / Ask", type="primary")

    if submitted:
        clean_question = question.strip()
        if not clean_question:
            st.warning("Enter a question to search the indexed documents.")
        else:
            history = serialize_history([
                message for message in st.session_state.messages
                if not message.get("error")
            ])
            st.session_state.messages.append({"role": "user", "content": clean_question})
            with st.spinner("Searching the knowledge base and preparing an answer..."):
                try:
                    result = query_backend(API_BASE, clean_question, history=history)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "result": result,
                    })
                except BackendError as exc:
                    st.session_state.messages.append({
                        "role": "assistant", "content": str(exc), "error": True,
                    })
                except Exception:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            "The request could not be completed. Please try again or "
                            "contact the system administrator."
                        ),
                        "error": True,
                    })

    st.divider()
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["content"])
            elif message["role"] == "assistant" and message.get("result"):
                _render_answer(message["result"], message_index)
            else:
                st.write(message["content"])


def _knowledge_base_page(status: Dict[str, Any]) -> None:
    st.title("Knowledge Base")
    st.write("Add a PDF to the existing backend index and check corpus readiness.")

    chunks = status.get("indexed_chunks")
    knowledge_status = knowledge_base_state(status)
    cols = st.columns(3)
    cols[0].metric("API", "Healthy" if status["api_healthy"] else "Unavailable")
    cols[1].metric("Vector store", "Connected" if status["qdrant_healthy"] else "Unavailable")
    cols[2].metric("Knowledge Base", knowledge_status)
    if chunks is not None:
        st.caption(f"Indexed chunks reported by the backend: {chunks}")
    if not status["api_healthy"]:
        st.warning("The API is unavailable. Start the backend before indexing documents.")
    elif not status["qdrant_healthy"]:
        st.warning("Qdrant is currently unavailable. Start the vector service and try again.")
    elif chunks == 0:
        st.info("The vector service is connected, but the backend reports no indexed chunks yet.")

    notice = st.session_state.pop("knowledge_base_notice", None)
    if notice:
        if notice["kind"] == "success":
            st.success(notice["message"])
        else:
            st.error(notice["message"])

    uploaded = st.file_uploader("Select a PDF", type=["pdf"], key="knowledge_pdf")
    if uploaded is not None:
        st.caption(f"Selected: {uploaded.name} · {uploaded.size:,} bytes")
    if st.button("Index PDF", type="primary", disabled=uploaded is None):
        with st.spinner("Uploading PDF and indexing through the backend..."):
            try:
                indexed = ingest_pdf(API_BASE, uploaded.name, uploaded.getvalue())
                indexed_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
                documents = st.session_state.setdefault("indexed_documents", [])
                documents[:] = [
                    item for item in documents if item["document"] != indexed["filename"]
                ]
                documents.insert(0, {
                    "document": indexed["filename"],
                    "status": "Indexed in this session",
                    "indexed": indexed_at,
                })
                st.session_state.knowledge_base_notice = {
                    "kind": "success",
                    "message": (
                        f"{indexed['filename']} was indexed successfully. "
                        f"The backend reports {indexed['chunks']:,} chunks in the index."
                    ),
                }
                _system_status.clear()
            except BackendError as exc:
                st.session_state.knowledge_base_notice = {
                    "kind": "error", "message": str(exc),
                }
            except Exception:
                st.session_state.knowledge_base_notice = {
                    "kind": "error",
                    "message": "Indexing failed. Check API and Qdrant status, then try again.",
                }
        st.rerun()

    st.subheader("Documents indexed in this session")
    documents = st.session_state.get("indexed_documents", [])
    if documents:
        st.dataframe(documents, hide_index=True, use_container_width=True)
    else:
        st.caption("No documents have been indexed from this session yet.")
    st.caption(
        "The backend does not expose a persistent document catalog or per-document page/chunk "
        "counts. The list above reflects successful uploads in this Streamlit session only."
    )


def _evaluation_page() -> None:
    st.title("Evaluation")
    evaluation = load_evaluation_artifacts()
    if not evaluation.get("available"):
        st.warning("Evaluation results are currently unavailable.")
        st.caption("The saved benchmark artifacts are missing or malformed. No evaluation was run.")
        return

    st.write("Latest saved corpus-grounded benchmark results.")
    summary = st.columns(3)
    summary[0].metric("Questions", evaluation["question_count"])
    summary[1].metric("Status", evaluation["status"].title())
    if evaluation.get("updated_at"):
        summary[2].metric("Artifact updated", evaluation["updated_at"])

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }
    metric_columns = st.columns(5)
    for column, (key, label) in zip(metric_columns, metric_labels.items()):
        column.metric(label, f"{evaluation['scores'][key]:.4f}")
    metric_columns[4].metric("Average", f"{evaluation['average']:.4f}")

    st.subheader("Per-question results")
    table_rows = []
    for item in evaluation["questions"]:
        table_rows.append({
            "ID": item["id"],
            "Question": item["question"],
            "Category": item.get("category") or "",
            "Difficulty": item.get("difficulty") or "",
            "Faithfulness": f"{item['metrics']['faithfulness']:.3f}",
            "Answer Relevancy": f"{item['metrics']['answer_relevancy']:.3f}",
            "Context Precision": f"{item['metrics']['context_precision']:.3f}",
            "Context Recall": f"{item['metrics']['context_recall']:.3f}",
            "Route": item.get("route") or "",
            "Evidence": item.get("evidence_source") or "",
        })
    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    questions = evaluation["questions"]
    selected_id = st.selectbox(
        "Inspect a question",
        [item["id"] for item in questions],
        format_func=lambda question_id: next(
            f"{item['id']} · {item['question']}"
            for item in questions if item["id"] == question_id
        ),
    )
    selected = next(item for item in questions if item["id"] == selected_id)
    with st.expander(f"Question detail · {selected['id']}", expanded=False):
        st.markdown(f"**Question**  \n{selected['question']}")
        st.markdown("**Answer**")
        st.write(selected["answer"] or "No answer was saved.")
        st.caption(
            f"Category: {selected.get('category') or 'not reported'} · "
            f"Difficulty: {selected.get('difficulty') or 'not reported'} · "
            f"Route: {selected.get('route') or 'not reported'} · "
            f"Evidence: {selected.get('evidence_source') or 'not reported'}"
        )
        question_metrics = st.columns(4)
        for column, (key, label) in zip(question_metrics, metric_labels.items()):
            column.metric(label, f"{selected['metrics'][key]:.4f}")


def _system_page(status: Dict[str, Any]) -> None:
    st.title("System")
    st.write("Runtime status and configured application components.")
    service_columns = st.columns(2)
    service_columns[0].metric("FastAPI", "Healthy" if status["api_healthy"] else "Unavailable")
    service_columns[1].metric("Qdrant", "Connected" if status["qdrant_healthy"] else "Unavailable")
    if status.get("indexed_chunks") is not None:
        st.metric("Indexed chunks", status["indexed_chunks"])

    st.subheader("Application components")
    configuration = status.get("configuration")
    if not isinstance(configuration, dict):
        st.info("Runtime configuration is unavailable while the backend is offline.")
    else:
        components = [
            ("LLM", f"{configuration.get('llm_provider', 'Not reported')} · "
                    f"{configuration.get('llm_model', 'Not reported')}"),
            ("LLM credentials", configuration.get("llm_credential_status", "Not reported")),
            ("Embeddings", configuration.get("embedding_model", "Not reported")),
            ("Vector store", configuration.get("vector_store", "Not reported")),
            ("Sparse retrieval", configuration.get("sparse_retrieval", "Not reported")),
            ("Fusion", configuration.get("fusion", "Not reported")),
            ("Reranking", configuration.get("reranking", "Not reported")),
            ("Orchestration", configuration.get("orchestration", "Not reported")),
            ("Web fallback", configuration.get("web_fallback", "Not reported")),
            ("API", configuration.get("api", "Not reported")),
            ("Frontend", configuration.get("frontend", "Not reported")),
        ]
        for start in range(0, len(components), 3):
            columns = st.columns(3)
            for column, (label, value) in zip(columns, components[start:start + 3]):
                column.metric(label, value)
    st.caption("Credentials are represented by status only; secret values are never displayed.")


def _retrieval_trace_page() -> None:
    st.title("Retrieval Trace")
    latest_result = next((
        message.get("result")
        for message in reversed(st.session_state.get("messages", []))
        if message.get("role") == "assistant" and isinstance(message.get("result"), dict)
    ), None)
    if latest_result is None:
        st.info("Ask a question first. Its backend retrieval trace will appear here.")
        return
    st.write("Latest answer workflow from the backend diagnostics.")
    _render_workflow(latest_result, expanded=True)


def main() -> None:
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = []
    status = _system_status(API_BASE)
    page = _render_sidebar(status)

    if page == "Ask":
        _ask_page()
    elif page == "Knowledge Base":
        _knowledge_base_page(status)
    elif page == "Retrieval Trace":
        _retrieval_trace_page()
    elif page == "Evaluation":
        _evaluation_page()
    else:
        _system_page(status)


main()
