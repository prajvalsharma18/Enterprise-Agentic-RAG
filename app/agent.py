"""
app/agent.py
────────────
Corrective RAG Agent built with LangGraph 0.4+

Graph nodes
───────────
  route_query      → decides: use documents OR answer from LLM knowledge
  retrieve         → hybrid vector+BM25 search + cross-encoder rerank
  grade_documents  → LLM judges if retrieved chunks are relevant
  rewrite_query    → rewrites query if grading failed (Corrective RAG loop)
  generate         → produces final answer with citations
  web_search       → Tavily fallback when docs fail after rewrite

State machine
─────────────
  route_query
      ├─→ "documents"  → retrieve → grade_documents
      │                     ├─→ "generate"     → generate → END
      │                     └─→ "rewrite"      → rewrite_query → retrieve (loop, max 2x)
      │                           └─→ "web_search" (after 2 rewrites) → generate → END
      └─→ "llm_only"   → generate → END
"""

from __future__ import annotations
import json
import os
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, List, Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from loguru import logger

load_dotenv()

from app.llm_provider import create_chat_model, extract_text_content
from vectorstore.store import hybrid_search, query_matches_indexed_corpus

# All LangGraph LLM operations use the selected provider.
llm = create_chat_model()

# ── Web search fallback (free Tavily key at tavily.com) ──────────────────────
web_search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
)

MAX_REWRITES = 2   # prevent infinite loops
MAX_HISTORY_MESSAGES = 6  # retain at most three recent user/assistant turns


# ── Agent state ──────────────────────────────────────────────────────────────

class _AgentStateRequired(TypedDict):
    messages:      Annotated[List[BaseMessage], add_messages]   # full conversation history
    query:         str
    rewrite_count: int
    context:       List[Dict[str, Any]]   # retrieved chunk dicts from hybrid_search
    web_results:   List[Dict[str, str]]
    answer:        str

class AgentState(_AgentStateRequired, total=False):
    """LangGraph state — _route and _grade are injected by nodes, not part of initial state."""
    _route: str   # 'documents' | 'llm_only'
    _grade: str   # 'generate' | 'rewrite'
    web_search_invoked: bool
    diagnostics: Dict[str, Any]
    generation_evidence_source: str
    generation_context: List[str]
    original_query: str


# ── Node helpers ─────────────────────────────────────────────────────────────

def _last_human_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return state.get("query") or ""  # type: ignore[union-attr]


def _contextualize_query(query: str, history: List[BaseMessage]) -> str:
    """Resolve follow-up references using recent conversation, without treating it as evidence."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    if not recent:
        return query

    history_text = "\n".join(
        f"{'User' if isinstance(message, HumanMessage) else 'Assistant'}: "
        f"{str(message.content)[:2000]}"
        for message in recent
        if isinstance(message, (HumanMessage, AIMessage))
    )
    if not history_text:
        return query

    prompt = (
        "Determine whether the current question depends on the conversation to resolve "
        "a reference, omitted subject, or follow-up. Conversation is for meaning only, "
        "not factual evidence. If the question stands alone, set uses_context to false "
        "and keep it unchanged. If it depends on prior context, rewrite it as a "
        "standalone question while preserving the user's intent. Do not answer it. "
        "Return only JSON with boolean uses_context and string query.\n\n"
        f"Recent conversation:\n{history_text}\n\nCurrent question: {query}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        result = json.loads(extract_text_content(response.content))
    except (TypeError, ValueError):
        return query
    contextualized = result.get("query") if isinstance(result, dict) else None
    if (
        isinstance(result, dict)
        and result.get("uses_context") is True
        and isinstance(contextualized, str)
        and contextualized.strip()
    ):
        return contextualized.strip()
    return query


def contextualize_query(state: AgentState) -> AgentState:
    """Resolve conversational references before the existing routing/retrieval flow."""
    messages = state.get("messages") or []
    history = messages[:-1]
    query = state.get("query") or _last_human_query(state)
    contextualized = _contextualize_query(query, history)
    if contextualized == query:
        return state
    diagnostics = dict(state.get("diagnostics") or {})
    diagnostics["query_contextualized"] = True
    return {
        **state,
        "query": contextualized,
        "original_query": query,
        "diagnostics": diagnostics,
    }


# ── Nodes ────────────────────────────────────────────────────────────────────

def route_query(state: AgentState) -> AgentState:
    """
    Ask LLM whether the question needs document retrieval.
    Returns updated state; routing decision is in 'query' metadata.
    """
    query = state.get("query") or _last_human_query(state)
    prompt = (
        "You are a routing assistant for an application with a supplied document corpus. "
        "Choose 'documents' for specific factual, technical, or business questions that "
        "may be answered by the corpus, even if the user does not explicitly mention a "
        "document. Choose 'llm_only' only for broad common knowledge or conversational "
        "questions that do not need corpus evidence.\n\n"
        f"Question: {query}\n\nReply with exactly one word: documents OR llm_only"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = extract_text_content(response.content).strip().lower()
    if "documents" not in decision:
        decision = "llm_only"
    corpus_match_override = False
    if decision == "llm_only" and query_matches_indexed_corpus(query):
        decision = "documents"
        corpus_match_override = True
    logger.info(f"[route_query] decision='{decision}' for query='{query[:60]}'")
    diagnostics = dict(state.get("diagnostics") or {})
    diagnostics.update({
        "initial_route": decision,
        "initial_query": query,
        "corpus_match_override": corpus_match_override,
    })
    diagnostics.setdefault("retrieval_attempts", [])
    diagnostics.setdefault("rewritten_queries", [])
    return {**state, "query": query, "_route": decision, "diagnostics": diagnostics}


def retrieve(state: AgentState) -> AgentState:
    """Hybrid search → cross-encoder rerank → store chunks in state."""
    query   = state["query"]
    retrieval_diagnostics: Dict[str, Any] = {}
    results = hybrid_search(query, top_k=5, diagnostics=retrieval_diagnostics)
    logger.info(f"[retrieve] got {len(results)} chunks")
    diagnostics = dict(state.get("diagnostics") or {})
    attempts = list(diagnostics.get("retrieval_attempts", []))
    attempts.append({
        "query": query,
        "dense_candidates": retrieval_diagnostics.get("dense_candidates", []),
        "bm25_candidates": retrieval_diagnostics.get("bm25_candidates", []),
        "retrieval_candidates": retrieval_diagnostics.get("rrf_candidates", []),
        "reranker_candidates": retrieval_diagnostics.get("reranker_candidates", []),
        "reranker_score_available": retrieval_diagnostics.get(
            "reranker_score_available", False
        ),
        "reranked_candidates": [
            {
                "source": str(chunk.get("source", "unknown")),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "rrf_rank": chunk.get("rrf_rank"),
                "rrf_score": chunk.get("rrf_score"),
                "reranker_rank": chunk.get("reranker_rank", rank),
                "reranker_score": chunk.get("reranker_score"),
                "relevance_decision": "pending",
                "text_preview": str(chunk.get("text", ""))[:300],
            }
            for rank, chunk in enumerate(results, start=1)
        ],
    })
    diagnostics["retrieval_attempts"] = attempts
    return {**state, "context": results, "diagnostics": diagnostics}


def grade_documents(state: AgentState) -> AgentState:
    """
    LLM grades each retrieved chunk for relevance.
    Marks state with '_grade': 'generate' or 'rewrite'.
    """
    query   = state["query"]
    context = state["context"]

    relevant = []
    diagnostics = dict(state.get("diagnostics") or {})
    attempts = list(diagnostics.get("retrieval_attempts", []))
    candidate_diagnostics = (
        attempts[-1].get("reranked_candidates", []) if attempts else []
    )
    for chunk_index, chunk in enumerate(context):
        prompt = (
            f"Question: {query}\n\n"
            f"Document chunk:\n{chunk['text']}\n\n"
            "Judge the information in the chunk, not just whether it discusses the same topic. "
            "Reply yes if it directly answers any material part of the question or provides "
            "a specific fact needed to answer it. For a multi-part question, partial evidence "
            "for one requested part is relevant even if the chunk does not answer every part. "
            "Reply no when the chunk is only topically related and provides no information "
            "that helps answer the question. Reply with only yes or no."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        accepted = "yes" in extract_text_content(resp.content).lower()
        if chunk_index < len(candidate_diagnostics):
            candidate_diagnostics[chunk_index]["relevance_decision"] = (
                "accepted" if accepted else "rejected"
            )
        if accepted:
            relevant.append(chunk)

    diagnostics["retrieval_attempts"] = attempts
    diagnostics["final_document_context"] = [
        {
            "source": str(chunk.get("source", "unknown")),
            "page": chunk.get("page"),
            "chunk_id": chunk.get("chunk_id"),
            "text_preview": str(chunk.get("text", ""))[:300],
        }
        for chunk in relevant
    ]

    logger.info(f"[grade_documents] {len(relevant)}/{len(context)} chunks relevant")

    if relevant:
        return {**state, "context": relevant, "_grade": "generate", "diagnostics": diagnostics}
    else:
        return {**state, "_grade": "rewrite", "diagnostics": diagnostics}


def rewrite_query(state: AgentState) -> AgentState:
    """Corrective RAG: rewrite the query to improve retrieval."""
    query = state["query"]
    prompt = (
        f"The following question didn't retrieve useful documents:\n{query}\n\n"
        "Rewrite it to be more specific and likely to match technical documentation. "
        "Return only the rewritten question, nothing else."
    )
    response      = llm.invoke([HumanMessage(content=prompt)])
    new_query     = extract_text_content(response.content).strip()
    rewrite_count = int(state.get("rewrite_count") or 0) + 1  # type: ignore[union-attr]
    diagnostics = dict(state.get("diagnostics") or {})
    rewritten_queries = list(diagnostics.get("rewritten_queries", []))
    rewritten_queries.append(new_query)
    diagnostics["rewritten_queries"] = rewritten_queries
    logger.info(f"[rewrite_query] attempt {rewrite_count}: '{new_query[:80]}'")
    return {**state, "query": new_query, "rewrite_count": rewrite_count, "diagnostics": diagnostics}


def web_search(state: AgentState) -> AgentState:
    """Tavily web search as last-resort fallback."""
    query = state["query"]
    search_error = None
    try:
        results = web_search_tool.invoke(query) or []
    except Exception as exc:
        logger.warning(f"[web_search] Tavily fallback failed: {type(exc).__name__}")
        results = []
        search_error = type(exc).__name__
    safe_results = []
    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            parsed = urlsplit(str(result.get("url", "")))
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            safe_url = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        except ValueError:
            safe_url = ""
        safe_results.append({
            "title": str(result.get("title", ""))[:200],
            "url": safe_url[:500],
            "snippet": str(result.get("content", ""))[:500],
        })
    logger.info(f"[web_search] got {len(safe_results)} web results")
    diagnostics = dict(state.get("diagnostics") or {})
    diagnostics["web_fallback_invoked"] = True
    diagnostics["web_results"] = safe_results
    if search_error:
        diagnostics["web_fallback_error"] = search_error
    return {**state, "web_results": safe_results, "web_search_invoked": True, "diagnostics": diagnostics}


def _select_generation_evidence(state: AgentState):
    """Select one source; an invoked fallback supersedes stale document context."""
    if state.get("web_search_invoked"):
        results = state.get("web_results") or []
        if not results:
            return "insufficient_evidence", "", []
        block = "\n\n".join(
            f"[Title: {item['title']} | URL: {item['url']}]\n{item['snippet']}"
            for item in results
        )
        generation_context = [
            f"[Title: {item['title']} | URL: {item['url']}]\n{item['snippet']}"
            for item in results
        ]
        return "web", block, generation_context

    context = state.get("context") or []
    if context:
        block = "\n\n".join(
            f"[Source: {item['source']}, Page {item['page']}]\n{item['text']}"
            for item in context
        )
        return "documents", block, [item["text"] for item in context]
    return "llm_only", "", []


def generate(state: AgentState) -> AgentState:
    """
    Final generation node — synthesises answer from context + conversation history.
    Adds citations (source + page) when using document context.
    """
    query:       str                    = state.get("original_query") or state["query"]
    history:     List[BaseMessage]     = state.get("messages") or []       # type: ignore[assignment]

    evidence_source, ctx_block, generation_context = _select_generation_evidence(state)
    if evidence_source == "documents":
        source_note = "Cite sources as [Source, Page X] in your answer."
    elif evidence_source == "web":
        source_note = "These results are from the web. Cite their URLs when relevant."
    elif evidence_source == "insufficient_evidence":
        answer = "I could not find sufficient document or web evidence to answer this question."
        diagnostics = dict(state.get("diagnostics") or {})
        diagnostics.update({"final_evidence_source": evidence_source, "generation_context": []})
        return {
            **state, "answer": answer, "generation_evidence_source": evidence_source,
            "generation_context": [], "diagnostics": diagnostics,
            "messages": state["messages"] + [AIMessage(content=answer)],
        }
    else:
        source_note = "Answer from your general knowledge."

    # Conversation history (last 6 turns for context window efficiency)
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history[-6:]
    )

    system = (
        "You are a precise, helpful document assistant. "
        "Answer only from the provided context. "
        "If the context is insufficient, say so clearly. "
        "Use conversation history only to resolve references and maintain continuity; "
        "prior user or assistant claims are not verified evidence. For document- or "
        "web-grounded answers, rely on the current provided context. "
        f"{source_note}"
    )
    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {query}"
    )
    response = llm.invoke([
        HumanMessage(content=f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}")
    ])
    answer = extract_text_content(response.content)
    logger.info(f"[generate] answer length={len(answer)} chars")
    diagnostics = dict(state.get("diagnostics") or {})
    diagnostics.update({
        "final_evidence_source": evidence_source,
        "generation_context": generation_context,
    })
    return {
        **state,
        "answer":   answer,
        "generation_evidence_source": evidence_source,
        "generation_context": generation_context,
        "diagnostics": diagnostics,
        "messages": state["messages"] + [AIMessage(content=answer)],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_after_routing(state: AgentState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state.get("_route") == "documents" else "generate"  # type: ignore[union-attr]


def route_after_grading(state: AgentState) -> Literal["generate", "rewrite_query", "web_search"]:
    grade = state.get("_grade") or "generate"                               # type: ignore[union-attr]
    if grade == "generate":
        return "generate"
    # After MAX_REWRITES failed attempts, fall through to web search
    if int(state.get("rewrite_count") or 0) >= MAX_REWRITES:               # type: ignore[union-attr]
        return "web_search"
    return "rewrite_query"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route_query",     route_query)
    g.add_node("contextualize_query", contextualize_query)
    g.add_node("retrieve",        retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query",   rewrite_query)
    g.add_node("web_search",      web_search)
    g.add_node("generate",        generate)

    g.set_entry_point("contextualize_query")
    g.add_edge("contextualize_query", "route_query")

    g.add_conditional_edges("route_query",     route_after_routing,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve",         "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading,
                            {"generate": "generate",
                             "rewrite_query": "rewrite_query",
                             "web_search": "web_search"})
    g.add_edge("rewrite_query",    "retrieve")
    g.add_edge("web_search",       "generate")
    g.add_edge("generate",         END)

    return g.compile()


# Singleton compiled graph
rag_agent = build_graph()


# ── Public interface ──────────────────────────────────────────────────────────

def ask(query: str, history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
    """
    Main entry point. Call from FastAPI or Streamlit.
    Returns {"answer": str, "sources": list, "rewrite_count": int}
    """
    messages = (history or [])[-MAX_HISTORY_MESSAGES:] + [HumanMessage(content=query)]
    initial_state: AgentState = {
        "messages":      messages,
        "query":         query,
        "rewrite_count": 0,
        "context":       [],
        "web_results":   [],
        "answer":        "",
    }
    final_state: Dict[str, Any] = rag_agent.invoke(initial_state)
    return {
        "answer":        final_state.get("answer", ""),
        "sources":       final_state.get("context", []),
        "rewrite_count": final_state.get("rewrite_count", 0),
        "used_web":      bool(final_state.get("web_results")),
        "diagnostics":   final_state.get("diagnostics", {}),
        "evidence_source": final_state.get("generation_evidence_source", "unknown"),
        "generation_context": final_state.get("generation_context", []),
        "route": (
            "web" if final_state.get("web_search_invoked")
            else final_state.get("_route", "unknown")
        ),
    }
