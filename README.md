# Enterprise Agentic RAG Platform

> Hybrid enterprise document intelligence using dense-sparse retrieval, reranking, corrective RAG, and grounded generation.

An end-to-end **Agentic RAG platform** for enterprise document Q&A using **BGE-M3, Qdrant, BM25, RRF, Cross-Encoder reranking, LangGraph, OpenAI, and RAGAS**.

### Key Capabilities

* Hybrid dense + lexical retrieval
* Cross-Encoder reranking and relevance grading
* Corrective RAG with query rewriting
* Tavily web fallback for unsupported questions
* Grounded answers with document/page evidence
* Conversational follow-up questions
* Retrieval and agent execution tracing
* 30-question RAGAS evaluation benchmark
* Streamlit frontend + FastAPI backend
  
<img width="1825" height="802" alt="Screenshot 2026-10-05 194745" src="https://github.com/user-attachments/assets/44605f6b-1fea-418b-9adb-57150bf84453" />
<img width="1785" height="847" alt="Screenshot 2026-10-05 194902" src="https://github.com/user-attachments/assets/b018d6bc-0d6e-4a61-8bfd-4f49a5e03377" />
<img width="1763" height="766" alt="Screenshot 2026-10-05 195021" src="https://github.com/user-attachments/assets/d87a2b08-d535-4838-972e-665ceca53ee1" />
<img width="1791" height="688" alt="Screenshot 2026-10-05 200455" src="https://github.com/user-attachments/assets/5d29698b-c50e-4a1a-a2fd-4d7f89baeb64" />
<img width="1671" height="757" alt="Screenshot 2026-10-05 200509" src="https://github.com/user-attachments/assets/2b52665e-817c-4957-9f48-d30e59ea51da" />

---

## Architecture

```text
                    Streamlit
                       │
                    FastAPI
                       │
                   LangGraph
                       │
              ┌────────┴────────┐
              │                 │
        BGE-M3 + Qdrant       BM25
        Dense Retrieval    Lexical Retrieval
              │                 │
              └────────┬────────┘
                       │
                  RRF Fusion
                       │
                Cross-Encoder
                  Reranking
                       │
                Relevance Grader
                       │
             ┌─────────┴─────────┐
             │                   │
         Relevant            Insufficient
             │                   │
             │              Query Rewrite
             │                   │
             │              Re-retrieval
             │                   │
             │              Web Fallback
             │                   │
             └─────────┬─────────┘
                       │
                OpenAI Responses API
                       │
                Grounded Answer
                       │
             Evidence + Trace
```

---

## RAG Pipeline

### 1. Document Ingestion

PDFs are loaded, chunked, embedded using **BGE-M3**, and indexed into **Qdrant**.

Current evaluation corpus:

```text
data/
├── encryption_google_cloud.pdf
├── google_cloud_security.pdf
└── microsoft_annual_report_2025.pdf
```

**3 PDFs · 510 chunks**

---

### 2. Dense Retrieval — BGE-M3 + Qdrant

**BAAI/bge-m3** converts queries and document chunks into dense vectors.

Qdrant performs semantic similarity search over these embeddings.

This handles conceptually similar queries even when the wording differs.

---

### 3. Lexical Retrieval — BM25

BM25 provides exact/lexical matching and is useful for:

* technical terminology
* acronyms
* product names
* protocol names
* exact phrases

---

### 4. RRF Fusion

Dense and BM25 retrieval produce independent rankings.

**Reciprocal Rank Fusion (RRF)** combines them into a unified candidate ranking.

```text
BGE-M3 / Qdrant
       +
BM25
       ↓
     RRF
       ↓
Unified candidates
```

---

### 5. Cross-Encoder Reranking

The retrieved candidates are reranked using a **Cross-Encoder**.

Instead of independently embedding the query and document, the model evaluates:

```text
Query + Document
       ↓
Relevance Score
```

This improves the ordering of the most relevant evidence.

---

### 6. Relevance Grading

The selected candidates are passed through a relevance grader.

The grader distinguishes between:

* direct evidence
* partial evidence for multi-part questions
* merely topical information

Only useful evidence is passed toward generation.

---

### 7. Corrective RAG

When retrieved evidence is insufficient:

```text
Initial Retrieval
       ↓
Relevance Grading
       ↓
Insufficient Evidence
       ↓
Query Rewrite
       ↓
Retrieval Again
       ↓
Grading
```

The system can perform multiple corrective retrieval attempts.

---

### 8. Web Fallback

If the indexed corpus still cannot answer the question, **Tavily** can be used as an explicit web fallback.

```text
Documents
   ↓
Rewrite / Re-retrieve
   ↓
Still insufficient
   ↓
Tavily
   ↓
Web Evidence
```

The final response records whether evidence came from the documents or web.

---

## Grounded Generation

Accepted evidence is passed to the **OpenAI Responses API**.

The generation layer is instructed to answer from the supplied evidence and avoid unsupported claims.

The application therefore separates:

```text
Retrieval → Evidence Selection → Generation
```

rather than relying on the LLM alone.

---

## Conversational RAG

Follow-up questions use recent conversation context for query interpretation.

Example:

```text
What are Microsoft's major revenue growth drivers?

        ↓

What about Copilot?

        ↓

How does that relate to the previous answer?
```

Conversation history is used for contextualization, while factual claims still require retrieved evidence.

**New Chat** clears the active conversation without clearing the knowledge base.

---

# LangGraph Agent

LangGraph orchestrates the workflow:

```text
Route
  ↓
Retrieve
  ↓
Grade
  ↓
Rewrite if required
  ↓
Retrieve again
  ↓
Web fallback if required
  ↓
Generate
```

This makes corrective behavior explicit rather than hiding everything inside one LLM prompt.

---

# Evaluation

The system uses a dedicated **30-question RAGAS benchmark** covering:

* exact factual questions
* semantic retrieval
* terminology/hybrid retrieval
* multi-hop questions
* cross-document questions
* unsupported questions

Difficulty:

```text
Easy    10
Medium  13
Hard     7
```

### RAGAS Results

| Metric            |      Score |
| ----------------- | ---------: |
| Faithfulness      | **0.8531** |
| Answer Relevancy  | **0.8659** |
| Context Precision | **0.8925** |
| Context Recall    | **0.9056** |
| **Average**       | **0.8793** |

**30/30 evaluation questions completed.**

The benchmark is kept as a fixed evaluation point for the current implementation.

---

# Observability

The application exposes retrieval diagnostics including:

```text
Dense candidates
BM25 candidates
RRF candidates
Reranked candidates
Selected evidence
Relevant evidence
```

Evidence can include:

* source document
* page
* chunk
* RRF rank
* Cross-Encoder rank
* Cross-Encoder score
* grader decision

This makes it possible to investigate whether a failure occurred during retrieval, reranking, grading, correction, or generation.

---

# Frontend

The Streamlit application contains:

### Ask

* conversational Q&A
* answers
* sources
* evidence
* agent trace

### Knowledge Base

* PDF upload
* indexing
* Qdrant status

### Retrieval Trace

* retrieval attempts
* reranking
* grading
* rewrites
* web fallback

### Evaluation

* RAGAS metrics
* per-question results

### System

* LLM
* embeddings
* vector store
* retrieval stack
* runtime health

---

# Backend

```text
Streamlit
    ↓
FastAPI
    ↓
LangGraph
    ↓
RAG Pipeline
```

FastAPI handles:

* PDF ingestion
* query execution
* agent execution
* evidence
* diagnostics
* health information

The frontend does not duplicate the retrieval or agent logic.

---

# Project Structure

```text
Enterprise-Agentic-RAG-Platform/
│
├── app/
│   ├── agent.py
│   ├── api.py
│   ├── frontend.py
│   ├── main.py
│   ├── llm_provider.py
│   └── runtime_info.py
│
├── ingestion/
├── vectorstore/
├── embeddings/
├── evaluation/
├── tests/
├── data/
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Layer            | Technology           |
| ---------------- | -------------------- |
| Frontend         | Streamlit            |
| Backend          | FastAPI              |
| Orchestration    | LangGraph            |
| LLM              | OpenAI Responses API |
| Embeddings       | BAAI/bge-m3          |
| Vector DB        | Qdrant               |
| Sparse Retrieval | BM25                 |
| Fusion           | RRF                  |
| Reranking        | Cross-Encoder        |
| Web Fallback     | Tavily               |
| Evaluation       | RAGAS                |
| Testing          | Pytest               |
| Containers       | Docker               |
| Language         | Python               |

---

# Running Locally

### 1. Configure environment

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.6-luna

TAVILY_API_KEY=your_tavily_key

VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
```

### 2. Start Qdrant

```powershell
docker start rag-qdrant
```

### 3. Ingest documents

```powershell
python -m ingestion.ingest
```

### 4. Start FastAPI

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

### 5. Start Streamlit

```powershell
streamlit run app/main.py
```

---

# Testing

The project contains regression tests covering:

* agent routing
* conversation contextualization
* relevance grading
* retrieval diagnostics
* LLM integration
* RAGAS configuration
* evaluation artifacts
* frontend behavior
* evidence rendering
* system/health UI

Core development validation reached:

```text
44 tests passed
```


# Current Status

```text
Hybrid Retrieval             ✅
BGE-M3 + Qdrant              ✅
BM25 + RRF                   ✅
Cross-Encoder                ✅
Relevance Grading            ✅
Corrective RAG               ✅
Web Fallback                 ✅
Grounded Generation          ✅
Conversational RAG           ✅
Retrieval Observability      ✅
RAGAS Evaluation             ✅
Streamlit Frontend           ✅
FastAPI Backend              ✅
```

**30-question benchmark: 0.8793 average RAGAS score**

**Context Recall: 0.9056**

---

## Core Idea

Retrieve broadly → fuse → rerank → grade → correct when necessary → generate from evidence → evaluate the entire pipeline.

> **Retrieve broadly → fuse → rerank → grade → correct when necessary → generate from evidence → evaluate the entire pipeline.**
