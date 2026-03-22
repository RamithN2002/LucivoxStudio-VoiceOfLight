<div align="center">

# LucivoxStudio — VoiceOfLight

**An offline, production-grade Retrieval-Augmented Generation (RAG) platform**
built with FastAPI, Next.js, ChromaDB, and Ollama — 100% private, no API keys required.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-Frontend-black?style=flat-square&logo=next.js)](https://nextjs.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=flat-square)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple?style=flat-square)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What is LucivoxStudio?

**Luci** = Light. **Vox** = Voice.

LucivoxStudio is a fully offline RAG product that lets you upload your own documents and have intelligent, context-aware conversations with them — powered entirely by local LLMs running on your machine. No OpenAI. No cloud. No data leaves your system.

It goes far beyond basic RAG — combining hybrid search, semantic chunking, cross-encoder reranking, and conversation-aware retrieval to deliver answers that are accurate, grounded, and context-rich.

---

## Features

### Core RAG Pipeline
- **Semantic Chunking** — splits documents by meaning, not fixed token size
- **SentenceTransformer Embeddings** — local dense vector embeddings
- **ChromaDB Vector Store** — fast, persistent vector storage per user
- **Hybrid Search** — combines Vector Search + BM25 keyword search + Query Expansion, fused with Reciprocal Rank Fusion (RRF)
- **Selective CrossEncoder Reranking** — re-scores top results for maximum relevance
- **Contextual Chunk Headers** — every chunk carries document context so the LLM always knows where information comes from
- **Answer Grounding Check** — verifies the final answer is actually supported by retrieved chunks
- **Context Compression** — trims retrieved context to only what is relevant before sending to LLM

### Conversation Intelligence
- **Conversation-Aware Retrieval** — rewrites queries based on chat history so follow-up questions work naturally
- **Query Expansion** — expands short queries into richer search terms for better recall
- **Query Rewriting** — reformulates ambiguous or conversational questions into precise retrieval queries

### Auth & Multi-User
- **JWT Authentication** — secure login with bcrypt password hashing
- **Per-User Document Isolation** — each user has their own ChromaDB collection, documents never mix
- **Multi-Document Selection** — users can select which uploaded documents to search across

### Frontend
- **Animated Splash Screen** — dark-to-light L mark logo transition
- **Real-time Token Streaming** — answers stream word by word like ChatGPT
- **Typing Indicator** — live feedback while the model is generating
- **Responsive Chat UI** — built with Next.js, Syne + DM Sans fonts, indigo + pink color scheme

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python |
| Frontend | Next.js + TypeScript |
| LLM | Ollama — llama3.1:8b-instruct-q4_K_M (fully local) |
| Embeddings | SentenceTransformers |
| Vector Store | ChromaDB |
| Auth | JWT + bcrypt |
| Reranking | CrossEncoder (sentence-transformers) |
| BM25 Search | rank-bm25 |

---

## Project Structure

```
LucivoxStudio/
├── main.py                    # FastAPI app entry point
├── auth/
│   ├── auth_utils.py          # JWT token logic
│   ├── auth_models.py         # User models
│   ├── auth_routes.py         # Login / register endpoints
│   └── auth_deps.py           # Auth dependencies
├── rag/
│   ├── rag_pipeline.py        # Main RAG orchestration
│   ├── parent_retrieval.py    # Parent document retrieval
│   ├── parent_document_ingestion.py
│   └── parent_store.py
├── routes/
│   ├── chat_routes.py         # Chat endpoints with streaming
│   └── upload_routes.py       # Document upload endpoints
├── utils/
│   ├── query_router.py        # Routes queries to different strategies
│   ├── query_rewriter.py      # Rewrites conversational queries
│   ├── query_expansion.py     # Expands queries for better recall
│   ├── rerank.py              # CrossEncoder reranking
│   ├── bm25.py                # BM25 keyword search
│   ├── bm25_store.py          # BM25 index persistence
│   ├── memory.py              # Conversation memory
│   ├── context_compression.py # Trims context before LLM
│   └── semantic_chunker.py    # Semantic document chunking
├── embeddings/                # Embedding model loader
├── aimodel/
│   └── llamamodel.py          # Ollama LLM wrapper
├── chroma/                    # ChromaDB persistence
└── models_request/
    └── request_models.py      # Pydantic request/response models
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai) installed and running
- 16GB+ RAM recommended

### 1. Clone the repository

```bash
git clone https://github.com/RamithN2002/LucivoxStudio-VoiceOfLight.git
cd LucivoxStudio-VoiceOfLight
```

### 2. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Pull the LLM model

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Start the backend

```bash
python main.py
# API running at http://localhost:8000
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
# App running at http://localhost:3000
```

---

## How It Works

```
User uploads document
        ↓
Semantic Chunking — splits by meaning
        ↓
SentenceTransformer embeds each chunk
        ↓
Stored in ChromaDB (per-user collection)
        ↓
User asks a question
        ↓
Query Rewriter — makes conversational queries precise
        ↓
Query Expansion — generates richer search terms
        ↓
Hybrid Search — Vector + BM25 + RRF fusion
        ↓
CrossEncoder Reranking — picks the most relevant chunks
        ↓
Context Compression — trims to only what matters
        ↓
llama3.1 generates grounded answer
        ↓
Answer Grounding Check — verifies answer is supported
        ↓
Streamed token by token to the user
```

---

## Hardware Used

Built and tested on:
- CPU: 8-core processor
- RAM: 32GB
- OS: Ubuntu
- LLM runs fully locally via Ollama — no GPU required

---

## Roadmap

- [ ] LLM Re-ranking — use Ollama to pick best chunks from BM25 top 10
- [ ] Sentence Window Retrieval — retrieve small chunks, return larger context window
- [ ] Document Summary Index — search document summaries first, then dive into chunks
- [ ] Query Routing — different retrieval strategies for different question types

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with purpose. Runs offline. No data leaves your machine.

**LucivoxStudio — Light through every word.**

</div>
