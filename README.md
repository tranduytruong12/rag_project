# RAG Project

A **production-ready scaffold** for a Retrieval-Augmented Generation (RAG) system — built for interns and junior engineers to learn, extend, and eventually ship.

> **Status**: 🏗️ Foundation / Scaffold — pipeline wired end-to-end with stubs. No real LLM or vector DB calls yet.

---

## Table of Contents

- [Goal](#goal)
- [Architecture](#architecture)
- [Stack](#stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running Tests](#running-tests)
- [API Reference](#api-reference)
- [Development Workflow](#development-workflow)
- [Next Steps (TODO)](#next-steps-todo)
- [Design Decisions](#design-decisions)

---

## Goal

Build a modular RAG system that can:
1. **Ingest** documents (PDF, TXT, URLs) and index them in a vector store.
2. **Retrieve** the most relevant chunks for a user query.
3. **Generate** a grounded answer using an LLM.
4. **Serve** everything via a REST API.
5. **Evaluate** quality with standard RAG metrics.

---

## Architecture

```
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  Ingestion  │───▶│ Chunking │───▶│Embedding │───▶│ Vector Store │
│  (loaders)  │    │(splitters│    │ (OpenAI) │    │  (ChromaDB)  │
└─────────────┘    └──────────┘    └──────────┘    └──────────────┘
                                                          │
                         ┌────────────────────────────────┘
                         ▼
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  User Query │───▶│Retriever │───▶│ Reranker │───▶│  Generator   │
│   (API)     │    │(similarity│   │(optional)│    │  (OpenAI)    │
└─────────────┘    └──────────┘    └──────────┘    └──────────────┘
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │ RAG Response │
                                                   └──────────────┘
```

**Layers:**

| Layer | Modules | Responsibility |
|---|---|---|
| Ingestion | `ingestion/` | Load raw docs from files, URLs, DBs |
| Chunking | `chunking/` | Split docs into overlapping text windows |
| Indexing | `embedding/` + `vector_store/` | Embed & store chunks |
| Retrieval | `retriever/` + `reranker/` | Find & rank relevant chunks |
| Generation | `generator/` + `prompts/` | LLM-powered answer synthesis |
| Serving | `api/` | FastAPI REST endpoints |
| Evaluation | `evaluation/` | Quality metrics (faithfulness, relevance) |

---

## Stack

| Concern | Choice | Reason |
|---|---|---|
| API framework | **FastAPI** | Async, auto-docs, Pythonic DI |
| Schema / validation | **Pydantic v2** | Fast, strict, excellent DX |
| Config | **pydantic-settings** | Type-safe env-var loading |
| Testing | **pytest** | Industry standard, rich plugin ecosystem |
| Lint + format | **ruff** | Replaces flake8 + isort + black in one tool |
| Type checking | **mypy** | Catch bugs before runtime |
| Logging | **structlog** | Structured JSON logs in prod, pretty console in dev |
| Vector DB (default) | **ChromaDB** | Simple local setup; swap to Qdrant/Pinecone later |
| LLM (default) | **OpenAI** | Most widely used; abstraction allows any backend |

---

## Project Structure

```
rag_project/
├── src/rag/
│   ├── config/          # Settings loaded from .env via Pydantic
│   ├── schemas/         # Domain models: Document, Chunk, Query, RAGResponse
│   ├── ingestion/       # Loaders: file, web (stub)
│   ├── chunking/        # Splitters: fixed-size, recursive (stub)
│   ├── embedding/       # Embedder: OpenAI (stub)
│   ├── vector_store/    # ChromaDB client (stub)
│   ├── retriever/       # Similarity retriever
│   ├── reranker/        # Cross-encoder reranker (stub)
│   ├── generator/       # OpenAI chat completion (stub)
│   ├── prompts/         # Prompt templates & builder
│   ├── pipeline/        # Orchestrators: ingestion & RAG query
│   ├── evaluation/      # Quality metrics (stubs)
│   ├── api/             # FastAPI app, routers, DI
│   └── utils/           # Logging, helpers
├── tests/               # pytest test suite (skeleton, fully passing)
├── scripts/             # CLI: ingest.py, query.py
├── pyproject.toml       # Single source of truth for deps + tools
├── .env.example         # All required env vars documented
└── .pre-commit-config.yaml
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd rag_project

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3. Start the API server

```bash
uvicorn src.rag.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

### 4. Ingest a document (CLI)

```bash
python scripts/ingest.py --source path/to/your/doc.txt
```

### 5. Ask a question (CLI)

```bash
python scripts/query.py --question "What is retrieval-augmented generation?"
```

---

## Running Tests

```bash
# Run all tests with coverage
pytest

# Run a specific module
pytest tests/test_chunking.py -v

# Run without coverage (faster)
pytest --no-cov
```

All tests pass on the scaffold because stubs return safe defaults.
**Coverage threshold is 0%** until real implementations are added — increase it incrementally.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe (checks dependencies) |
| `POST` | `/api/v1/ingest/file` | Ingest a single file |
| `POST` | `/api/v1/ingest/batch` | Ingest multiple files |
| `POST` | `/api/v1/query` | Single-turn Q&A |
| `POST` | `/api/v1/chat` | Multi-turn chat (stub) |

Full schema available at `/docs` (Swagger) or `/redoc`.

---

## Development Workflow

```bash
# Install pre-commit hooks (run once)
pre-commit install

# Lint & format manually
ruff check src/ tests/ --fix
ruff format src/ tests/

# Type check
mypy src/

# Run tests with watch mode (requires pytest-watch)
ptw tests/
```

---

## Next Steps (TODO)

Ordered by priority for a typical intern sprint:

### Sprint 1 — Connect real backends
- [x] `embedding/openai_embedder.py` — wire up `openai.embeddings.create()`
- [x] `vector_store/chroma_store.py` — install `chromadb`, implement `add_chunks()` + `search()`
- [x] `generator/openai_generator.py` — wire up `openai.chat.completions.create()`
- [x] Update `.env` with a real `OPENAI_API_KEY` and test end-to-end

### Sprint 2 — Improve chunking & retrieval
- [x] `chunking/text_splitter.py` — implement true recursive separator logic
- [x] `retriever/` — add hybrid search (BM25 + dense)
- [x] `reranker/cross_encoder.py` — install `sentence-transformers`, implement `rerank()`

### Sprint 3 — Better ingestion
- [ ] `ingestion/file_loader.py` — add PDF loader (pypdf/pymupdf)
- [ ] `ingestion/web_loader.py` — implement with httpx + trafilatura
- [ ] Add document deduplication by content hash

### Sprint 4 — Evaluation
- [ ] `evaluation/metrics.py` — implement LLM-judge scoring for faithfulness + relevance
- [ ] Build eval dataset in `data/eval/dataset.json`
- [ ] Add `scripts/evaluate.py` CLI

### Sprint 5 — Production hardening
- [ ] Add retry + exponential back-off (tenacity) to all API clients
- [ ] `api/` — add API-key authentication header
- [ ] Add background task processing for large ingestion jobs
- [ ] Raise test coverage threshold to ≥ 70%
- [ ] Add Docker + docker-compose setup
- [ ] Add GitHub Actions CI workflow

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Package manager | `pyproject.toml` (setuptools) | Single file for deps + tool config; no extra tooling needed |
| Default vector DB | ChromaDB | Zero external service required; runs fully local for development |
| Default LLM | OpenAI | Widest ecosystem and documentation; abstraction makes swapping easy |
| Logging | structlog | JSON-structured in prod, human-readable in dev; zero stdlib friction |
| Abstract classes | `ABC` + `abstractmethod` | Enforces consistent interfaces; makes dependency injection and mocking trivial |
| No Celery/task queue | — | Out of scope for intern sprint; noted as Sprint 5 item |
| No frontend | — | API-first; UI can be added independently (e.g. Streamlit, Next.js) |

---

*Built with ❤️ as a learning scaffold. Questions? Check the TODO comments inside each module.*
