# Claude.md — Multi Tenant RAG

Instructions for Claude Code when working in this repository.

---

## Project Identity

This is a production-grade multi-agent RAG system. It is NOT a demo or tutorial project.
Every module is meant to be extended. Do not simplify unless explicitly asked.

---

## Stack

- **Python 3.11+**
- **FastAPI** — web server and API
- **ChromaDB** — vector store (per-client collections)
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, no API key needed
- **Anthropic Python SDK** — LLM calls (Claude claude-sonnet-4-20250514)
- **Pydantic v2** — schema validation throughout
- **Docker + docker-compose** — containerized deployment

---

## Code Conventions

### General
- Type-annotate every function signature
- Use `dataclasses` or Pydantic models for structured data — never raw dicts across module boundaries
- Log with the stdlib `logging` module — structured JSON logs preferred
- Every public function must have a docstring
- Raise specific exceptions — never bare `except: pass`

### Async
- All FastAPI route handlers are `async def`
- CPU-bound work (embedding, chunking) runs in `asyncio.run_in_executor` with a thread pool
- I/O-bound ChromaDB calls use async wrappers

### Agents
- All agents inherit from `BaseAgent`
- Each agent has a `run(context: AgentContext) -> AgentResult` method
- Agents must not call each other directly — only through the Orchestrator
- Agent state is immutable per run — pass context forward, return results

### RAG Pipeline
- Chunker output is always `list[Chunk]` where `Chunk` has `text`, `metadata`, `chunk_id`
- Embedder input is always `list[str]`, output is always `list[list[float]]`
- Retriever always returns `list[RetrievedChunk]` sorted by relevance score descending
- Context pruner always returns `list[RetrievedChunk]` with total tokens ≤ `MAX_CONTEXT_TOKENS`

### ChromaDB
- One collection per client: `f"client_{client_id}"`
- Metadata fields on every document: `doc_id`, `client_id`, `source`, `chunk_index`, `ingested_at`, `chunk_hash`
- Never delete and recreate collections — use upsert

### Evaluation
- Metrics module must be pure functions — no side effects
- Signals store is append-only — never mutate past signals
- Drift detection stores snapshots with timestamps — keep last 10

---

## File Locations

```
app/
  config.py          — all env vars and constants
  main.py            — FastAPI app factory
  agents/            — all agent classes
  rag/               — chunker, embedder, retriever, pruner, pipeline
  ingestion/         — intake, cleaner, deduplicator, anomaly_detector
  evaluation/        — metrics, drift, signals, pipeline
  validation/        — schema validator
  clients/           — client manager
  db/                — ChromaDB client singleton
  api/
    routes/          — FastAPI routers
    models.py        — request/response Pydantic models
static/              — frontend HTML/JS/CSS (served by FastAPI)
scripts/             — seed_data.py, run_eval.py
tests/               — pytest unit tests
```

---

## Common Tasks

### Add a new agent
1. Create `app/agents/my_agent.py`
2. Inherit from `BaseAgent`
3. Implement `run(context: AgentContext) -> AgentResult`
4. Register in `app/agents/orchestrator.py`

### Add a new evaluation metric
1. Add pure function to `app/evaluation/metrics.py`
2. Call it in `app/evaluation/pipeline.py`
3. Include in the `/eval/run` response model

### Add a new API route
1. Create router in `app/api/routes/my_route.py`
2. Include in `app/main.py` under appropriate prefix
3. Add request/response models to `app/api/models.py`

### Change embedding model
1. Update `EMBED_MODEL` in `.env`
2. Clear all ChromaDB collections (different vector dimensions)
3. Re-ingest all documents

---

## Pitfalls to Avoid

- Never import `app.agents.orchestrator` from within an agent — circular deps
- Never store raw embeddings in RAM for large corpora — always use ChromaDB
- Never pass entire ChromaDB collection to LLM — always prune first
- Never swallow `ValidationError` — surface it to the evaluation signal logger
- Do not use `time.sleep` in async code — use `asyncio.sleep`
- ChromaDB `query` returns distances not similarities — convert: `similarity = 1 - distance`

---

## Testing

```bash
pytest tests/ -v
```

- Unit tests for all evaluation metrics (deterministic, no LLM)
- Unit tests for chunker, deduplicator, anomaly detector
- Integration test for full query pipeline (requires `ANTHROPIC_API_KEY`)
- Use `pytest-asyncio` for async tests

---

## Running Locally (without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Start ChromaDB separately: chroma run --host 0.0.0.0 --port 8001
export CHROMA_PORT=8001
uvicorn app.main:app --reload --port 8080
```

---

## Docker

```bash
docker compose up --build
# App: http://localhost:8080
# ChromaDB: http://localhost:8000
```

---

## Seed Data

```bash
docker compose exec app python scripts/seed_data.py
```

This creates two demo clients (`acme_corp`, `techstart`) and ingests sample documents.
