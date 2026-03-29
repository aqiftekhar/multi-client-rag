# Multi Tenant RAG

Production-grade multi-agent RAG system with hierarchical retrieval, per-client ChromaDB isolation,
embedding drift detection, and automated CI/CD evaluation pipelines.

Built in pure Python. Runs in Docker. No toys.

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 2. Start everything
docker compose up --build

# 3. Seed demo data
docker compose exec app python scripts/seed_data.py

# 4. Open the UI
open http://localhost:8080
```

---

## Architecture

```
User → FastAPI → Orchestrator Agent
                    ├── RetrievalAgent
                    │     ├── Stage 1: Coarse semantic search (top-20)
                    │     ├── Stage 2: MMR rerank (top-5)
                    │     └── Context Pruner (token budget)
                    ├── LLM (Claude) with structured output prompt
                    ├── ValidationAgent
                    │     ├── JSON schema validation
                    │     ├── Source attribution check
                    │     └── Confidence gate (< 0.4 → retry)
                    ├── CorrectionAgent (on failure)
                    │     └── Query reformulation → retry loop
                    └── EvaluationAgent
                          ├── Implicit signal logging
                          └── Drift snapshot recording
```

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| Multi-client isolation | ChromaDB collection per client (`client_{id}`) |
| Hierarchical retrieval | Coarse (top-20) → MMR rerank (top-5) |
| Dynamic context pruning | Token budget (default 2000), greedily fills from ranked chunks |
| Hallucination mitigation | Constrained context + source attribution + validation |
| Embedding drift detection | Centroid cosine similarity, threshold=0.15 |
| Implicit failure signals | Agent retries, validation failures, task completions |
| CI/CD evaluation gate | Recall@k, Precision@k, MRR, NDCG — parallel per client |
| Intake pipeline | Clean → dedup (MinHash) → anomaly filter → chunk → embed → store |

---

## API Reference

| Endpoint | Description |
|----------|-------------|
| `POST /clients/` | Register a new client |
| `GET /clients/` | List all clients |
| `POST /ingest/` | Ingest a document |
| `DELETE /ingest/{client_id}` | Clear a client's corpus |
| `POST /query/` | Run a RAG query |
| `POST /eval/run` | Run CI/CD evaluation pipeline |
| `GET /eval/metrics/{client_id}` | Get signal summary |
| `GET /eval/drift/{client_id}` | Get drift report |
| `GET /eval/signals/{client_id}` | Get recent implicit signals |
| `GET /health` | Health check |

Full interactive docs: `http://localhost:8080/docs`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Your Anthropic API key |
| `CHROMA_HOST` | localhost | ChromaDB host |
| `CHROMA_PORT` | 8000 | ChromaDB port |
| `EMBED_MODEL` | all-MiniLM-L6-v2 | Local embedding model |
| `MAX_CONTEXT_TOKENS` | 2000 | Token budget for context window |
| `COARSE_K` | 20 | Stage-1 retrieval count |
| `FINE_K` | 5 | Stage-2 after MMR rerank |
| `DRIFT_THRESHOLD` | 0.15 | Cosine drift score for re-index trigger |
| `EVAL_RECALL_MIN` | 0.5 | CI/CD pass threshold (Recall@k) |
| `EVAL_NDCG_MIN` | 0.6 | CI/CD pass threshold (NDCG@k) |

---

## Running Tests

```bash
# All tests (no LLM or DB required)
pytest tests/ -v

# Specific modules
pytest tests/test_metrics.py -v
pytest tests/test_ingestion.py -v
pytest tests/test_drift.py -v
```

---

## CI/CD Evaluation

The `/eval/run` endpoint is the pre-deploy gate:

```bash
# Via HTTP
curl -X POST http://localhost:8080/eval/run \
  -H "Content-Type: application/json" \
  -d '{
    "client_queries": {
      "acme_corp": [
        {"query": "What is the leave policy?", "relevant_chunk_ids": []}
      ]
    },
    "k": 5
  }'

# Via CLI script
docker compose exec app python scripts/run_eval.py --clients acme_corp techstart --k 5
# Exit 0 = pass, 1 = fail
```

---

## Project Structure

```
app/
  config.py              — env vars and constants
  main.py                — FastAPI app factory
  agents/
    base.py              — BaseAgent, AgentContext, AgentResult
    orchestrator.py      — Main pipeline coordinator
    retrieval_agent.py   — Hierarchical retrieval + pruning
    validation_agent.py  — Schema + attribution + confidence
    correction_agent.py  — Query reformulation on failure
    evaluation_agent.py  — Signal logging + drift snapshots
  rag/
    chunker.py           — Sentence-aware overlapping chunker
    embedder.py          — sentence-transformers (local)
    retriever.py         — Two-stage MMR retrieval
    context_pruner.py    — Token-budget context trimmer
  ingestion/
    intake.py            — Orchestrates full intake pipeline
    cleaner.py           — HTML strip, encoding, normalisation
    deduplicator.py      — MinHash LSH + exact hash dedup
    anomaly_detector.py  — Length, character ratio, word count checks
  evaluation/
    metrics.py           — Recall@k, Precision@k, MRR, NDCG
    drift_detector.py    — Centroid cosine similarity over time
    signals.py           — Append-only implicit signal store
    pipeline.py          — Parallel CI/CD eval runner
  validation/
    schema_validator.py  — Pydantic + source attribution check
  clients/
    manager.py           — Client registry and per-client config
  db/
    chroma_client.py     — ChromaDB singleton + collection routing
  api/
    models.py            — Request/response Pydantic models
    routes/
      clients.py         — /clients endpoints
      ingest.py          — /ingest endpoints
      query.py           — /query endpoint
      eval.py            — /eval endpoints
static/
  index.html             — Single-page dashboard UI
scripts/
  seed_data.py           — Register demo clients + ingest sample docs
  run_eval.py            — CLI evaluation runner (CI-friendly)
tests/
  test_metrics.py        — Unit tests for all eval metrics
  test_ingestion.py      — Unit tests for cleaning, chunking, validation
  test_drift.py          — Unit tests for drift detection
```

---

## Design Decisions

**Standardise infrastructure, customise logic.** Chunking, embedding, and evaluation pipelines are
identical across all clients. Retrieval parameters (k, thresholds) and context budgets are
configurable per client.

**Implicit signals over explicit ratings.** Agent retries, validation failures, and downstream
task failures are logged as implicit signals. These are far more reliable as ground truth than
thumbs-up/down ratings, which are noisy and underused.

**Hierarchical retrieval over single-shot search.** Coarse retrieval casts a wide net; MMR
reranking balances relevance and diversity; the context pruner ensures the model only sees what
it genuinely needs — cutting noise and improving consistency.

**Evaluation in CI/CD.** The `/eval/run` endpoint is designed to run as a pre-deploy gate.
Recall@k and NDCG thresholds are configurable. Embedding drift is reported but treated as a
warning rather than a hard failure by default.
