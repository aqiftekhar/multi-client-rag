# Multi Tenant RAG — Project Specification

## Overview

A production-grade, multi-agent Retrieval-Augmented Generation (RAG) system built in Python,
designed to handle real-world enterprise data challenges: messy ingestion, embedding drift,
hallucination mitigation, multi-client isolation, and continuous evaluation in CI/CD pipelines.

This project directly implements every pattern discussed in the Multi Tenant AI technical conversation:
hierarchical retrieval, dynamic context pruning, per-client pipelines with reusable infrastructure,
embedding drift detection, implicit failure signals, and automated evaluation before deploys.

---

## Goals

- Multi-client RAG with isolated ChromaDB collections per client
- Hierarchical retrieval (coarse → fine) with dynamic context pruning
- Production ingestion pipeline: cleaning, dedup, anomaly detection, metadata tagging
- Structured output validation + hallucination mitigation via source attribution
- Multi-agent orchestration: Orchestrator, Retrieval, Validation, Evaluation, Correction agents
- Automated evaluation pipeline: Recall@k, Precision@k, MRR, NDCG, task completion rate
- Embedding drift detection with automatic re-index triggers
- Implicit failure signals from agent retries and downstream task failures
- CI/CD evaluation endpoint for pre-deploy regression checks
- Web UI for querying, ingesting documents, and viewing evaluation dashboards
- Fully Dockerized, runs with `docker compose up`

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                   │
│   /ingest   /query   /eval   /clients   /dashboard      │
└────────────────────────┬────────────────────────────────┘
                         │
           ┌─────────────▼─────────────┐
           │     Orchestrator Agent    │
           │  Routes, coordinates,     │
           │  retries, logs signals    │
           └──┬──────────┬─────────────┘
              │          │
    ┌─────────▼──┐   ┌───▼──────────────┐
    │ Retrieval  │   │  Validation      │
    │   Agent    │   │    Agent         │
    │            │   │  Schema + source │
    │ Hierarchic │   │  attribution     │
    │ retrieval  │   │  hallucination   │
    └─────┬──────┘   └───┬──────────────┘
          │              │
    ┌─────▼──────┐   ┌───▼──────────────┐
    │ Correction │   │  Evaluation      │
    │   Agent    │   │    Agent         │
    │ Retry on   │   │  Metrics, drift, │
    │ failure    │   │  CI/CD pipeline  │
    └────────────┘   └──────────────────┘
              │
    ┌─────────▼──────────────────────────┐
    │         ChromaDB (per-client)      │
    │  client_A  |  client_B  |  ...     │
    └────────────────────────────────────┘
```

---

## Module Breakdown

### `app/ingestion/`

| Module | Responsibility |
|--------|---------------|
| `intake.py` | Orchestrates full intake pipeline per document |
| `cleaner.py` | HTML strip, whitespace norm, encoding fix |
| `deduplicator.py` | MinHash + exact hash dedup across client corpus |
| `anomaly_detector.py` | Length outliers, language detection, structural anomalies |

### `app/rag/`

| Module | Responsibility |
|--------|---------------|
| `chunker.py` | Sentence-aware chunking with configurable overlap |
| `embedder.py` | Sentence-transformers (`all-MiniLM-L6-v2`) with batch support |
| `retriever.py` | Two-stage hierarchical retrieval (coarse→fine) |
| `context_pruner.py` | Token-budget pruner — feeds model only what it needs |
| `pipeline.py` | End-to-end RAG pipeline composing all rag modules |

### `app/agents/`

| Agent | Responsibility |
|-------|---------------|
| `orchestrator.py` | Main controller: routes query, coordinates agents, logs retries |
| `retrieval_agent.py` | Calls retriever, reranks, prunes context |
| `validation_agent.py` | Schema validates LLM output, checks source attribution |
| `evaluation_agent.py` | Computes metrics, detects drift, fires re-index triggers |
| `correction_agent.py` | Reformulates query on failure, handles fallback |

### `app/evaluation/`

| Module | Responsibility |
|--------|---------------|
| `metrics.py` | Recall@k, Precision@k, MRR, NDCG implementations |
| `drift_detector.py` | Cosine similarity monitoring across embedding snapshots |
| `signals.py` | Implicit signal collector (failures, retries, corrections) |
| `pipeline.py` | CI/CD evaluation runner — parallel per-client eval |

### `app/clients/`

| Module | Responsibility |
|--------|---------------|
| `manager.py` | Client registry, per-client config, collection routing |

### `app/validation/`

| Module | Responsibility |
|--------|---------------|
| `schema_validator.py` | Pydantic-based structured output validation |

### `app/db/`

| Module | Responsibility |
|--------|---------------|
| `chroma_client.py` | ChromaDB singleton, collection management |

---

## Data Flow

### Ingestion
```
Raw Document
  → Cleaner (strip, normalize)
  → Deduplicator (hash check)
  → Anomaly Detector (flag outliers)
  → Chunker (sentence-aware, overlapping)
  → Embedder (sentence-transformers)
  → ChromaDB (client-isolated collection)
```

### Query
```
User Query
  → Orchestrator Agent
    → Retrieval Agent
        → Coarse retrieval (top-20, client collection)
        → Fine retrieval (MMR rerank, top-5)
        → Context Pruner (token budget)
    → LLM (Claude) with pruned context
    → Validation Agent
        → Schema validation
        → Source attribution check
        → Confidence scoring
    → [if fail] Correction Agent
        → Query reformulation → retry
    → Evaluation Agent
        → Log signal (success / fail / retry)
        → Rank metrics
  → Response to user
```

### Evaluation (CI/CD)
```
POST /eval/run
  → Load representative queries per client
  → Parallel eval pipeline per client
    → Retrieve + compare expected vs actual chunks
    → Compute Recall@k, Precision@k, MRR, NDCG
    → Check task completion rate
    → Check embedding drift score
  → Aggregate results
  → Pass / Fail decision
  → Return detailed report
```

---

## Key Design Decisions

### 1. Standardize Infrastructure, Customize Logic
- Chunking + embedding pipeline is identical across all clients
- Retrieval config (k, thresholds, collection name) is per-client
- Evaluation pipeline runs in parallel per client

### 2. Hierarchical Retrieval
- Stage 1: Broad semantic search (top-20, lower threshold)
- Stage 2: MMR (Maximal Marginal Relevance) rerank to top-5
- Stage 3: Context pruner applies token budget before LLM call

### 3. Dynamic Context Pruning
- Model never gets more than `MAX_CONTEXT_TOKENS` (default: 2000)
- Chunks scored and trimmed — only the most relevant make the cut
- Source attribution: every chunk tagged with `doc_id`, `source`, `page`

### 4. Hallucination Mitigation
- Constrained context window (pruner)
- Source attribution enforced in prompt
- Validation agent checks that claims in output are grounded in retrieved chunks
- Confidence score < threshold triggers fallback to correction agent

### 5. Embedding Drift Detection
- Snapshot of centroid embedding taken after each re-index
- Cosine similarity between current centroid and stored snapshot
- If drift > threshold → trigger re-index warning
- Periodic background task re-indexes when drift is detected

### 6. Implicit Failure Signals
- Every agent retry is logged with reason
- Downstream task failures (validation fail, correction triggered) written to signals store
- Signals aggregated in evaluation pipeline as ground truth

### 7. Evaluation in CI/CD
- `POST /eval/run` runs full evaluation before deploys
- Test queries and expected chunk IDs stored per client
- Hard fail if Recall@1 < 0.5 or NDCG < 0.6

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/clients/` | Register new client |
| GET | `/clients/` | List all clients |
| POST | `/ingest/` | Ingest document for a client |
| POST | `/query/` | Query RAG system |
| GET | `/eval/metrics/{client_id}` | Get latest metrics for client |
| POST | `/eval/run` | Run CI/CD evaluation pipeline |
| GET | `/eval/drift/{client_id}` | Get embedding drift report |
| GET | `/eval/signals/{client_id}` | Get implicit failure signals |
| DELETE | `/ingest/{client_id}` | Clear client corpus |

---

## Environment Variables

```
ANTHROPIC_API_KEY=         # Required: Claude API key
CHROMA_HOST=chromadb       # ChromaDB host (docker service name)
CHROMA_PORT=8000           # ChromaDB port
EMBED_MODEL=all-MiniLM-L6-v2
MAX_CONTEXT_TOKENS=2000
COARSE_K=20                # Stage-1 retrieval count
FINE_K=5                   # Stage-2 after rerank
DRIFT_THRESHOLD=0.15       # Cosine distance threshold for re-index trigger
EVAL_RECALL_MIN=0.5        # CI/CD pass threshold
EVAL_NDCG_MIN=0.6          # CI/CD pass threshold
```

---

## Non-Goals (v1)

- No auth/JWT (can be added)
- No streaming responses
- No fine-tuning or adapter layers
- No graph-based retrieval
- Single-region deployment only
