"""Structured hallucination log — turns every hallucination event into
a learning record that can be analyzed to improve retrieval and prompts.

Every record captures:
  - what the user asked
  - what was retrieved
  - what the model said
  - why it was flagged
  - what happened during recovery

Over time this dataset tells you:
  - which queries consistently hallucinate → improve chunking/retrieval for those topics
  - which documents produce bad retrievals → re-ingest or re-chunk those docs
  - which prompts work better in strict mode → tune the system prompt
  - what recovery strategies actually work → adjust retry logic
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)

_LOG_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "hallucination_log.jsonl"
)

# In-memory store for fast querying — also written to disk for persistence
_records: list[dict] = []


@dataclass
class HallucinationRecord:
    """One complete hallucination event with full context for offline analysis."""

    run_id: str
    client_id: str
    timestamp: str

    # What the user asked
    original_query: str
    final_query: str                  # may differ after reformulation

    # What was retrieved
    retrieved_sources: list[str]      # filenames of retrieved chunks
    retrieved_chunk_count: int
    top_chunk_preview: str            # first 200 chars of top chunk

    # What the model said
    llm_raw_output: str               # raw output before validation
    faithfulness_score: float         # cross-encoder score — lower = worse

    # Why it was flagged
    hallucination_type: str           # "content_not_grounded" | "citation_mismatch"
    unmatched_citations: list[str]    # citations model made that don't exist

    # Recovery outcome
    retry_count: int
    recovery_strategy: str            # "hallucination_recovery" | "safe_fallback" | "clarification"
    final_answer: str                 # what was ultimately returned to user
    recovery_succeeded: bool          # did we get a valid answer after recovery?

    # Analysis helpers
    query_length: int = 0
    context_length: int = 0

    def __post_init__(self):
        self.query_length = len(self.original_query)
        self.context_length = len(self.top_chunk_preview)


def log_hallucination(record: HallucinationRecord) -> None:
    """Write a hallucination record to memory and append to disk log."""
    data = asdict(record)
    _records.append(data)

    try:
        os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
        logger.debug("Hallucination logged: run_id=%s", record.run_id)
    except Exception as exc:
        logger.error("Failed to write hallucination log: %s", exc)


def load_from_disk() -> None:
    """Load existing hallucination log from disk on startup."""
    global _records
    if not os.path.exists(_LOG_FILE):
        return
    try:
        with open(_LOG_FILE) as f:
            _records = [json.loads(line) for line in f if line.strip()]
        logger.info("Loaded %d hallucination records from disk.", len(_records))
    except Exception as exc:
        logger.error("Failed to load hallucination log: %s", exc)


def get_records(client_id: str | None = None, limit: int = 100) -> list[dict]:
    """Return recent hallucination records, optionally filtered by client."""
    records = _records
    if client_id:
        records = [r for r in records if r.get("client_id") == client_id]
    return records[-limit:]


def analyze(client_id: str | None = None) -> dict:
    """Analyze hallucination patterns to surface actionable improvements."""
    records = get_records(client_id=client_id, limit=1000)

    if not records:
        return {
            "total_hallucinations": 0,
            "message": "No hallucination records yet.",
        }

    total = len(records)
    recovered = sum(1 for r in records if r.get("recovery_succeeded"))
    unrecovered = total - recovered

    # ── Query term analysis ───────────────────────────────────────────────────
    # Use original_query only — not the reformulated strict-mode version
    # Filter out stop words and system-injected terms
    _STOP_WORDS = {
        "the", "is", "in", "of", "and", "to", "a", "that", "with",
        "for", "this", "are", "was", "what", "how", "does", "from",
        # System-injected strict mode terms — must be excluded
        "only", "information", "explicitly", "stated", "provided",
        "context", "use", "please", "specific", "cite", "source",
        "documents", "about", "tell", "me", "give", "describe",
        "explain", "using", "based", "document",
    }

    query_words: dict[str, int] = defaultdict(int)
    for r in records:
        # Always use original_query — reformulated queries contain system noise
        query = r.get("original_query") or r.get("final_query", "")
        for word in query.lower().split():
            clean = word.strip("?.!,")
            if len(clean) > 3 and clean not in _STOP_WORDS:
                query_words[clean] += 1
    common_query_terms = sorted(
        query_words.items(), key=lambda x: x[1], reverse=True
    )[:10]

    # ── Source analysis — distinguish blame from co-occurrence ────────────────
    # A source is "problematic" only if it was the TOP retrieved source
    # AND the answer was not recovered. Being retrieved during a failed
    # attempt does not mean the source caused the failure.
    source_unrecovered: dict[str, int] = defaultdict(int)
    source_total: dict[str, int] = defaultdict(int)

    for r in records:
        sources = r.get("retrieved_sources", [])
        if not sources:
            continue
        # Only count the primary source (first retrieved)
        primary = sources[0]
        source_total[primary] += 1
        if not r.get("recovery_succeeded"):
            source_unrecovered[primary] += 1

    # Only flag as problematic if majority of its appearances are unrecovered
    problematic_sources = []
    for src, total_count in source_total.items():
        unrecovered_count = source_unrecovered.get(src, 0)
        failure_rate = unrecovered_count / total_count if total_count > 0 else 0
        if failure_rate >= 0.5 and total_count >= 2:
            problematic_sources.append((src, unrecovered_count))
    problematic_sources.sort(key=lambda x: x[1], reverse=True)

    # ── Faithfulness scores ───────────────────────────────────────────────────
    scores = [r.get("faithfulness_score", 0) for r in records]
    avg_faithfulness = sum(scores) / len(scores) if scores else 0

    # ── Hallucination type breakdown ──────────────────────────────────────────
    type_counts: dict[str, int] = defaultdict(int)
    for r in records:
        type_counts[r.get("hallucination_type", "unknown")] += 1

    false_premise_count = type_counts.get("false_premise", 0) + \
                          type_counts.get("grounded_but_distorted", 0)

    # ── Recovery strategy effectiveness ──────────────────────────────────────
    strategy_counts: dict[str, int] = defaultdict(int)
    strategy_success: dict[str, int] = defaultdict(int)
    for r in records:
        strategy = r.get("recovery_strategy", "unknown")
        strategy_counts[strategy] += 1
        if r.get("recovery_succeeded"):
            strategy_success[strategy] += 1

    strategy_effectiveness = {
        strategy: {
            "total": count,
            "recovered": strategy_success.get(strategy, 0),
            "rate": round(strategy_success.get(strategy, 0) / count, 2) if count else 0,
        }
        for strategy, count in strategy_counts.items()
    }

    # ── Actionable recommendations ────────────────────────────────────────────
    recommendations = []

    # Only recommend COARSE_K increase if faithfulness is low AND
    # the failure type is content_not_grounded (not false premise)
    content_not_grounded = type_counts.get("content_not_grounded", 0)
    if avg_faithfulness < -1.0 and content_not_grounded > false_premise_count:
        recommendations.append(
            "Average faithfulness score is low and failures are mostly content retrieval issues. "
            "Consider increasing COARSE_K to retrieve more candidates, "
            "or re-chunk your documents with smaller chunk sizes."
        )

    # False premise questions are a user education issue, not a data issue
    if false_premise_count > total * 0.3:
        recommendations.append(
            f"{false_premise_count}/{total} hallucinations involved false premises or "
            "distorted interpretations — the document explicitly contradicted the question. "
            "This is correct system behaviour. Consider adding example queries to "
            "help users ask better questions."
        )

    if unrecovered > total * 0.3:
        recommendations.append(
            f"{unrecovered}/{total} hallucinations could not be recovered after all retries. "
            "Review the queries in the records below — most are likely unanswerable "
            "from current documents."
        )

    # Only recommend re-ingestion for sources with proven high failure rates
    if problematic_sources:
        top_src, count = problematic_sources[0]
        recommendations.append(
            f"Source '{top_src}' is the primary retrieved source in {count} unrecovered "
            "hallucination events (>50% failure rate). Consider re-chunking or "
            "improving this document's content quality."
        )

    if not recommendations:
        recommendations.append(
            "No critical issues detected. System is handling hallucinations correctly."
        )

    return {
        "total_hallucinations": total,
        "recovered": recovered,
        "unrecovered": unrecovered,
        "recovery_rate": round(recovered / total, 2) if total else 0,
        "avg_faithfulness_score": round(avg_faithfulness, 4),
        "hallucination_types": dict(type_counts),
        "common_query_terms": common_query_terms,
        "problematic_sources": problematic_sources,
        "strategy_effectiveness": strategy_effectiveness,
        "recommendations": recommendations,
    }