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
    """Analyze hallucination patterns to surface actionable improvements.

    This is the 'Offline Analysis' step in the improvement loop.
    Returns structured insights about what to fix.
    """
    records = get_records(client_id=client_id, limit=1000)

    if not records:
        return {
            "total_hallucinations": 0,
            "message": "No hallucination records yet.",
        }

    total = len(records)
    recovered = sum(1 for r in records if r.get("recovery_succeeded"))
    unrecovered = total - recovered

    # Which queries hallucinate most — find common patterns
    query_words: dict[str, int] = defaultdict(int)
    for r in records:
        for word in r.get("original_query", "").lower().split():
            if len(word) > 4:  # skip short words
                query_words[word] += 1
    common_query_terms = sorted(query_words.items(), key=lambda x: x[1], reverse=True)[:10]

    # Which sources produce the most hallucinations
    source_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for src in r.get("retrieved_sources", []):
            source_counts[src] += 1
    problematic_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Average faithfulness score — lower = worse retrieval quality overall
    scores = [r.get("faithfulness_score", 0) for r in records]
    avg_faithfulness = sum(scores) / len(scores) if scores else 0

    # Recovery strategy effectiveness
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

    # Actionable recommendations
    recommendations = []

    if avg_faithfulness < -1.0:
        recommendations.append(
            "Average faithfulness score is low. Consider increasing COARSE_K "
            "to retrieve more candidates, or re-chunk your documents with smaller chunk sizes."
        )

    if unrecovered > total * 0.3:
        recommendations.append(
            f"{unrecovered}/{total} hallucinations could not be recovered. "
            "Review the top problematic sources and consider re-ingesting them."
        )

    if problematic_sources:
        top_src = problematic_sources[0][0]
        recommendations.append(
            f"Source '{top_src}' appears in {problematic_sources[0][1]} hallucination events. "
            "Consider re-chunking or improving the content quality of this document."
        )

    if common_query_terms:
        terms = [t[0] for t in common_query_terms[:3]]
        recommendations.append(
            f"Queries containing {terms} frequently hallucinate. "
            "Ensure documents covering these topics are properly ingested and chunked."
        )

    return {
        "total_hallucinations": total,
        "recovered": recovered,
        "unrecovered": unrecovered,
        "recovery_rate": round(recovered / total, 2) if total else 0,
        "avg_faithfulness_score": round(avg_faithfulness, 4),
        "hallucination_types": {
            r.get("hallucination_type"): sum(
                1 for x in records if x.get("hallucination_type") == r.get("hallucination_type")
            )
            for r in records
        },
        "common_query_terms": common_query_terms,
        "problematic_sources": problematic_sources,
        "strategy_effectiveness": strategy_effectiveness,
        "recommendations": recommendations,
    }