"""Anomaly detection for ingested document chunks."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Configurable thresholds
MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 10_000
MAX_SPECIAL_CHAR_RATIO = 0.35  # fraction of non-alphanumeric/space chars
MAX_NUMERIC_RATIO = 0.6        # fraction of digit characters
MIN_WORD_COUNT = 5


@dataclass
class AnomalyReport:
    """Result of anomaly analysis for a single chunk."""

    chunk_text: str
    is_anomalous: bool
    reasons: list[str]


def _char_ratios(text: str) -> tuple[float, float]:
    """Return (special_char_ratio, numeric_ratio) for *text*."""
    if not text:
        return 0.0, 0.0
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    numeric = sum(1 for c in text if c.isdigit())
    return special / len(text), numeric / len(text)


def analyze(text: str) -> AnomalyReport:
    """Analyse a single text chunk for anomalies.

    Checks:
    - Too short or too long
    - Too few real words
    - Excessive special characters (likely binary / encoding artifact)
    - Excessive numeric content (likely data dump)
    """
    reasons: list[str] = []

    if len(text) < MIN_CHUNK_CHARS:
        reasons.append(f"too_short ({len(text)} chars < {MIN_CHUNK_CHARS})")

    if len(text) > MAX_CHUNK_CHARS:
        reasons.append(f"too_long ({len(text)} chars > {MAX_CHUNK_CHARS})")

    words = re.findall(r"[a-zA-Z]{2,}", text)
    if len(words) < MIN_WORD_COUNT:
        reasons.append(f"too_few_words ({len(words)} < {MIN_WORD_COUNT})")

    special_ratio, numeric_ratio = _char_ratios(text)
    if special_ratio > MAX_SPECIAL_CHAR_RATIO:
        reasons.append(f"high_special_chars ({special_ratio:.2f} > {MAX_SPECIAL_CHAR_RATIO})")

    if numeric_ratio > MAX_NUMERIC_RATIO:
        reasons.append(f"high_numeric_ratio ({numeric_ratio:.2f} > {MAX_NUMERIC_RATIO})")

    is_anomalous = bool(reasons)
    if is_anomalous:
        logger.debug("Anomaly detected: %s | snippet: %.80s", reasons, text)

    return AnomalyReport(chunk_text=text, is_anomalous=is_anomalous, reasons=reasons)


def filter_anomalies(texts: list[str]) -> tuple[list[str], list[AnomalyReport]]:
    """Filter *texts*, returning (clean_texts, anomaly_reports).

    Anomalous chunks are excluded from clean_texts but reported separately.
    """
    clean: list[str] = []
    reports: list[AnomalyReport] = []

    for text in texts:
        report = analyze(text)
        reports.append(report)
        if not report.is_anomalous:
            clean.append(text)

    removed = len(texts) - len(clean)
    if removed:
        logger.info("Anomaly filter removed %d/%d chunks.", removed, len(texts))

    return clean, reports
