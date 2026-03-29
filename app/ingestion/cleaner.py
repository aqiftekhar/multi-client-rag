"""Text cleaning utilities for raw ingested documents."""

import html
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace/newlines into single spaces."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_encoding(text: str) -> str:
    """Normalize Unicode to NFC form and replace common mojibake."""
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns (headers, footers, page numbers)."""
    # Remove standalone page numbers like "Page 1 of 12" or "- 3 -"
    text = re.sub(r"(?m)^[-\s]*Page\s+\d+\s+of\s+\d+[-\s]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^[-\s]*\d+\s*[-\s]*$", "", text)
    # Remove repeated dashes used as dividers
    text = re.sub(r"-{4,}", "", text)
    return text


def clean(text: str) -> str:
    """Apply full cleaning pipeline to raw text.

    Steps: HTML strip → encoding fix → boilerplate removal → whitespace normalisation.
    """
    text = strip_html(text)
    text = fix_encoding(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    logger.debug("Cleaned text: %d chars", len(text))
    return text
