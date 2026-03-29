"""Unit tests for ingestion pipeline components."""

import pytest
from app.ingestion.cleaner import clean, strip_html, normalize_whitespace
from app.ingestion.anomaly_detector import analyze, filter_anomalies
from app.rag.chunker import chunk_text


class TestCleaner:
    def test_strip_html(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_html_entities(self):
        result = strip_html("AT&amp;T &lt;rocks&gt;")
        assert "&amp;" not in result
        assert "AT&T" in result

    def test_normalize_whitespace(self):
        result = normalize_whitespace("hello   world\n\n\n\nfoo")
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_clean_pipeline(self):
        dirty = "<html><body><p>Hello  world!</p></body></html>"
        result = clean(dirty)
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result


class TestAnomalyDetector:
    def test_normal_text_passes(self):
        text = "This is a normal paragraph with sufficient content to pass anomaly detection checks."
        report = analyze(text)
        assert not report.is_anomalous

    def test_too_short(self):
        report = analyze("Hi.")
        assert report.is_anomalous
        assert any("too_short" in r for r in report.reasons)

    def test_too_few_words(self):
        report = analyze("123 456 789 000 ###")
        assert report.is_anomalous

    def test_filter_removes_anomalies(self):
        texts = [
            "Hi.",  # too short
            "This is a normal sentence with enough words and characters to be valid content.",
            "x",   # too short
        ]
        clean_texts, reports = filter_anomalies(texts)
        assert len(clean_texts) == 1
        assert len([r for r in reports if r.is_anomalous]) == 2


class TestChunker:
    def test_basic_chunking(self):
        text = " ".join(["This is sentence number %d." % i for i in range(100)])
        chunks = chunk_text(text, doc_id="test_doc")
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text
            assert chunk.doc_id == "test_doc"
            assert chunk.chunk_id

    def test_short_text_single_chunk(self):
        text = "This is a short document. It has just two sentences."
        chunks = chunk_text(text, doc_id="short")
        assert len(chunks) == 1
        assert "short document" in chunks[0].text

    def test_chunk_ids_unique(self):
        text = " ".join(["Word %d is here for testing purposes today." % i for i in range(200)])
        chunks = chunk_text(text, doc_id="unique_test")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_index_sequential(self):
        text = " ".join(["Sentence %d contains enough words to fill a chunk buffer." % i for i in range(100)])
        chunks = chunk_text(text, doc_id="seq_test")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestValidationSchema:
    def test_valid_json_response(self):
        from app.validation.schema_validator import validate_rag_response
        raw = '{"answer": "42", "sources": ["doc1"], "confidence": 0.9, "needs_clarification": false}'
        result = validate_rag_response(raw)
        assert result.is_valid
        assert result.parsed.answer == "42"
        assert result.confidence == 0.9

    def test_invalid_json_graceful(self):
        from app.validation.schema_validator import validate_rag_response
        raw = "This is not JSON at all, just plain text."
        result = validate_rag_response(raw)
        assert not result.is_valid
        assert result.parsed is not None  # Graceful fallback
        assert result.parsed.answer  # Should still have some answer

    def test_source_attribution(self):
        from app.validation.schema_validator import check_source_attribution, RAGResponse
        response = RAGResponse(answer="test", sources=["doc1", "invented_doc"], confidence=0.8)
        retrieved = ["doc1", "doc2"]
        hallucinated = check_source_attribution(response, retrieved)
        assert "invented_doc" in hallucinated
        assert "doc1" not in hallucinated
