"""Tests for Multilingual Cross-Encoder Reranker configuration, fallback, and resilience."""

from unittest.mock import PropertyMock, patch

import pytest
from langchain_core.documents import Document

from rag_langchain_chatbot.config import Settings
from rag_langchain_chatbot.reranker import (
    CrossEncoderReranker,
    _RERANKER_CACHE,
    get_cross_encoder,
)


def test_reranker_default_configuration():
    """Verify BAAI/bge-reranker-v2-m3 is the default configured reranker."""
    settings = Settings(gemini_api_key="test-key")
    assert settings.reranker_model_name == "BAAI/bge-reranker-v2-m3"


def test_reranker_env_var_override(monkeypatch):
    """Verify reranker model can be overridden through RERANKER_MODEL_NAME env var."""
    monkeypatch.setenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    settings = Settings(gemini_api_key="test-key")
    assert settings.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-2-v2"


def test_cross_encoder_graceful_fallback():
    """Verify that when primary model is unavailable offline, it gracefully falls back to cached models."""
    settings = Settings(
        gemini_api_key="test-key",
        reranker_model_name="nonexistent-org/unobtainable-reranker-model-v999",
    )
    
    # Should fall back to cached local model (e.g. TinyBERT or MiniLM)
    model = get_cross_encoder(settings)
    assert model is not None
    # Model should be able to score pairs
    scores = model.predict([["insurance deductible", "The deductible is 500 EUR."]])
    assert len(scores) == 1
    assert isinstance(float(scores[0]), float)


def test_reranker_graceful_degradation_when_all_models_fail():
    """Verify CrossEncoderReranker returns unreranked candidates if model fails completely."""
    reranker = CrossEncoderReranker()
    with patch.object(CrossEncoderReranker, "model", new_callable=PropertyMock, return_value=None):
        docs = [
            Document(page_content="Doc 1", metadata={"rrf_score": 0.05, "rank": 1}),
            Document(page_content="Doc 2", metadata={"rrf_score": 0.02, "rank": 2}),
        ]
        reranked = reranker.rerank(query="What is the deductible?", documents=docs, top_k=2)
        assert len(reranked) == 2
        doc0, score0 = reranked[0]
        assert doc0.page_content == "Doc 1"
        assert score0 == 0.05


def test_multilingual_scoring_french_english_arabic():
    """Verify cross-encoder scores French, English, and Arabic query-document pairs."""
    reranker = CrossEncoderReranker()
    
    pairs = [
        # English
        Document(
            page_content="The fraud detection threshold is 0.6 for human review.",
            metadata={"rrf_score": 0.05}
        ),
        # French
        Document(
            page_content="Le seuil de détection des fraudes est fixé à 0.6 pour examen humain.",
            metadata={"rrf_score": 0.04}
        ),
        # Arabic
        Document(
            page_content="عتبة كشف الاحتيال المحددة للمراجعة البشرية هي 0.6.",
            metadata={"rrf_score": 0.03}
        ),
    ]

    reranked = reranker.rerank(
        query="fraud threshold / seuil de fraude / عتبة الاحتيال",
        documents=pairs,
        top_k=3
    )
    assert len(reranked) == 3
    for doc, score in reranked:
        assert "rerank_score" in doc.metadata
        assert isinstance(float(score), float)
