"""Tests for grounded generation, hallucination prevention fallback, and explainability provenance."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag_langchain_chatbot.pipeline import (
    FALLBACK_UNAVAILABLE_MSG,
    HybridRAGPipeline,
    RetrievedChunk,
)
from rag_langchain_chatbot.retrieve import reciprocal_rank_fusion


def test_empty_retrieval_triggers_fallback_without_hallucination():
    """Verify when retrieval returns empty results, the pipeline immediately returns fallback."""
    pipeline = HybridRAGPipeline()
    # Mock retriever to return no candidates
    pipeline.retriever.retrieve = MagicMock(return_value=[])

    response = pipeline.query("Nonexistent query with zero evidence", generate_answer=True)

    assert response.is_grounded is False
    assert len(response.chunks) == 0
    assert FALLBACK_UNAVAILABLE_MSG in response.answer


def test_rrf_provenance_tracking():
    """Verify reciprocal_rank_fusion populates dense and sparse ranks and scores in metadata."""
    doc1 = Document(page_content="Content 1", metadata={"chunk_id": "c1", "source": "f1.pdf"})
    doc2 = Document(page_content="Content 2", metadata={"chunk_id": "c2", "source": "f2.pdf"})

    dense_list = [(doc1, 0.9), (doc2, 0.4)]
    sparse_list = [(doc2, 12.5), (doc1, 3.2)]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_list, sparse_list],
        k=60,
        top_k=2,
        source_names=["dense", "sparse"]
    )

    assert len(fused) == 2
    top_doc, top_score = fused[0]

    # Verify provenance metadata exists
    assert "dense_rank" in top_doc.metadata
    assert "sparse_rank" in top_doc.metadata
    assert "dense_score" in top_doc.metadata
    assert "sparse_score" in top_doc.metadata
    assert "rrf_score" in top_doc.metadata
