"""Comprehensive test suite for the Hybrid RAG pipeline."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_langchain_chatbot.bm25 import BM25Index, tokenize
from rag_langchain_chatbot.config import Settings, get_settings
from rag_langchain_chatbot.index import generate_deterministic_chunk_id
from rag_langchain_chatbot.pipeline import HybridRAGPipeline
from rag_langchain_chatbot.reranker import CrossEncoderReranker
from rag_langchain_chatbot.retrieve import reciprocal_rank_fusion


def test_config_loading():
    """Verify settings defaults and environment mapping."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.collection_name in ("rag_documents", "claimeye_documents", "kortex_documents")
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.dense_top_k == 10
    assert settings.bm25_top_k == 10
    assert settings.rrf_k == 60
    assert settings.reranker_top_k == 5
    assert settings.data_dir.is_dir()
    assert settings.chroma_dir.exists()


def test_deterministic_chunk_ids():
    """Verify chunk ID generation is deterministic and content-based."""
    doc1 = Document(
        page_content="Plateforme SaaS multi-tenant d'aide a l'expertise de sinistres automobiles.",
        metadata={"source": "test.pdf", "page": 1}
    )
    doc2 = Document(
        page_content="Plateforme SaaS multi-tenant d'aide a l'expertise de sinistres automobiles.",
        metadata={"source": "test.pdf", "page": 1}
    )
    doc_different = Document(
        page_content="Chiffrage deterministe (pieces + main-d'oeuvre - franchise).",
        metadata={"source": "test.pdf", "page": 1}
    )

    id1 = generate_deterministic_chunk_id(doc1)
    id2 = generate_deterministic_chunk_id(doc2)
    id3 = generate_deterministic_chunk_id(doc_different)

    assert id1 == id2, "Identical content and metadata must yield identical chunk IDs"
    assert id1 != id3, "Different content must yield different chunk IDs"
    assert len(id1) == 64, "Chunk ID must be a valid 64-character SHA-256 hex string"


def test_bm25_tokenization_and_search(tmp_path: Path):
    """Verify BM25 tokenization handles French accents, decimals (0.6 and 0,6) and ranking."""
    tokens = tokenize("Le fraud_score depasse 0,6 ou 0.6 avec l'interruption.")
    assert "0.6" in tokens
    assert "fraud_score" in tokens or "fraud" in tokens

    docs = [
        Document(
            page_content="Si le fraud_score > 0,6 le workflow est interrompu par un expert.",
            metadata={"chunk_id": "chunk_hitl", "source": "test.pdf", "page": 2}
        ),
        Document(
            page_content="La tarification mensuelle de base est de 490 TND par mois pour l'organisation.",
            metadata={"chunk_id": "chunk_pricing", "source": "test.pdf", "page": 3}
        ),
    ]

    index = BM25Index(documents=docs)
    results = index.search("fraud score 0.6", top_k=2)

    assert len(results) >= 1
    top_doc, top_score = results[0]
    assert top_doc.metadata["chunk_id"] == "chunk_hitl"
    assert top_score > 0

    # Test serialization and loading
    save_path = tmp_path / "test_bm25.pkl"
    index.save(save_path)
    assert save_path.is_file()

    loaded_index = BM25Index.load(save_path)
    loaded_results = loaded_index.search("fraud score 0.6", top_k=2)
    assert len(loaded_results) == len(results)
    assert loaded_results[0][0].metadata["chunk_id"] == "chunk_hitl"


def test_reciprocal_rank_fusion_math():
    """Verify RRF mathematical combination and rank ordering."""
    doc_a = Document(page_content="Doc A", metadata={"chunk_id": "A"})
    doc_b = Document(page_content="Doc B", metadata={"chunk_id": "B"})
    doc_c = Document(page_content="Doc C", metadata={"chunk_id": "C"})

    # Dense: [A (rank 1), B (rank 2)]
    dense_ranking = [(doc_a, 0.95), (doc_b, 0.85)]
    # BM25: [A (rank 1), C (rank 2)]
    bm25_ranking = [(doc_a, 5.2), (doc_c, 3.1)]

    # Expected RRF scores with k=60:
    # A: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278688
    # B: 1/(60+2)            = 1/62 ≈ 0.01612903
    # C: 1/(60+2)            = 1/62 ≈ 0.01612903
    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60, top_k=3)

    assert len(fused) == 3
    assert fused[0][0].metadata["chunk_id"] == "A"
    assert pytest.approx(fused[0][1], rel=1e-4) == (1 / 61 + 1 / 61)
    assert pytest.approx(fused[1][1], rel=1e-4) == 1 / 62


def test_cross_encoder_reranker_relevance():
    """Verify Cross-Encoder correctly promotes semantically relevant documents over distractor text."""
    reranker = CrossEncoderReranker()

    query = "What happens when the fraud score exceeds 0.6?"
    relevant_doc = Document(
        page_content="Human-in-the-Loop : si fraud_score > 0,6, gel du workflow et arbitrage par un EXPERT.",
        metadata={"chunk_id": "rel_1"}
    )
    distractor_doc = Document(
        page_content="Tarification et paiement : abonnement mensuel de 490 TND avec 100 analyses incluses.",
        metadata={"chunk_id": "dist_1"}
    )

    reranked = reranker.rerank(query, [distractor_doc, relevant_doc], top_k=2)

    assert len(reranked) == 2
    top_doc, top_score = reranked[0]
    second_doc, second_score = reranked[1]

    assert top_doc.metadata["chunk_id"] == "rel_1"
    assert top_score > second_score
    assert "rerank_score" in top_doc.metadata


def test_end_to_end_hybrid_rag_pipeline():
    """Verify end-to-end pipeline execution, context formatting, and latency."""
    pipeline = HybridRAGPipeline()
    response = pipeline.query("What happens when the fraud score exceeds 0.6?", top_k=3, rerank=True)

    assert response.query == "What happens when the fraud score exceeds 0.6?"
    assert len(response.chunks) == 3
    assert response.reranked is True
    assert response.latency_ms > 0

    top_chunk = response.chunks[0]
    assert top_chunk.score is not None
    assert top_chunk.source != ""
    assert top_chunk.chunk_id != ""

    # Verify context formatting for LLM injection
    context_str = response.format_context()
    assert "Source:" in context_str
    assert "Relevance:" in context_str
    assert len(context_str) > 100
