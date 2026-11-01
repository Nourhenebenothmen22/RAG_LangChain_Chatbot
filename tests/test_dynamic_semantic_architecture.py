"""Tests for Dynamic, Semantic, and Context-Aware Architecture improvements.

Validates:
1. Query-Adaptive RRF channel weighting.
2. Semantic Context Compression and deduplication across overlapping chunks.
3. Dynamic uppercase structure parsing.
4. LLM-as-a-Judge semantic evaluation harness.
"""

from typing import List, Tuple
from unittest.mock import MagicMock
import pytest

from langchain_core.documents import Document

from rag_langchain_chatbot.chunking import DocumentStructureParser
from rag_langchain_chatbot.evaluation import RAGEvaluator
from rag_langchain_chatbot.pipeline import compress_context_tuples
from rag_langchain_chatbot.retrieve import calculate_adaptive_weights, reciprocal_rank_fusion


def test_calculate_adaptive_weights():
    """Verify adaptive weights prioritize sparse BM25 for code/number queries and dense for long questions."""
    # 1. Specific code/number query (should boost sparse BM25)
    weights_code = calculate_adaptive_weights("What is threshold T301 in BF-06?")
    assert weights_code["sparse"] > weights_code["dense"]

    weights_short_num = calculate_adaptive_weights("seuil 0.6")
    assert weights_short_num["sparse"] > weights_short_num["dense"]

    # 2. Long conceptual question (should boost dense vector search)
    weights_long = calculate_adaptive_weights(
        "Could you please explain in detailed terms how the insurance claim arbitration process functions?"
    )
    assert weights_long["dense"] > weights_long["sparse"]


def test_reciprocal_rank_fusion_with_adaptive_weights():
    """Verify reciprocal rank fusion correctly applies channel importance weights."""
    doc_a = Document(page_content="Policy details on BF-06", metadata={"chunk_id": "doc_a"})
    doc_b = Document(page_content="General car repair terms", metadata={"chunk_id": "doc_b"})

    dense_list = [(doc_b, 0.95), (doc_a, 0.70)]  # doc_b is #1 in dense
    sparse_list = [(doc_a, 12.5), (doc_b, 4.0)]  # doc_a is #1 in sparse

    # Without weights (equal 1.0)
    equal_fused = reciprocal_rank_fusion(
        ranked_lists=[dense_list, sparse_list],
        k=60,
        source_names=["dense", "sparse"],
        channel_weights={"dense": 1.0, "sparse": 1.0}
    )
    # Both have rank 1 in one list and rank 2 in the other -> identical scores
    assert len(equal_fused) == 2
    assert pytest.approx(equal_fused[0][1], rel=1e-4) == equal_fused[1][1]

    # With sparse boost (e.g. for exact code search BF-06)
    sparse_boosted = reciprocal_rank_fusion(
        ranked_lists=[dense_list, sparse_list],
        k=60,
        source_names=["dense", "sparse"],
        channel_weights={"dense": 0.8, "sparse": 1.4}
    )
    # doc_a was #1 in sparse, so doc_a must be ranked first
    assert sparse_boosted[0][0].metadata["chunk_id"] == "doc_a"
    assert sparse_boosted[0][1] > sparse_boosted[1][1]


def test_compress_context_tuples_deduplication():
    """Verify compress_context_tuples removes redundant sentences from overlapping chunk boundaries."""
    shared_sentence = "All comprehensive windshield damage under 200 EUR is reimbursed without deductible."
    chunk_1 = f"[Policy.pdf > Section 1]\n{shared_sentence}\nFirst unique clause regarding roadside assistance."
    chunk_2 = f"[Policy.pdf > Section 1]\n{shared_sentence}\nSecond unique clause regarding towing fees."

    raw_tuples: List[Tuple[str, str, int, str]] = [
        ("id_1", chunk_1, 1, "Policy.pdf"),
        ("id_2", chunk_2, 1, "Policy.pdf"),
    ]

    compressed = compress_context_tuples(raw_tuples)
    assert len(compressed) == 2

    c1_content = compressed[0][1]
    c2_content = compressed[1][1]

    # Chunk 1 contains the shared sentence
    assert shared_sentence in c1_content
    # Chunk 2 has deduplicated the verbatim sentence from chunk 1
    assert shared_sentence not in c2_content
    # Chunk 2 still retains its unique clause and headers
    assert "Second unique clause regarding towing fees." in c2_content
    assert "[Policy.pdf > Section 1]" in c2_content


def test_document_structure_parser_dynamic_uppercase_headings():
    """Verify DocumentStructureParser detects dynamic uppercase headings in unstructured documents."""
    parser = DocumentStructureParser()
    text = (
        "Document preamble text.\n\n"
        "TERMES ET CONDITIONS GENERALES\n"
        "La présente convention régit les rapports entre assureur et assuré.\n\n"
        "EXCLUSIONS DE GARANTIE\n"
        "Sont formellement exclus les sinistres résultant de fautes intentionnelles."
    )

    blocks = parser.parse(text, default_title="Contrat")
    assert len(blocks) >= 2

    titles = [b.title for b in blocks]
    assert any("Termes Et Conditions Generales" in t for t in titles)
    assert any("Exclusions De Garantie" in t for t in titles)


def test_ragevaluator_llm_judge():
    """Verify RAGEvaluator.evaluate_with_llm_judge parses structured semantic judgements."""
    evaluator = RAGEvaluator()

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"faithfulness": 0.95, "answer_relevance": 0.92, "context_precision": 0.88}'
    )

    scores = evaluator.evaluate_with_llm_judge(
        question="What is the deductible?",
        answer="The deductible is 200 EUR.",
        context="Contract specifies a deductible of 200 EUR per accident.",
        ground_truth="200 EUR deductible per accident.",
        llm=mock_llm
    )

    assert scores is not None
    assert scores["faithfulness"] == 0.95
    assert scores["answer_relevance"] == 0.92
    assert scores["context_precision"] == 0.88
