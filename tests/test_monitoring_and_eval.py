"""Tests for LangSmith observability, metrics tracking, cost estimation, and evaluation."""

from pathlib import Path

import pytest

from rag_langchain_chatbot.config import get_settings
from rag_langchain_chatbot.evaluation import (
    BENCHMARK_DATASET,
    RAGEvaluator,
    compute_answer_relevance,
    compute_context_precision,
    compute_faithfulness,
)
from rag_langchain_chatbot.monitoring import CostCalculator, MetricsTracker
from rag_langchain_chatbot.pipeline import HybridRAGPipeline


def test_cost_calculator():
    """Verify published rate cost calculation strictly separates tokens and estimated USD."""
    calculator = CostCalculator()

    # gpt-4o-mini rates: $0.15/1M in, $0.60/1M out
    cost = calculator.estimate_cost_usd(
        model_name="gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000
    )
    assert pytest.approx(cost, rel=1e-3) == 0.75

    # 10,000 prompt tokens, 1,000 completion tokens
    cost_small = calculator.estimate_cost_usd(
        model_name="gpt-4o-mini",
        prompt_tokens=10_000,
        completion_tokens=1_000
    )
    expected = (10_000 * 0.15 + 1_000 * 0.60) / 1_000_000
    assert pytest.approx(cost_small, rel=1e-3) == expected


def test_metrics_tracker_logging(tmp_path: Path):
    """Verify SQLite metrics tracker records requests and calculates accurate KPI summaries."""
    settings = get_settings()
    custom_db = tmp_path / "test_metrics.sqlite3"
    
    # Override db path for test
    tracker = MetricsTracker(settings=settings)
    tracker.db_path = custom_db
    tracker._init_db()

    # Log two distinct requests
    tracker.log_request(
        query="What is the fraud score threshold?",
        answer="The threshold is 0.6.",
        model_name="gpt-4o-mini",
        prompt_tokens=150,
        completion_tokens=25,
        retrieval_latency_ms=45.0,
        generation_latency_ms=120.0,
        retrieved_chunks_count=5
    )

    tracker.log_request(
        query="Failed query test",
        answer="Error encountered",
        model_name="gpt-4o-mini",
        prompt_tokens=50,
        completion_tokens=5,
        retrieval_latency_ms=10.0,
        generation_latency_ms=5.0,
        retrieved_chunks_count=1,
        error="Connection timeout"
    )

    kpis = tracker.get_summary_kpis()
    assert kpis["total_requests"] == 2
    assert kpis["total_tokens"] == (175 + 55)
    assert kpis["prompt_tokens"] == (150 + 50)
    assert kpis["completion_tokens"] == (25 + 5)
    assert kpis["error_count"] == 1
    assert kpis["error_rate_pct"] == 50.0

    recent = tracker.get_recent_requests(limit=5)
    assert len(recent) == 2
    assert recent[0]["query"] == "Failed query test"


def test_evaluation_metrics_computation():
    """Verify Faithfulness, Relevance, and Context Precision metric formulas."""
    context = (
        "BF-06 Human-in-the-Loop : si fraud_score > 0,6, gel du workflow et arbitrage par un EXPERT (valider/rejeter)."
    )

    # 1. Faithfulness
    faithful_answer = "Si le fraud_score depasse 0,6, le workflow est gele et un expert arbitre la decision."
    unfaithful_answer = "Le client recoit un remboursement immediat par virement bancaire automatique."

    score_faithful = compute_faithfulness(faithful_answer, context)
    score_unfaithful = compute_faithfulness(unfaithful_answer, context)

    assert score_faithful > score_unfaithful
    assert score_faithful >= 0.7

    # 2. Answer relevance
    ground_truth = "An interrupt() call freezes the workflow, escalating to an expert."
    keywords = ["interrupt", "0.6", "expert", "workflow"]
    rel_score = compute_answer_relevance("When 0.6 is exceeded, workflow interrupts for an expert.", ground_truth, keywords)
    assert rel_score >= 0.75

    # 3. Context precision
    prec_score = compute_context_precision(context, ["fraud_score", "0.6", "expert"])
    assert prec_score == 1.0


def test_pipeline_generation_and_tracking():
    """Verify that querying the pipeline executes generation and records metrics."""
    pipeline = HybridRAGPipeline()
    response = pipeline.query("What happens when the fraud score exceeds 0.6?", top_k=3, rerank=True)

    assert response.query == "What happens when the fraud score exceeds 0.6?"
    assert len(response.answer) > 20
    assert response.total_tokens > 0
    assert response.total_latency_ms > 0
    assert response.estimated_cost_usd >= 0.0
    assert len(response.chunks) == 3


def test_rag_evaluator_benchmark_execution(tmp_path: Path):
    """Verify end-to-end benchmark evaluation suite generates valid scorecard."""
    settings = get_settings()
    evaluator = RAGEvaluator(settings=settings)

    pipeline = HybridRAGPipeline(settings=settings)
    report = evaluator.evaluate_pipeline(pipeline)

    assert "mean_faithfulness" in report
    assert "mean_answer_relevance" in report
    assert "mean_context_precision" in report
    assert "mean_overall_score" in report
    assert len(report["samples"]) == len(BENCHMARK_DATASET)
    assert report["mean_overall_score"] >= 0.40
