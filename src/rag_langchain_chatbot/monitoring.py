"""Metrics tracking, API consumption monitoring, and cost estimation."""

import datetime
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from rag_langchain_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)


class CostCalculator:
    """Calculates estimated API costs based on published pricing rates.
    
    Strictly distinguishes exact token counts from estimated financial expenditures.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.rates = self.settings.pricing_rates

    def estimate_cost_usd(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Compute estimated USD cost for a given model and token consumption.
        
        Formula: (prompt_tokens * input_rate_per_M + completion_tokens * output_rate_per_M) / 1,000,000
        """
        model_key = model_name.lower()
        matched_rate = None

        for rate_key in self.rates:
            if rate_key in model_key:
                matched_rate = self.rates[rate_key]
                break

        if not matched_rate:
            matched_rate = {"input_per_million": 0.0, "output_per_million": 0.0}

        cost_in = (prompt_tokens * matched_rate.get("input_per_million", 0.0)) / 1_000_000.0
        cost_out = (completion_tokens * matched_rate.get("output_per_million", 0.0)) / 1_000_000.0

        return round(cost_in + cost_out, 6)


class MetricsTracker:
    """Persistent SQLite-backed logger for pipeline requests, token usage, latency, and evaluations."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.db_path = self.settings.metrics_db_path
        self.cost_calculator = CostCalculator(self.settings)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode enabled for high concurrency."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema if not present."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer_preview TEXT,
                    model_name TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    retrieval_latency_ms REAL NOT NULL,
                    generation_latency_ms REAL NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    error TEXT,
                    retrieved_chunks_count INTEGER NOT NULL,
                    langsmith_run_id TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    question TEXT NOT NULL,
                    faithfulness_score REAL NOT NULL,
                    answer_relevance_score REAL NOT NULL,
                    context_precision_score REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    report_json TEXT
                );
            """)
            conn.commit()

    def log_request(
        self,
        query: str,
        answer: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        retrieved_chunks_count: int,
        error: Optional[str] = None,
        langsmith_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record an API request event with latency, token metrics, and estimated cost."""
        total_tokens = prompt_tokens + completion_tokens
        total_latency_ms = retrieval_latency_ms + generation_latency_ms
        estimated_cost = self.cost_calculator.estimate_cost_usd(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        answer_preview = (answer[:200] + "...") if len(answer) > 200 else answer

        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO request_metrics (
                    timestamp, query, answer_preview, model_name,
                    prompt_tokens, completion_tokens, total_tokens,
                    retrieval_latency_ms, generation_latency_ms, total_latency_ms,
                    estimated_cost_usd, error, retrieved_chunks_count, langsmith_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso, query, answer_preview, model_name,
                prompt_tokens, completion_tokens, total_tokens,
                retrieval_latency_ms, generation_latency_ms, total_latency_ms,
                estimated_cost, error, retrieved_chunks_count, langsmith_run_id
            ))
            conn.commit()
            record_id = cursor.lastrowid

        return {
            "id": record_id,
            "timestamp": now_iso,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency_ms,
            "estimated_cost_usd": estimated_cost,
        }

    def log_evaluation(
        self,
        dataset_name: str,
        question: str,
        faithfulness: float,
        relevance: float,
        precision: float,
        report_json: str = "{}"
    ) -> None:
        """Record an evaluation benchmark result."""
        overall = (faithfulness + relevance + precision) / 3.0
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO evaluation_metrics (
                    timestamp, dataset_name, question,
                    faithfulness_score, answer_relevance_score,
                    context_precision_score, overall_score, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso, dataset_name, question,
                faithfulness, relevance, precision, overall, report_json
            ))
            conn.commit()

    def get_summary_kpis(self) -> Dict[str, Any]:
        """Aggregate high-level KPIs across all historical requests."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                    COALESCE(AVG(total_latency_ms), 0.0) as avg_latency_ms,
                    COALESCE(SUM(estimated_cost_usd), 0.0) as total_cost_usd,
                    COALESCE(SUM(CASE WHEN error IS NOT NULL AND error != '' AND error NOT LIKE '%API notice%' THEN 1 ELSE 0 END), 0) as error_count
                FROM request_metrics;
            """).fetchone()

        total_req = row["total_requests"]
        error_count = row["error_count"]
        error_rate = (error_count / total_req * 100.0) if total_req > 0 else 0.0

        return {
            "total_requests": total_req,
            "total_tokens": row["total_tokens"],
            "prompt_tokens": row["total_prompt_tokens"],
            "completion_tokens": row["total_completion_tokens"],
            "avg_latency_ms": round(row["avg_latency_ms"], 2),
            "total_cost_usd": round(row["total_cost_usd"], 6),
            "error_rate_pct": round(error_rate, 2),
            "error_count": error_count,
        }

    def get_recent_requests(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent request logs for dashboard tabular inspection."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM request_metrics ORDER BY id DESC LIMIT ?;
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_evaluation_benchmarks(self) -> List[Dict[str, Any]]:
        """Retrieve historical evaluation benchmark results."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM evaluation_metrics ORDER BY id DESC LIMIT 50;
            """).fetchall()
        return [dict(r) for r in rows]


_TRACKER_INSTANCE: Optional[MetricsTracker] = None


def get_metrics_tracker(settings: Optional[Settings] = None) -> MetricsTracker:
    """Return the global MetricsTracker singleton instance."""
    global _TRACKER_INSTANCE
    if _TRACKER_INSTANCE is None:
        _TRACKER_INSTANCE = MetricsTracker(settings=settings)
    return _TRACKER_INSTANCE
