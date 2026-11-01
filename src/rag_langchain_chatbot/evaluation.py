"""Evaluation suite for Hybrid RAG: Faithfulness, Relevance, and Context Precision with LangSmith cloud sync."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag_langchain_chatbot.bm25 import normalize_diacritics
from rag_langchain_chatbot.config import Settings, get_settings
from rag_langchain_chatbot.monitoring import get_metrics_tracker

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """A benchmark test sample with reference ground-truth requirements."""
    id: str
    question: str
    ground_truth: str
    reference_keywords: List[str]
    expected_clause: str


# Standard benchmark test cases derived from the Kortex AI specification
BENCHMARK_DATASET: List[EvaluationSample] = [
    EvaluationSample(
        id="eval-01",
        question="What happens when the fraud score exceeds 0.6?",
        ground_truth="An interrupt() call is triggered, freezing the workflow state in Redis, alerting an EXPERT for manual arbitration.",
        reference_keywords=["interrupt", "0.6", "expert", "redis", "arbitrage"],
        expected_clause="BF-06 / T304"
    ),
    EvaluationSample(
        id="eval-02",
        question="What is the monthly subscription pricing for an organisation?",
        ground_truth="Base subscription is 490 TND per month including 100 analyses and 1 ADMIN account. Extra seats are 60 TND for Gestionnaire and 120 TND for Expert.",
        reference_keywords=["490", "tnd", "100", "admin", "60", "120"],
        expected_clause="Section 3: Pricing et billing"
    ),
    EvaluationSample(
        id="eval-03",
        question="What rights and duties are assigned exclusively to the EXPERT role?",
        ground_truth="The EXPERT role is the only role authorized to arbitrate fraud alerts (validate or reject) and has access to the review queue.",
        reference_keywords=["expert", "arbitrer", "valider", "rejeter", "fraude"],
        expected_clause="Section 2: Acteurs et rôles"
    ),
    EvaluationSample(
        id="eval-04",
        question="How is the claim repair cost calculated deterministically?",
        ground_truth="Deterministic calculation in Python: parts + labor minus deductible, capped according to versioned tariff tables.",
        reference_keywords=["chiffrage", "déterministe", "pièces", "main-d'œuvre", "franchise"],
        expected_clause="BF-08 / T213"
    ),
    EvaluationSample(
        id="eval-05",
        question="Which vision model is used for detecting damaged car parts?",
        ground_truth="YOLO is used with bounding boxes, part classification, and severity estimation, exported to ONNX for inference.",
        reference_keywords=["yolo", "onnx", "boîtes", "sévérité"],
        expected_clause="BF-03 / T205"
    ),
    EvaluationSample(
        id="eval-06",
        question="What is the quantum encryption protocol used by Kortex AI satellites?",
        ground_truth="Information not found in the indexed database and documents. Please verify the query or ingest the relevant sources.",
        reference_keywords=["not found", "information", "documents"],
        expected_clause="N/A (Unhallucinated Out-of-Domain Control)"
    ),
]


def compute_faithfulness(answer: str, context: str) -> float:
    """Measure whether answer statements are faithful to the retrieved context (penalizing hallucination)."""
    if not answer or not context:
        return 0.0

    # Extract sentences / assertions from answer
    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    supported_count = 0
    context_norm = normalize_diacritics(context.lower())
    context_words = set(re.findall(r"\w+", context_norm))

    for sentence in sentences:
        sent_norm = normalize_diacritics(sentence.lower())
        tokens = [t for t in re.findall(r"\w+", sent_norm) if len(t) >= 3]
        if not tokens:
            continue

        overlap = sum(
            1 for t in tokens
            if t in context_norm or any(c.startswith(t[:4]) or t.startswith(c[:4]) for c in context_words if len(c) >= 3)
        )
        overlap_ratio = overlap / len(tokens)
        if overlap_ratio >= 0.35:
            supported_count += 1

    return round(supported_count / max(1, len(sentences)), 4)


def compute_answer_relevance(answer: str, ground_truth: str, reference_keywords: List[str]) -> float:
    """Measure whether the generated response directly addresses the question and captures target facts."""
    if not answer:
        return 0.0

    answer_norm = normalize_diacritics(re.sub(r"(\d+),(\d+)", r"\1.\2", answer.lower()))
    matched_keywords = sum(
        1 for kw in reference_keywords
        if normalize_diacritics(re.sub(r"(\d+),(\d+)", r"\1.\2", kw.lower())) in answer_norm
    )
    relevance_score = matched_keywords / max(1, len(reference_keywords))
    return round(min(1.0, relevance_score), 4)


def compute_context_precision(retrieved_context: str, reference_keywords: List[str]) -> float:
    """Measure whether the retrieved chunks contain the necessary facts."""
    if not retrieved_context:
        return 0.0

    context_norm = normalize_diacritics(re.sub(r"(\d+),(\d+)", r"\1.\2", retrieved_context.lower()))
    matched = 0
    for kw in reference_keywords:
        kw_norm = normalize_diacritics(re.sub(r"(\d+),(\d+)", r"\1.\2", kw.lower()))
        if kw_norm in context_norm:
            matched += 1

    return round(matched / max(1, len(reference_keywords)), 4)


class RAGEvaluator:
    """Evaluates RAG pipeline responses against benchmark datasets and synchronizes with LangSmith cloud."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.metrics_tracker = get_metrics_tracker(self.settings)

    def evaluate_with_llm_judge(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
        llm: Optional[Any] = None,
    ) -> Optional[Dict[str, float]]:
        """Perform semantic LLM-as-a-Judge evaluation of faithfulness, relevance, and precision."""
        judge_llm = llm or getattr(self.settings, "llm", None)
        if judge_llm is None or not hasattr(judge_llm, "invoke"):
            return None

        prompt = (
            "You are an expert AI evaluator assessing RAG performance.\n"
            f"Question: {question}\n"
            f"Ground Truth Reference: {ground_truth}\n"
            f"Retrieved Context: {context[:2000]}\n"
            f"Generated Answer: {answer}\n\n"
            "Evaluate the answer across three metrics from 0.0 to 1.0:\n"
            "1. Faithfulness: Is the answer strictly supported by the retrieved context without hallucinations?\n"
            "2. Answer Relevance: Does the answer directly address the question?\n"
            "3. Context Precision: Does the retrieved context contain the facts needed?\n\n"
            "Respond in JSON format:\n"
            '{"faithfulness": float, "answer_relevance": float, "context_precision": float}'
        )
        try:
            res = judge_llm.invoke(prompt)
            content = getattr(res, "content", str(res))
            m = re.search(r"\{.*?\}", content, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                return {
                    "faithfulness": float(data.get("faithfulness", 0.0)),
                    "answer_relevance": float(data.get("answer_relevance", 0.0)),
                    "context_precision": float(data.get("context_precision", 0.0)),
                }
        except Exception as e:
            logger.debug("LLM-as-a-judge evaluation failed (%s). Falling back to heuristic metrics.", e)
        return None

    def sync_dataset_to_langsmith(self) -> Optional[str]:
        """Create or update benchmark dataset in LangSmith Cloud."""
        try:
            import socket
            with socket.create_connection(("api.smith.langchain.com", 443), timeout=1.0):
                pass
        except Exception:
            return None

        try:
            from langsmith import Client
            client = Client()
            if not client.api_key:
                logger.info("LangSmith API key not configured; skipping cloud dataset sync.")
                return None

            dataset_name = "kortex-rag-benchmark"
            try:
                dataset = client.read_dataset(dataset_name=dataset_name)
                logger.info("Found existing LangSmith dataset: %s (id: %s)", dataset_name, dataset.id)
            except Exception:
                dataset = client.create_dataset(
                    dataset_name=dataset_name,
                    description="Benchmark evaluation dataset for Kortex AI contract specifications"
                )
                logger.info("Created new LangSmith dataset: %s (id: %s)", dataset_name, dataset.id)

                for sample in BENCHMARK_DATASET:
                    client.create_example(
                        inputs={"question": sample.question},
                        outputs={
                            "ground_truth": sample.ground_truth,
                            "reference_keywords": sample.reference_keywords,
                            "expected_clause": sample.expected_clause,
                        },
                        dataset_id=dataset.id
                    )
                logger.info("Synced %d benchmark examples to LangSmith dataset.", len(BENCHMARK_DATASET))

            return str(dataset.id)

        except Exception as e:
            logger.warning("Could not sync dataset to LangSmith: %s", e)
            return None

    def evaluate_pipeline(self, pipeline: Any) -> Dict[str, Any]:
        """Run benchmark evaluation over all samples in the benchmark dataset and push metrics."""
        # 1. Sync dataset to LangSmith Cloud if configured
        langsmith_dataset_id = self.sync_dataset_to_langsmith()

        # Try initializing LangSmith client for feedback logging
        ls_client = None
        ls_project_id = None
        is_online = False
        try:
            import socket
            with socket.create_connection(("api.smith.langchain.com", 443), timeout=1.0):
                is_online = True
        except Exception:
            is_online = False

        if is_online:
            try:
                from langsmith import Client
                c = Client()
                if c.api_key:
                    ls_client = c
                    target_project = getattr(self.settings, "langchain_project", None) or getattr(self.settings, "langsmith_project", "hybrid-rag")
                    try:
                        proj = c.read_project(project_name=target_project)
                        ls_project_id = str(proj.id)
                    except Exception:
                        try:
                            proj = c.create_project(project_name=target_project)
                            ls_project_id = str(proj.id)
                        except Exception:
                            ls_project_id = None
            except Exception:
                pass

        results: List[Dict[str, Any]] = []

        total_faithfulness = 0.0
        total_relevance = 0.0
        total_precision = 0.0

        for sample in BENCHMARK_DATASET:
            # Query pipeline
            response = pipeline.query(sample.question, top_k=3, rerank=True)
            context = response.format_context()
            answer = response.answer if hasattr(response, "answer") else context

            llm_eval = self.evaluate_with_llm_judge(
                question=sample.question,
                answer=answer,
                context=context,
                ground_truth=sample.ground_truth,
                llm=getattr(pipeline, "llm", None)
            )

            if llm_eval:
                faithfulness = llm_eval["faithfulness"]
                relevance = llm_eval["answer_relevance"]
                precision = llm_eval["context_precision"]
            else:
                faithfulness = compute_faithfulness(answer, context)
                relevance = compute_answer_relevance(answer, sample.ground_truth, sample.reference_keywords)
                precision = compute_context_precision(context, sample.reference_keywords)

            overall = round((faithfulness + relevance + precision) / 3.0, 4)

            total_faithfulness += faithfulness
            total_relevance += relevance
            total_precision += precision

            sample_result = {
                "sample_id": sample.id,
                "question": sample.question,
                "expected_clause": sample.expected_clause,
                "faithfulness": faithfulness,
                "answer_relevance": relevance,
                "context_precision": precision,
                "overall_score": overall,
                "answer_excerpt": (answer[:150] + "...") if len(answer) > 150 else answer,
                "langsmith_run_id": getattr(response, "langsmith_run_id", None)
            }
            results.append(sample_result)

            # Log to local metrics database
            self.metrics_tracker.log_evaluation(
                dataset_name="kortex-benchmark-v1",
                question=sample.question,
                faithfulness=faithfulness,
                relevance=relevance,
                precision=precision,
                report_json=json.dumps(sample_result)
            )

            # Send evaluation feedback to LangSmith Cloud run if run ID exists
            if ls_client and response.langsmith_run_id:
                try:
                    fb_kwargs: Dict[str, Any] = {}
                    if ls_project_id:
                        fb_kwargs["session_id"] = ls_project_id
                    elif getattr(self.settings, "langchain_project", None):
                        fb_kwargs["session_name"] = self.settings.langchain_project

                    ls_client.create_feedback(
                        run_id=response.langsmith_run_id,
                        key="faithfulness",
                        score=faithfulness,
                        comment="Automated Kortex AI Faithfulness evaluator",
                        **fb_kwargs
                    )
                    ls_client.create_feedback(
                        run_id=response.langsmith_run_id,
                        key="answer_relevance",
                        score=relevance,
                        comment="Automated Kortex AI Answer Relevance evaluator",
                        **fb_kwargs
                    )
                    ls_client.create_feedback(
                        run_id=response.langsmith_run_id,
                        key="context_precision",
                        score=precision,
                        comment="Automated Kortex AI Context Precision evaluator",
                        **fb_kwargs
                    )
                except Exception as ex:
                    logger.debug("Could not record feedback to LangSmith run: %s", ex)

        n = len(BENCHMARK_DATASET)
        summary = {
            "dataset_name": "kortex-benchmark-v1",
            "langsmith_dataset_id": langsmith_dataset_id,
            "total_samples": n,
            "mean_faithfulness": round(total_faithfulness / n, 4),
            "mean_answer_relevance": round(total_relevance / n, 4),
            "mean_context_precision": round(total_precision / n, 4),
            "mean_overall_score": round((total_faithfulness + total_relevance + total_precision) / (3 * n), 4),
            "samples": results
        }

        # Persist report to disk
        self.settings.eval_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings.eval_report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation report saved to %s", self.settings.eval_report_path)
        return summary
