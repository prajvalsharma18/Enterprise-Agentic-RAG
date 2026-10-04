"""Run the corpus-grounded benchmark through RAGAS and save auditable results."""

from __future__ import annotations

import json
import math
import numbers
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from evaluation.eval_dataset import EVAL_DATASET
from evaluation.ragas_config import BgeM3EvaluationEmbeddings, build_ragas_llm

load_dotenv()

SCORES_FILE = Path("evaluation/latest_scores.json")
RESULTS_FILE = Path("evaluation/latest_results.json")
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _json_value(value: Any) -> Any:
    """Convert common NumPy/Pandas scalar values to JSON-compatible values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def _aggregate_metric_scores(
    metric_rows: list[dict[str, Any]], expected_question_count: int
) -> dict[str, float]:
    """Reject missing/non-finite metric outputs instead of turning them into scores."""
    if len(metric_rows) != expected_question_count:
        raise RuntimeError(
            "RAGAS evaluation failed: expected metric results for "
            f"{expected_question_count} questions but received {len(metric_rows)}. "
            "No benchmark score was produced."
        )

    values_by_metric: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    failed_rows: dict[str, list[int]] = {name: [] for name in METRIC_NAMES}
    for row_index, row in enumerate(metric_rows, start=1):
        for metric in METRIC_NAMES:
            value = row.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
            ):
                failed_rows[metric].append(row_index)
            else:
                values_by_metric[metric].append(float(value))

    failed = {
        metric: rows for metric, rows in failed_rows.items() if rows
    }
    if failed:
        details = ", ".join(
            f"{metric} (invalid rows: {rows})" for metric, rows in failed.items()
        )
        raise RuntimeError(
            "RAGAS evaluation failed: missing or non-finite metric outputs for "
            f"{details}. No benchmark score was produced."
        )

    return {
        metric: round(sum(values) / len(values), 4)
        for metric, values in values_by_metric.items()
    }


def _write_results(question_results: list[dict[str, Any]]) -> None:
    RESULTS_FILE.write_text(
        json.dumps(question_results, indent=2, default=_json_value)
    )


def run_evaluation() -> dict[str, float]:
    # Invalidate prior outputs so an interrupted run cannot leave stale scores
    # appearing to describe the current benchmark execution.
    SCORES_FILE.unlink(missing_ok=True)
    RESULTS_FILE.unlink(missing_ok=True)

    from app.agent import ask

    logger.info(
        f"[rag-pipeline] Running ask() for all {len(EVAL_DATASET)} questions..."
    )
    questions, answers, contexts, ground_truths = [], [], [], []
    question_results = []

    for item in EVAL_DATASET:
        result = ask(item["question"])
        sources = result["sources"]
        context_texts = result.get(
            "generation_context", [source["text"] for source in sources]
        )

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])
        question_results.append({
            **item,
            "answer": result["answer"],
            "contexts": context_texts,
            "sources": sources,
            "route": result.get("route", "unknown"),
            "rewrite_count": result.get("rewrite_count", 0),
            "web_used": result.get("used_web", False),
            "evidence_source": result.get("evidence_source", "unknown"),
            "diagnostics": result.get("diagnostics", {}),
            "agent": {
                "route": result.get("route", "unknown"),
                "rewrite_count": result.get("rewrite_count", 0),
                "tavily_used": result.get("used_web", False),
            },
            "evaluation_status": "pending",
            "metrics": {},
        })
        _write_results(question_results)

    routes = Counter(item["agent"]["route"] for item in question_results)
    logger.info(
        f"[rag-pipeline] Completed {len(question_results)}/{len(EVAL_DATASET)} "
        f"questions; routes={dict(routes)}"
    )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    logger.info("[ragas] Starting four metric evaluations...")
    try:
        ragas_llm = build_ragas_llm()
        ragas_embeddings = BgeM3EvaluationEmbeddings()
        ragas_result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=True,
        )
        metric_rows = ragas_result.to_pandas().to_dict(orient="records")
        scores = _aggregate_metric_scores(metric_rows, len(EVAL_DATASET))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for question_result in question_results:
            question_result["evaluation_status"] = "failed"
            question_result["evaluation_error"] = error
            question_result["metrics"] = {}
        _write_results(question_results)
        raise RuntimeError(
            "RAGAS evaluation failed: metric execution or result validation failed. "
            "No benchmark score was produced."
        ) from exc

    for question_result, metric_row in zip(question_results, metric_rows):
        question_result["evaluation_status"] = "completed"
        question_result["metrics"] = {
            metric: _json_value(metric_row[metric]) for metric in METRIC_NAMES
        }

    # Write scores only after all four metrics returned finite values for all rows.
    SCORES_FILE.write_text(json.dumps(scores, indent=2))
    _write_results(question_results)
    logger.success(f"[ragas] Scores saved to {SCORES_FILE}")
    logger.success(f"Per-question results saved to {RESULTS_FILE}")

    print("\nRAGAS Evaluation Results")
    for metric in METRIC_NAMES:
        print(f"  {metric}: {scores[metric]:.4f}")
    print()
    return scores


if __name__ == "__main__":
    run_evaluation()
