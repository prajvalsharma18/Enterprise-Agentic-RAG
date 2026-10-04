"""Run RAGAS over the corpus-grounded evaluation dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.agent import ask
from app.llm_provider import create_chat_model
from evaluation.eval_dataset import EVAL_DATASET

load_dotenv()

SCORES_FILE = Path("evaluation/latest_scores.json")
RESULTS_FILE = Path("evaluation/latest_results.json")
SCORES_FILE.parent.mkdir(exist_ok=True)
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

# RAGAS uses the configured provider as its judge model.
RAGAS_LLM = LangchainLLMWrapper(create_chat_model(temperature=0))
for _metric in (faithfulness, answer_relevancy, context_precision, context_recall):
    _metric.llm = RAGAS_LLM  # type: ignore[attr-defined]


def _json_value(value: Any) -> Any:
    """Convert common NumPy/Pandas scalar values to JSON-compatible values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def run_evaluation() -> dict:
    logger.info(f"Running RAGAS evaluation on {len(EVAL_DATASET)} questions...")

    questions, answers, contexts, ground_truths = [], [], [], []
    question_results = []

    for item in EVAL_DATASET:
        result = ask(item["question"])
        sources = result["sources"]
        context_texts = [source["text"] for source in sources]

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])
        question_results.append({
            **item,
            "answer": result["answer"],
            "contexts": context_texts,
            "sources": sources,
            "agent": {
                "rewrite_count": result.get("rewrite_count", 0),
                "tavily_used": result.get("used_web", False),
            },
        })

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    result_frame = ragas_result.to_pandas()
    metric_rows = result_frame.to_dict(orient="records")
    scores = {}
    for metric in METRIC_NAMES:
        if metric in result_frame.columns:
            mean_score = result_frame[metric].dropna().mean()
            if mean_score == mean_score:  # Do not emit NaN for an empty metric.
                scores[metric] = round(float(mean_score), 4)

    for question_result, metric_row in zip(question_results, metric_rows):
        question_result["metrics"] = {
            metric: _json_value(metric_row[metric])
            for metric in METRIC_NAMES
            if metric in metric_row
        }

    SCORES_FILE.write_text(json.dumps(scores, indent=2))
    RESULTS_FILE.write_text(
        json.dumps(question_results, indent=2, default=_json_value)
    )
    logger.success(f"Scores saved to {SCORES_FILE}")
    logger.success(f"Per-question results saved to {RESULTS_FILE}")

    print("\nRAGAS Evaluation Results")
    for metric, score in scores.items():
        print(f"  {metric:<25} {score:.4f}")
    print()
    return scores


if __name__ == "__main__":
    run_evaluation()
