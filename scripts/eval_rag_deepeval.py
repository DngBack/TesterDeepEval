#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


REQUIRED_COLUMNS = ["input", "actual_output", "expected_output", "retrieval_context"]
METRIC_COLUMNS = [
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
]
DEFAULT_G_EVAL_CRITERIA = {
    "answer_relevancy": (
        "Determine whether ACTUAL_OUTPUT answers INPUT directly and stays on-topic. "
        "Penalize irrelevant or missing content."
    ),
    "faithfulness": (
        "Determine whether ACTUAL_OUTPUT is faithful to RETRIEVAL_CONTEXT, without hallucinated facts."
    ),
    "contextual_precision": (
        "Determine whether RETRIEVAL_CONTEXT is precise for producing EXPECTED_OUTPUT from INPUT, "
        "with minimal irrelevant context."
    ),
    "contextual_recall": (
        "Determine whether RETRIEVAL_CONTEXT covers enough information to support EXPECTED_OUTPUT."
    ),
    "contextual_relevancy": (
        "Determine whether RETRIEVAL_CONTEXT is relevant to INPUT and useful for answering it."
    ),
}
DEFAULT_G_EVAL_PARAMS = {
    "answer_relevancy": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    "faithfulness": [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    "contextual_precision": [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    "contextual_recall": [LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    "contextual_relevancy": [LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
}


def parse_json_list(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]

    raw = str(value).strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    return [part.strip() for part in raw.split("||") if part.strip()]


def load_prompt_config(path: Optional[str], section: str) -> Dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Prompt config must be a JSON object.")

    section_data = raw.get(section, raw)
    if section_data is None:
        return {}
    if not isinstance(section_data, dict):
        raise ValueError(f"Prompt config section '{section}' must be a JSON object.")
    return section_data


def _build_geval_metric(
    metric_name: str,
    spec: Any,
    threshold: float,
    model: Optional[str],
):
    criteria = DEFAULT_G_EVAL_CRITERIA[metric_name]
    evaluation_steps = None

    if isinstance(spec, str):
        criteria = spec
    elif isinstance(spec, dict):
        criteria = str(spec.get("criteria", criteria))
        raw_steps = spec.get("evaluation_steps")
        if raw_steps is not None:
            if not isinstance(raw_steps, list) or not all(isinstance(step, str) for step in raw_steps):
                raise ValueError(f"'{metric_name}.evaluation_steps' must be a list of strings.")
            evaluation_steps = raw_steps
    else:
        raise ValueError(f"Prompt config for metric '{metric_name}' must be a string or object.")

    kwargs: Dict[str, Any] = {"threshold": threshold}
    if model:
        kwargs["model"] = model

    return GEval(
        name=f"RAG {metric_name}",
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=DEFAULT_G_EVAL_PARAMS[metric_name],
        **kwargs,
    )


def build_metrics(threshold: float, model: str | None, prompt_config: Optional[Dict[str, Any]] = None):
    prompt_config = prompt_config or {}

    kwargs: Dict[str, Any] = {"threshold": threshold, "include_reason": True}
    if model:
        kwargs["model"] = model

    metric_builders = {
        "answer_relevancy": lambda: AnswerRelevancyMetric(**kwargs),
        "faithfulness": lambda: FaithfulnessMetric(**kwargs),
        "contextual_precision": lambda: ContextualPrecisionMetric(**kwargs),
        "contextual_recall": lambda: ContextualRecallMetric(**kwargs),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(**kwargs),
    }

    metrics = []
    for name in METRIC_COLUMNS:
        if name in prompt_config:
            metrics.append((name, _build_geval_metric(name, prompt_config[name], threshold, model)))
        else:
            metrics.append((name, metric_builders[name]()))
    return metrics


def evaluate_row(row: pd.Series, metrics) -> Dict[str, Any]:
    test_case = LLMTestCase(
        input=str(row["input"]),
        actual_output=str(row["actual_output"]),
        expected_output=str(row["expected_output"]),
        retrieval_context=parse_json_list(row["retrieval_context"]),
    )

    scores: Dict[str, Any] = {}
    for name, metric in metrics:
        try:
            metric.measure(test_case)
            scores[name] = float(metric.score)
            scores[f"{name}_reason"] = metric.reason
            scores[f"{name}_passed"] = bool(metric.success)
        except Exception as exc:  # noqa: BLE001
            scores[name] = None
            scores[f"{name}_reason"] = f"metric_error: {exc}"
            scores[f"{name}_passed"] = False

    numeric_scores = [scores[col] for col in METRIC_COLUMNS if isinstance(scores[col], (int, float))]
    scores["overall_score"] = float(sum(numeric_scores) / len(numeric_scores)) if numeric_scores else None
    return scores


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate RAG outputs with DeepEval.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", default="outputs/rag_scores.csv", help="Detailed output CSV path.")
    parser.add_argument("--summary", default="outputs/rag_summary.csv", help="Summary output CSV path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pass threshold for metrics.")
    parser.add_argument("--model", default=None, help="Judge model override (e.g. gpt-4.1).")
    parser.add_argument(
        "--prompt-config",
        default=None,
        help="Optional JSON file for custom metric prompts (section key: rag).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df, REQUIRED_COLUMNS)
    prompt_config = load_prompt_config(args.prompt_config, "rag")
    metrics = build_metrics(args.threshold, args.model, prompt_config)

    results = []
    for idx, row in df.iterrows():
        row_id = row["id"] if "id" in df.columns else idx + 1
        score_data = evaluate_row(row, metrics)
        results.append({"id": row_id, **score_data})

    result_df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    summary_dict: Dict[str, Any] = {"rows": len(result_df)}
    for col in METRIC_COLUMNS + ["overall_score"]:
        summary_dict[f"avg_{col}"] = (
            float(result_df[col].mean(skipna=True)) if col in result_df.columns else None
        )
    for col in METRIC_COLUMNS:
        pass_col = f"{col}_passed"
        summary_dict[f"pass_rate_{col}"] = (
            float(result_df[pass_col].mean(skipna=True)) if pass_col in result_df.columns else None
        )

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary_dict]).to_csv(summary_path, index=False)

    print(f"Saved detailed scores to: {output_path}")
    print(f"Saved summary scores to: {summary_path}")


if __name__ == "__main__":
    main()
