#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric, GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall


REQUIRED_COLUMNS = ["input", "actual_output", "expected_output", "tools_called", "expected_tools"]
METRIC_COLUMNS = ["answer_relevancy", "tool_correctness", "task_completion"]


def parse_json_list(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value

    raw = str(value).strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    return [part.strip() for part in raw.split("||") if part.strip()]


def parse_tool_calls(value: Any) -> List[ToolCall]:
    items = parse_json_list(value)
    tool_calls: List[ToolCall] = []

    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                tool_calls.append(ToolCall(name=name))
        else:
            name = str(item).strip()
            if name:
                tool_calls.append(ToolCall(name=name))
    return tool_calls


def build_metrics(threshold: float, model: str | None):
    common_kwargs: Dict[str, Any] = {"threshold": threshold, "include_reason": True}
    if model:
        common_kwargs["model"] = model

    task_completion = GEval(
        name="Task Completion",
        criteria=(
            "Determine if the agent completes the requested task correctly, "
            "covers key requirements, and avoids unnecessary or incorrect steps."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        **common_kwargs,
    )

    return [
        ("answer_relevancy", AnswerRelevancyMetric(**common_kwargs)),
        ("tool_correctness", ToolCorrectnessMetric(**common_kwargs)),
        ("task_completion", task_completion),
    ]


def evaluate_row(row: pd.Series, metrics) -> Dict[str, Any]:
    test_case = LLMTestCase(
        input=str(row["input"]),
        actual_output=str(row["actual_output"]),
        expected_output=str(row["expected_output"]),
        tools_called=parse_tool_calls(row["tools_called"]),
        expected_tools=parse_tool_calls(row["expected_tools"]),
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

    parser = argparse.ArgumentParser(description="Evaluate agent outputs with DeepEval.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", default="outputs/agent_scores.csv", help="Detailed output CSV path.")
    parser.add_argument("--summary", default="outputs/agent_summary.csv", help="Summary output CSV path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pass threshold for metrics.")
    parser.add_argument("--model", default=None, help="Judge model override (e.g. gpt-4.1).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df, REQUIRED_COLUMNS)
    metrics = build_metrics(args.threshold, args.model)

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
