#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric, GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall


REQUIRED_COLUMNS = ["input", "actual_output", "expected_output", "tools_called", "expected_tools"]
METRIC_COLUMNS = ["answer_relevancy", "tool_correctness", "task_completion"]
DEFAULT_G_EVAL_CRITERIA = {
    "answer_relevancy": (
        "Determine whether ACTUAL_OUTPUT answers INPUT directly, correctly, and concisely."
    ),
    "tool_correctness": (
        "Determine whether TOOLS_CALLED is appropriate for INPUT and aligns with EXPECTED_TOOLS. "
        "Penalize missing critical tools and unnecessary tool calls."
    ),
    "task_completion": (
        "Determine if the agent completes the requested task correctly, covers key requirements, "
        "and avoids unnecessary or incorrect steps."
    ),
}
DEFAULT_G_EVAL_PARAMS = {
    "answer_relevancy": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    "tool_correctness": [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.TOOLS_CALLED,
        LLMTestCaseParams.EXPECTED_TOOLS,
    ],
    "task_completion": [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
}


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
        name=f"Agent {metric_name}",
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=DEFAULT_G_EVAL_PARAMS[metric_name],
        **kwargs,
    )


def build_metrics(threshold: float, model: str | None, prompt_config: Optional[Dict[str, Any]] = None):
    prompt_config = prompt_config or {}

    metric_kwargs: Dict[str, Any] = {"threshold": threshold, "include_reason": True}
    geval_kwargs: Dict[str, Any] = {"threshold": threshold}
    if model:
        metric_kwargs["model"] = model
        geval_kwargs["model"] = model

    metric_builders = {
        "answer_relevancy": lambda: AnswerRelevancyMetric(**metric_kwargs),
        "tool_correctness": lambda: ToolCorrectnessMetric(**metric_kwargs),
        "task_completion": lambda: GEval(
            name="Task Completion",
            criteria=DEFAULT_G_EVAL_CRITERIA["task_completion"],
            evaluation_params=DEFAULT_G_EVAL_PARAMS["task_completion"],
            **geval_kwargs,
        ),
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
    parser.add_argument(
        "--prompt-config",
        default=None,
        help="Optional JSON file for custom metric prompts (section key: agent).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df, REQUIRED_COLUMNS)
    prompt_config = load_prompt_config(args.prompt_config, "agent")
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
