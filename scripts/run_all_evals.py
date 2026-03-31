#!/usr/bin/env python3
import argparse
import subprocess
import sys


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run both RAG and Agent DeepEval scripts.")
    parser.add_argument("--rag-input", default="data/rag_input_sample.csv")
    parser.add_argument("--agent-input", default="data/agent_input_sample.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--rag-prompt-config", default=None)
    parser.add_argument("--agent-prompt-config", default=None)
    parser.add_argument(
        "--prompt-config",
        default=None,
        help="Shortcut to use the same prompt config file for both RAG and Agent.",
    )
    args = parser.parse_args()

    rag_cmd = [
        sys.executable,
        "scripts/eval_rag_deepeval.py",
        "--input",
        args.rag_input,
        "--output",
        "outputs/rag_scores.csv",
        "--summary",
        "outputs/rag_summary.csv",
        "--threshold",
        str(args.threshold),
    ]
    agent_cmd = [
        sys.executable,
        "scripts/eval_agent_deepeval.py",
        "--input",
        args.agent_input,
        "--output",
        "outputs/agent_scores.csv",
        "--summary",
        "outputs/agent_summary.csv",
        "--threshold",
        str(args.threshold),
    ]

    if args.model:
        rag_cmd.extend(["--model", args.model])
        agent_cmd.extend(["--model", args.model])

    rag_prompt_config = args.rag_prompt_config or args.prompt_config
    agent_prompt_config = args.agent_prompt_config or args.prompt_config
    if rag_prompt_config:
        rag_cmd.extend(["--prompt-config", rag_prompt_config])
    if agent_prompt_config:
        agent_cmd.extend(["--prompt-config", agent_prompt_config])

    run_cmd(rag_cmd)
    run_cmd(agent_cmd)

    print("Done. Check outputs/ for numeric results.")


if __name__ == "__main__":
    main()
