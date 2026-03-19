#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_policy_pilot.runner import EvalConfig, run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible agent policy pilot on HotpotQA dev.")
    parser.add_argument("--hotpot_dev_path", type=str, required=True, help="Path to HotpotQA dev .json file.")
    parser.add_argument("--output_dir", type=str, default="outputs/pilot", help="Output directory.")
    parser.add_argument("--model_name", type=str, required=True, help="Chat model name.")
    parser.add_argument("--model_base_url", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--model_api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--corpus_jsonl_path",
        type=str,
        default=None,
        help="Optional local corpus jsonl path with fields: pid,title,text.",
    )
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_tool_calls", type=str, default="1,3", help="Comma-separated budgets, e.g. 1,3.")
    parser.add_argument(
        "--methods",
        type=str,
        default="direct,workflow,react,threshold",
        help="Comma-separated: direct,workflow,react,threshold",
    )
    parser.add_argument("--threshold", type=float, default=0.15, help="Threshold policy trigger threshold.")
    parser.add_argument("--cost_weight", type=float, default=0.35, help="Threshold policy cost weight.")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = tuple(int(x.strip()) for x in args.max_tool_calls.split(",") if x.strip())
    methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    cfg = EvalConfig(
        hotpot_dev_path=args.hotpot_dev_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_base_url=args.model_base_url,
        model_api_key=args.model_api_key,
        corpus_jsonl_path=args.corpus_jsonl_path,
        sample_size=args.sample_size,
        seed=args.seed,
        topk=args.topk,
        max_tool_calls_list=budgets,
        methods=methods,
        threshold=args.threshold,
        cost_weight=args.cost_weight,
        temperature=args.temperature,
    )
    run_eval(cfg)


if __name__ == "__main__":
    main()
