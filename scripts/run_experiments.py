#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_policy_pilot.experiments import ExperimentConfig, run_all_experiments


def _int_tuple(csv_text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in csv_text.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded agent-policy experiments.")
    parser.add_argument("--hotpot_dev_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/expanded")
    parser.add_argument("--primary_model", type=str, required=True)
    parser.add_argument("--secondary_model", type=str, default=None)
    parser.add_argument("--model_base_url", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--model_api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--corpus_jsonl_path", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--token_budgets", type=str, default="400,600,800,1000,1500,2000")
    parser.add_argument("--react_steps", type=str, default="1,2,3,4,5")
    parser.add_argument("--main_tool_calls", type=str, default="1,3,5")
    parser.add_argument("--max_completion_tokens", type=int, default=64)
    parser.add_argument("--max_decision_tokens", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        hotpot_dev_path=args.hotpot_dev_path,
        output_dir=args.output_dir,
        primary_model=args.primary_model,
        secondary_model=args.secondary_model,
        model_base_url=args.model_base_url,
        model_api_key=args.model_api_key,
        corpus_jsonl_path=args.corpus_jsonl_path,
        sample_size=args.sample_size,
        seed=args.seed,
        topk=args.topk,
        temperature=args.temperature,
        token_budgets=_int_tuple(args.token_budgets),
        react_steps=_int_tuple(args.react_steps),
        main_tool_calls=_int_tuple(args.main_tool_calls),
        max_completion_tokens=args.max_completion_tokens,
        max_decision_tokens=args.max_decision_tokens,
    )
    run_all_experiments(cfg)


if __name__ == "__main__":
    main()

