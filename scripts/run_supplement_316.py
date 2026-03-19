#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_policy_pilot.supplement_316 import SupplementConfig, run_supplement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal supplement experiments for 增补实验表316.")
    parser.add_argument("--hotpot_dev_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/supplement_316")
    parser.add_argument("--base_results_dir", type=str, default="outputs/expanded")
    parser.add_argument("--primary_model", type=str, default="llama-3.1-8b-local")
    parser.add_argument("--model_base_url", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--model_api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--corpus_jsonl_path", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_tool_calls", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_completion_tokens", type=int, default=64)
    parser.add_argument("--max_decision_tokens", type=int, default=96)
    parser.add_argument("--token_cost_reference", type=float, default=1200.0)
    parser.add_argument("--latency_cost_reference", type=float, default=1.0)
    parser.add_argument("--analysis_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SupplementConfig(
        hotpot_dev_path=args.hotpot_dev_path,
        output_dir=args.output_dir,
        base_results_dir=args.base_results_dir,
        primary_model=args.primary_model,
        model_base_url=args.model_base_url,
        model_api_key=args.model_api_key,
        corpus_jsonl_path=args.corpus_jsonl_path,
        sample_size=args.sample_size,
        seed=args.seed,
        topk=args.topk,
        max_tool_calls=args.max_tool_calls,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        max_decision_tokens=args.max_decision_tokens,
        token_cost_reference=args.token_cost_reference,
        latency_cost_reference=args.latency_cost_reference,
        analysis_only=args.analysis_only,
    )
    run_supplement(cfg)


if __name__ == "__main__":
    main()
