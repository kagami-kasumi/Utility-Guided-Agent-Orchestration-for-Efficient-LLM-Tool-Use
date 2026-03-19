from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tqdm import tqdm

from .data import (
    HotpotExample,
    build_corpus_from_contexts,
    load_hotpot_dev,
    load_passages_jsonl,
    sample_examples,
)
from .llm import OpenAIChatModel
from .metrics import exact_match_score, f1_score
from .policies import (
    BasePolicy,
    DirectPolicy,
    ReActPolicy,
    ThresholdPolicy,
    WorkflowSearchOncePolicy,
)
from .search import BM25Searcher


@dataclass(frozen=True)
class EvalConfig:
    hotpot_dev_path: str
    output_dir: str
    model_name: str
    model_base_url: str | None
    model_api_key: str | None
    corpus_jsonl_path: str | None = None
    sample_size: int = 200
    seed: int = 42
    topk: int = 3
    max_tool_calls_list: tuple[int, ...] = (1, 3)
    methods: tuple[str, ...] = ("direct", "workflow", "react", "threshold")
    threshold: float = 0.15
    cost_weight: float = 0.35
    temperature: float = 0.0


def _policy_factory(method: str, max_tool_calls: int, topk: int, threshold: float, cost_weight: float) -> BasePolicy:
    if method == "direct":
        return DirectPolicy(max_tool_calls=max_tool_calls, topk=topk)
    if method == "workflow":
        return WorkflowSearchOncePolicy(max_tool_calls=max_tool_calls, topk=topk)
    if method == "react":
        return ReActPolicy(max_tool_calls=max_tool_calls, topk=topk)
    if method == "threshold":
        return ThresholdPolicy(
            max_tool_calls=max_tool_calls,
            topk=topk,
            threshold=threshold,
            cost_weight=cost_weight,
        )
    raise ValueError(f"Unknown method: {method}")


def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    trace_dir = root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return root, trace_dir


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_pareto(points: list[dict[str, Any]], cost_key: str, quality_key: str) -> set[str]:
    frontier: set[str] = set()
    for p in points:
        dominated = False
        for q in points:
            if p["run_id"] == q["run_id"]:
                continue
            cheaper_or_equal = q[cost_key] <= p[cost_key]
            better_or_equal = q[quality_key] >= p[quality_key]
            strictly_better_one = q[cost_key] < p[cost_key] or q[quality_key] > p[quality_key]
            if cheaper_or_equal and better_or_equal and strictly_better_one:
                dominated = True
                break
        if not dominated:
            frontier.add(p["run_id"])
    return frontier


def _plot_pareto(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    frontier_token = _compute_pareto(summary_rows, "avg_tokens", "avg_f1")
    frontier_time = _compute_pareto(summary_rows, "avg_wall_time", "avg_f1")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    configs = [
        (axes[0], "avg_tokens", "F1 vs Tokens", frontier_token),
        (axes[1], "avg_wall_time", "F1 vs Wall Time", frontier_time),
    ]
    for ax, x_key, title, frontier in configs:
        for row in summary_rows:
            label = f'{row["method"]}@{row["max_tool_calls"]}'
            marker = "o" if row["run_id"] in frontier else "x"
            size = 50 + 30 * row["avg_tool_calls"]
            ax.scatter(row[x_key], row["avg_f1"], s=size, marker=marker)
            ax.annotate(label, (row[x_key], row["avg_f1"]), fontsize=8)
        ax.set_xlabel(x_key)
        ax.set_ylabel("avg_f1")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_eval(config: EvalConfig) -> None:
    output_root, trace_dir = _ensure_dirs(Path(config.output_dir))
    _write_json(output_root / "run_config.json", asdict(config))

    examples = load_hotpot_dev(config.hotpot_dev_path)
    sample = sample_examples(examples, sample_size=config.sample_size, seed=config.seed)
    _write_json(
        output_root / "sampled_ids.json",
        {"sample_size": len(sample), "seed": config.seed, "ids": [ex.qid for ex in sample]},
    )

    if config.corpus_jsonl_path:
        corpus = load_passages_jsonl(config.corpus_jsonl_path)
    else:
        corpus = build_corpus_from_contexts(examples)
    searcher = BM25Searcher(corpus)
    model = OpenAIChatModel(
        model=config.model_name,
        base_url=config.model_base_url,
        api_key=config.model_api_key,
        temperature=config.temperature,
        seed=config.seed,
    )

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    methods = list(config.methods)
    budgets = list(config.max_tool_calls_list)

    for max_tool_calls in budgets:
        for method in methods:
            policy = _policy_factory(
                method=method,
                max_tool_calls=max_tool_calls,
                topk=config.topk,
                threshold=config.threshold,
                cost_weight=config.cost_weight,
            )
            run_id = f"{policy.name}_k{config.topk}_m{max_tool_calls}"
            trace_path = trace_dir / f"{run_id}.jsonl"
            with trace_path.open("w", encoding="utf-8") as trace_f:
                em_sum = 0.0
                f1_sum = 0.0
                token_sum = 0
                tool_sum = 0
                wall_sum = 0.0

                pbar = tqdm(sample, desc=f"{run_id}", ncols=110)
                for ex in pbar:
                    t0 = time.perf_counter()
                    result = policy.run(ex=ex, model=model, searcher=searcher)
                    wall = time.perf_counter() - t0

                    em = exact_match_score(result.prediction, ex.answer)
                    f1 = f1_score(result.prediction, ex.answer)
                    em_sum += em
                    f1_sum += f1
                    token_sum += result.token_total
                    tool_sum += result.tool_calls
                    wall_sum += wall

                    trace_row = {
                        "run_id": run_id,
                        "method": method,
                        "max_tool_calls": max_tool_calls,
                        "sample_id": ex.qid,
                        "question": ex.question,
                        "gold_answer": ex.answer,
                        "prediction": result.prediction,
                        "em": em,
                        "f1": f1,
                        "tokens": result.token_total,
                        "tool_calls": result.tool_calls,
                        "wall_time": wall,
                        "trajectory": result.trajectory,
                    }
                    trace_f.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
                    detail_rows.append(
                        {
                            "run_id": run_id,
                            "method": method,
                            "max_tool_calls": max_tool_calls,
                            "sample_id": ex.qid,
                            "em": em,
                            "f1": f1,
                            "tokens": result.token_total,
                            "tool_calls": result.tool_calls,
                            "wall_time": wall,
                        }
                    )

                n = len(sample)
                row = {
                    "run_id": run_id,
                    "method": method,
                    "max_tool_calls": max_tool_calls,
                    "topk": config.topk,
                    "num_samples": n,
                    "avg_em": em_sum / n,
                    "avg_f1": f1_sum / n,
                    "avg_tokens": token_sum / n,
                    "avg_tool_calls": tool_sum / n,
                    "avg_wall_time": wall_sum / n,
                }
                summary_rows.append(row)

    summary_rows.sort(key=lambda x: (x["max_tool_calls"], x["method"]))

    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary_rows)

    summary_csv = output_root / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_csv = output_root / "per_sample_metrics.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    _plot_pareto(summary_rows, output_root / "pareto.png")
