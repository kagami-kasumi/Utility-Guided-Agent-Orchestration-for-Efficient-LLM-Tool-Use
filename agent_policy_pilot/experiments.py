from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tqdm import tqdm

from .data import HotpotExample, build_corpus_from_contexts, load_hotpot_dev, load_passages_jsonl, sample_examples
from .llm import OpenAIChatModel
from .metrics import exact_match_score, f1_score
from .policies import (
    BasePolicy,
    DirectPolicy,
    ReActPolicy,
    ThresholdPolicy,
    UtilityPolicy,
    WorkflowSearchOncePolicy,
    WorkflowSearchTwicePolicy,
    WorkflowSearchVerifyPolicy,
)
from .search import BM25Searcher
from .visualization import method_color, method_marker, pareto_frontier, save_figure_all, setup_plot_style


@dataclass(frozen=True)
class ExperimentConfig:
    hotpot_dev_path: str
    output_dir: str
    primary_model: str
    secondary_model: str | None
    model_base_url: str | None
    model_api_key: str | None
    corpus_jsonl_path: str | None = None
    sample_size: int = 200
    seed: int = 42
    topk: int = 3
    temperature: float = 0.0
    token_budgets: tuple[int, ...] = (400, 600, 800, 1000, 1500, 2000)
    react_steps: tuple[int, ...] = (1, 2, 3, 4, 5)
    main_tool_calls: tuple[int, ...] = (1, 3, 5)
    max_completion_tokens: int = 64
    max_decision_tokens: int = 96


@dataclass(frozen=True)
class RunCase:
    experiment: str
    run_id: str
    method: str
    model_name: str
    max_tool_calls: int
    token_budget: int | None = None
    lambda_cost: float = 0.3
    lambda_uncertainty: float = 0.2
    lambda_redundancy: float = 0.2
    cost_mode: str = "step"
    redundancy_mode: str = "exact"
    token_cost_reference: float = 1200.0
    latency_cost_reference: float = 1.0
    use_expected_gain: bool = True
    use_uncertainty: bool = True
    use_redundancy: bool = True
    use_stop_policy: bool = True


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_dirs(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": output_root,
        "traces": output_root / "traces",
        "figures": output_root / "figures",
        "tables": output_root / "tables",
    }
    for p in paths.values():
        if p != output_root:
            p.mkdir(parents=True, exist_ok=True)
    return paths


def _policy_from_case(
    case: RunCase,
    topk: int,
    max_completion_tokens: int,
    max_decision_tokens: int,
) -> BasePolicy:
    common = {
        "max_tool_calls": case.max_tool_calls,
        "topk": topk,
        "token_budget": case.token_budget,
        "max_completion_tokens": max_completion_tokens,
        "max_decision_tokens": max_decision_tokens,
    }
    if case.method == "direct":
        return DirectPolicy(**common)
    if case.method == "workflow":
        return WorkflowSearchOncePolicy(**common)
    if case.method == "workflow-search-twice":
        return WorkflowSearchTwicePolicy(**common)
    if case.method == "workflow-search-verify":
        return WorkflowSearchVerifyPolicy(**common)
    if case.method == "react":
        return ReActPolicy(**common)
    if case.method == "threshold":
        return ThresholdPolicy(**common)
    if case.method == "policy":
        return UtilityPolicy(
            **common,
            lambda_cost=case.lambda_cost,
            lambda_uncertainty=case.lambda_uncertainty,
            lambda_redundancy=case.lambda_redundancy,
            cost_mode=case.cost_mode,
            redundancy_mode=case.redundancy_mode,
            token_cost_reference=case.token_cost_reference,
            latency_cost_reference=case.latency_cost_reference,
            use_expected_gain=case.use_expected_gain,
            use_uncertainty=case.use_uncertainty,
            use_redundancy=case.use_redundancy,
            use_stop_policy=case.use_stop_policy,
        )
    raise ValueError(f"Unknown method {case.method}")


def _trace_stats(trajectory: list[dict[str, Any]]) -> tuple[int, int, int]:
    steps = sum(1 for x in trajectory if x.get("type") == "model")
    redundant_flags = [x.get("is_redundant") for x in trajectory if x.get("type") == "tool"]
    if any(flag is not None for flag in redundant_flags):
        redundant = sum(1 for flag in redundant_flags if flag)
    else:
        queries = [str(x.get("query", "")).strip().lower() for x in trajectory if x.get("type") == "tool"]
        redundant = max(0, len(queries) - len(set(q for q in queries if q)))
    stop_step = steps
    return steps, redundant, stop_step


def _run_case(
    case: RunCase,
    sample: list[HotpotExample],
    searcher: BM25Searcher,
    model: OpenAIChatModel,
    trace_path: Path,
    topk: int,
    max_completion_tokens: int,
    max_decision_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = _policy_from_case(
        case=case,
        topk=topk,
        max_completion_tokens=max_completion_tokens,
        max_decision_tokens=max_decision_tokens,
    )
    detail_rows: list[dict[str, Any]] = []
    stop_dist: dict[int, int] = {}
    with trace_path.open("w", encoding="utf-8") as trace_f:
        em_sum = 0.0
        f1_sum = 0.0
        tokens_sum = 0
        wall_sum = 0.0
        tool_sum = 0
        step_sum = 0
        redundant_sum = 0

        pbar = tqdm(sample, desc=case.run_id, ncols=120)
        for ex in pbar:
            t0 = time.perf_counter()
            result = policy.run(ex=ex, model=model, searcher=searcher)
            wall = time.perf_counter() - t0

            em = exact_match_score(result.prediction, ex.answer)
            f1 = f1_score(result.prediction, ex.answer)
            steps, redundant, stop_step = _trace_stats(result.trajectory)
            stop_dist[stop_step] = stop_dist.get(stop_step, 0) + 1

            em_sum += em
            f1_sum += f1
            tokens_sum += result.token_total
            wall_sum += wall
            tool_sum += result.tool_calls
            step_sum += steps
            redundant_sum += redundant

            row = {
                "run_id": case.run_id,
                "experiment": case.experiment,
                "method": case.method,
                "model_name": case.model_name,
                "max_tool_calls": case.max_tool_calls,
                "token_budget": case.token_budget,
                "cost_mode": case.cost_mode,
                "redundancy_mode": case.redundancy_mode,
                "sample_id": ex.qid,
                "question": ex.question,
                "gold_answer": ex.answer,
                "prediction": result.prediction,
                "em": em,
                "f1": f1,
                "tokens": result.token_total,
                "tool_calls": result.tool_calls,
                "steps": steps,
                "redundant_tool_calls": redundant,
                "stop_step": stop_step,
                "wall_time": wall,
                "trajectory": result.trajectory,
            }
            trace_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            detail_rows.append(
                {
                    "run_id": case.run_id,
                    "experiment": case.experiment,
                    "method": case.method,
                    "model_name": case.model_name,
                    "max_tool_calls": case.max_tool_calls,
                    "token_budget": case.token_budget,
                    "cost_mode": case.cost_mode,
                    "redundancy_mode": case.redundancy_mode,
                    "sample_id": ex.qid,
                    "f1": f1,
                    "tokens": result.token_total,
                    "wall_time": wall,
                    "tool_calls": result.tool_calls,
                    "steps": steps,
                    "redundant_tool_calls": redundant,
                    "stop_step": stop_step,
                }
            )

    n = len(sample)
    summary = {
        "run_id": case.run_id,
        "experiment": case.experiment,
        "method": case.method,
        "model_name": case.model_name,
        "max_tool_calls": case.max_tool_calls,
        "token_budget": case.token_budget,
        "cost_mode": case.cost_mode,
        "redundancy_mode": case.redundancy_mode,
        "num_samples": n,
        "avg_em": em_sum / n,
        "avg_f1": f1_sum / n,
        "avg_tokens": tokens_sum / n,
        "avg_wall_time": wall_sum / n,
        "avg_tool_calls": tool_sum / n,
        "tool_calls": tool_sum / n,
        "avg_steps": step_sum / n,
        "redundant_tool_calls": redundant_sum / n,
        "efficiency": (f1_sum / n) / max(1.0, (tokens_sum / n)),
        "f1": f1_sum / n,
        "token_usage": tokens_sum / n,
        "wall_time": wall_sum / n,
        "stop_step_distribution": stop_dist,
        "lambda_cost": case.lambda_cost,
        "lambda_uncertainty": case.lambda_uncertainty,
        "lambda_redundancy": case.lambda_redundancy,
        "token_cost_reference": case.token_cost_reference,
        "latency_cost_reference": case.latency_cost_reference,
        "use_expected_gain": case.use_expected_gain,
        "use_uncertainty": case.use_uncertainty,
        "use_redundancy": case.use_redundancy,
        "use_stop_policy": case.use_stop_policy,
    }
    return summary, detail_rows


def _plot_react_steps(rows: list[dict[str, Any]], out_dir: Path) -> None:
    setup_plot_style()
    data = sorted(rows, key=lambda x: x["max_tool_calls"])
    steps = [r["max_tool_calls"] for r in data]
    f1 = [r["avg_f1"] for r in data]
    tokens = [r["avg_tokens"] for r in data]
    latency = [r["avg_wall_time"] for r in data]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps, f1, marker="o", color="orange")
    ax.set_xlabel("Reasoning Steps (max_tool_calls)")
    ax.set_ylabel("avg_f1")
    ax.set_title("Step vs F1")
    ax.grid(alpha=0.3)
    save_figure_all(fig, out_dir / "react_step_vs_f1")
    save_figure_all(fig, out_dir / "figure3_step_analysis")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps, tokens, marker="o", color="orange")
    ax.set_xlabel("Reasoning Steps (max_tool_calls)")
    ax.set_ylabel("avg_tokens")
    ax.set_title("Step vs Tokens")
    ax.grid(alpha=0.3)
    save_figure_all(fig, out_dir / "react_step_vs_tokens")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps, latency, marker="o", color="orange")
    ax.set_xlabel("Reasoning Steps (max_tool_calls)")
    ax.set_ylabel("avg_wall_time")
    ax.set_title("Step vs Latency")
    ax.grid(alpha=0.3)
    save_figure_all(fig, out_dir / "react_step_vs_time")
    plt.close(fig)


def _plot_budget(rows: list[dict[str, Any]], out_dir: Path) -> None:
    setup_plot_style()
    methods = sorted(set(r["method"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for method in methods:
        mrows = sorted([r for r in rows if r["method"] == method], key=lambda x: x["token_budget"])
        budgets = [r["token_budget"] for r in mrows]
        axes[0].plot(budgets, [r["avg_f1"] for r in mrows], label=method, color=method_color(method), marker="o")
        axes[1].plot(budgets, [r["avg_tokens"] for r in mrows], label=method, color=method_color(method), marker="o")
        axes[2].plot(budgets, [r["avg_wall_time"] for r in mrows], label=method, color=method_color(method), marker="o")

    axes[0].set_title("Budget vs F1")
    axes[0].set_xlabel("token_budget")
    axes[0].set_ylabel("avg_f1")
    axes[1].set_title("Budget vs Tokens")
    axes[1].set_xlabel("token_budget")
    axes[1].set_ylabel("avg_tokens")
    axes[2].set_title("Budget vs Latency")
    axes[2].set_xlabel("token_budget")
    axes[2].set_ylabel("avg_wall_time")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    save_figure_all(fig, out_dir / "figure4_budget_analysis")
    plt.close(fig)


def _plot_pareto(rows: list[dict[str, Any]], out_dir: Path) -> None:
    setup_plot_style()
    token_front = pareto_frontier(rows, "avg_tokens", "avg_f1")
    time_front = pareto_frontier(rows, "avg_wall_time", "avg_f1")

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        marker = method_marker(r["max_tool_calls"])
        face = method_color(r["method"])
        edge = "black" if r["run_id"] in token_front else "none"
        ax.scatter(r["avg_tokens"], r["avg_f1"], c=face, marker=marker, s=130, edgecolors=edge)
        ax.annotate(f'{r["method"]}@{r["max_tool_calls"]}', (r["avg_tokens"], r["avg_f1"]))
    ax.set_xlabel("avg_tokens")
    ax.set_ylabel("avg_f1")
    ax.set_title("F1 vs Tokens (Pareto)")
    ax.grid(alpha=0.3)
    save_figure_all(fig, out_dir / "figure1_pareto_tokens")
    save_figure_all(fig, out_dir / "pareto_tokens")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        marker = method_marker(r["max_tool_calls"])
        face = method_color(r["method"])
        edge = "black" if r["run_id"] in time_front else "none"
        ax.scatter(r["avg_wall_time"], r["avg_f1"], c=face, marker=marker, s=130, edgecolors=edge)
        ax.annotate(f'{r["method"]}@{r["max_tool_calls"]}', (r["avg_wall_time"], r["avg_f1"]))
    ax.set_xlabel("avg_wall_time")
    ax.set_ylabel("avg_f1")
    ax.set_title("F1 vs Wall Time (Pareto)")
    ax.grid(alpha=0.3)
    save_figure_all(fig, out_dir / "figure2_pareto_latency")
    save_figure_all(fig, out_dir / "pareto_time")
    plt.close(fig)


def _plot_efficiency(rows: list[dict[str, Any]], out_dir: Path) -> None:
    setup_plot_style()
    data = sorted(rows, key=lambda x: x["method"])
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = list(range(len(data)))
    ax.bar(xs, [r["efficiency"] for r in data], color=[method_color(r["method"]) for r in data])
    ax.set_xticks(xs, [f'{r["method"]}@{r["max_tool_calls"]}' for r in data], rotation=30, ha="right")
    ax.set_ylabel("efficiency = F1 / tokens")
    ax.set_title("Efficiency vs Method")
    ax.grid(axis="y", alpha=0.3)
    save_figure_all(fig, out_dir / "efficiency_plot")
    plt.close(fig)


def _plot_tradeoff(rows: list[dict[str, Any]], out_dir: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in rows:
        ax.scatter(
            r["avg_tokens"],
            r["avg_f1"],
            s=max(30.0, 1200 * r["avg_wall_time"]),
            c=method_color(r["method"]),
            marker=method_marker(r["max_tool_calls"]),
            alpha=0.45,
        )
        ax.annotate(f'{r["method"]}@{r["max_tool_calls"]}', (r["avg_tokens"], r["avg_f1"]))
    ax.set_xlabel("tokens")
    ax.set_ylabel("f1")
    ax.set_title("Token / Latency Tradeoff")
    ax.grid(alpha=0.3)
    save_figure_all(fig, out_dir / "tradeoff_plot")
    plt.close(fig)


def _plot_policy_ablation(rows: list[dict[str, Any]], out_dir: Path) -> None:
    setup_plot_style()
    data = sorted(rows, key=lambda x: x["run_id"])
    labels = [r["run_id"].replace("ablation_", "") for r in data]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(data)), [r["avg_f1"] for r in data], color="red", alpha=0.75)
    ax.set_xticks(range(len(data)), labels, rotation=20, ha="right")
    ax.set_ylabel("avg_f1")
    ax.set_title("Policy Ablation")
    ax.grid(axis="y", alpha=0.3)
    save_figure_all(fig, out_dir / "figure5_policy_ablation")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latex_table(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{title}}}",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Method & F1 & Tokens & Latency & Tool Calls \\\\",
        "\\hline",
    ]
    for r in rows:
        label = f'{r["method"]}@{r["max_tool_calls"]}'
        lines.append(
            f'{label} & {r["avg_f1"]:.4f} & {r["avg_tokens"]:.1f} & {r["avg_wall_time"]:.3f} & {r["avg_tool_calls"]:.2f} \\\\'
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_cases(config: ExperimentConfig) -> list[RunCase]:
    cases: list[RunCase] = []

    for m in ("direct", "threshold", "workflow", "react", "policy"):
        for k in config.main_tool_calls:
            cases.append(
                RunCase(
                    experiment="main",
                    run_id=f"main_{m}_k{k}_{config.primary_model}",
                    method=m,
                    model_name=config.primary_model,
                    max_tool_calls=k,
                )
            )

    for k in config.react_steps:
        cases.append(
            RunCase(
                experiment="react_steps",
                run_id=f"react_steps_k{k}_{config.primary_model}",
                method="react",
                model_name=config.primary_model,
                max_tool_calls=k,
            )
        )

    for budget in config.token_budgets:
        for m in ("direct", "threshold", "workflow", "react", "policy"):
            cases.append(
                RunCase(
                    experiment="token_budget",
                    run_id=f"budget_{budget}_{m}_{config.primary_model}",
                    method=m,
                    model_name=config.primary_model,
                    max_tool_calls=3,
                    token_budget=budget,
                )
            )

    for l1 in (0.1, 0.3, 0.5, 1.0):
        for l2 in (0.0, 0.2, 0.4):
            for l3 in (0.0, 0.2):
                cases.append(
                    RunCase(
                        experiment="utility_ablation",
                        run_id=f"utility_l1_{l1}_l2_{l2}_l3_{l3}_{config.primary_model}",
                        method="policy",
                        model_name=config.primary_model,
                        max_tool_calls=3,
                        lambda_cost=l1,
                        lambda_uncertainty=l2,
                        lambda_redundancy=l3,
                    )
                )

    cases.extend(
        [
            RunCase(
                experiment="policy_ablation",
                run_id=f"ablation_full_policy_{config.primary_model}",
                method="policy",
                model_name=config.primary_model,
                max_tool_calls=3,
            ),
            RunCase(
                experiment="policy_ablation",
                run_id=f"ablation_no_uncertainty_{config.primary_model}",
                method="policy",
                model_name=config.primary_model,
                max_tool_calls=3,
                use_uncertainty=False,
            ),
            RunCase(
                experiment="policy_ablation",
                run_id=f"ablation_no_redundancy_{config.primary_model}",
                method="policy",
                model_name=config.primary_model,
                max_tool_calls=3,
                use_redundancy=False,
            ),
            RunCase(
                experiment="policy_ablation",
                run_id=f"ablation_no_expected_gain_{config.primary_model}",
                method="policy",
                model_name=config.primary_model,
                max_tool_calls=3,
                use_expected_gain=False,
            ),
            RunCase(
                experiment="policy_ablation",
                run_id=f"ablation_no_stop_policy_{config.primary_model}",
                method="policy",
                model_name=config.primary_model,
                max_tool_calls=3,
                use_stop_policy=False,
            ),
        ]
    )

    if config.secondary_model:
        for model_name in (config.primary_model, config.secondary_model):
            cases.append(
                RunCase(
                    experiment="model_generalization",
                    run_id=f"model_generalization_policy_k3_{model_name}",
                    method="policy",
                    model_name=model_name,
                    max_tool_calls=3,
                )
            )
    return cases


def run_all_experiments(config: ExperimentConfig) -> None:
    paths = _ensure_dirs(Path(config.output_dir))
    _write_json(paths["root"] / "experiment_config.json", asdict(config))

    examples = load_hotpot_dev(config.hotpot_dev_path)
    sample = sample_examples(examples, sample_size=config.sample_size, seed=config.seed)
    _write_json(
        paths["root"] / "sampled_ids.json",
        {"sample_size": len(sample), "seed": config.seed, "ids": [ex.qid for ex in sample]},
    )

    if config.corpus_jsonl_path:
        corpus = load_passages_jsonl(config.corpus_jsonl_path)
    else:
        corpus = build_corpus_from_contexts(examples)
    searcher = BM25Searcher(corpus)

    models: dict[str, OpenAIChatModel] = {}
    for model_name in set([config.primary_model] + ([config.secondary_model] if config.secondary_model else [])):
        models[model_name] = OpenAIChatModel(
            model=model_name,
            base_url=config.model_base_url,
            api_key=config.model_api_key,
            temperature=config.temperature,
            seed=config.seed,
        )

    cases = _build_cases(config)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for case in cases:
        trace_path = paths["traces"] / f"{case.run_id}.jsonl"
        summary, details = _run_case(
            case=case,
            sample=sample,
            searcher=searcher,
            model=models[case.model_name],
            trace_path=trace_path,
            topk=config.topk,
            max_completion_tokens=config.max_completion_tokens,
            max_decision_tokens=config.max_decision_tokens,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    summary_rows.sort(key=lambda x: (x["experiment"], x["method"], x["model_name"], x["max_tool_calls"]))

    _write_json(paths["root"] / "results.json", summary_rows)
    _write_csv(paths["root"] / "results.csv", summary_rows)
    _write_csv(paths["root"] / "per_sample_results.csv", detail_rows)

    react_rows = [r for r in summary_rows if r["experiment"] == "react_steps"]
    budget_rows = [r for r in summary_rows if r["experiment"] == "token_budget"]
    utility_rows = [r for r in summary_rows if r["experiment"] == "utility_ablation"]
    ablation_rows = [r for r in summary_rows if r["experiment"] == "policy_ablation"]
    model_rows = [r for r in summary_rows if r["experiment"] == "model_generalization"]
    main_rows = [r for r in summary_rows if r["experiment"] == "main"]

    _write_csv(paths["root"] / "table_react_steps.csv", react_rows)
    _write_csv(paths["root"] / "budget_experiment.csv", budget_rows)
    _write_csv(paths["root"] / "utility_ablation.csv", utility_rows)
    _write_csv(paths["root"] / "policy_ablation.csv", ablation_rows)
    _write_csv(paths["root"] / "model_generalization.csv", model_rows)

    _plot_react_steps(react_rows, paths["figures"])
    _plot_budget(budget_rows, paths["figures"])
    _plot_pareto(main_rows, paths["figures"])
    _plot_efficiency(main_rows, paths["figures"])
    _plot_tradeoff(main_rows, paths["figures"])
    _plot_policy_ablation(ablation_rows, paths["figures"])

    _latex_table(paths["tables"] / "table_main_results.tex", main_rows, "Main Results")
    _latex_table(paths["tables"] / "table_ablation.tex", ablation_rows, "Policy Ablation")
    _latex_table(paths["tables"] / "table_budget.tex", budget_rows, "Token Budget")
    _latex_table(paths["root"] / "table_main_results.tex", main_rows, "Main Results")
    _latex_table(paths["root"] / "table_ablation.tex", ablation_rows, "Policy Ablation")
    _latex_table(paths["root"] / "table_budget.tex", budget_rows, "Token Budget")
