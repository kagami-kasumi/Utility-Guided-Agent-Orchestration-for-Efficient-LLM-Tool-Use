from __future__ import annotations

import csv
import json
import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .data import HotpotExample, build_corpus_from_contexts, load_hotpot_dev, load_passages_jsonl, sample_examples
from .experiments import RunCase, _ensure_dirs, _run_case, _write_csv, _write_json
from .llm import OpenAIChatModel
from .search import BM25Searcher
from .visualization import save_figure_all, setup_plot_style


@dataclass(frozen=True)
class SupplementConfig:
    hotpot_dev_path: str
    output_dir: str
    base_results_dir: str
    primary_model: str
    model_base_url: str | None
    model_api_key: str | None
    corpus_jsonl_path: str | None = None
    sample_size: int = 200
    seed: int = 42
    topk: int = 3
    max_tool_calls: int = 3
    temperature: float = 0.0
    max_completion_tokens: int = 64
    max_decision_tokens: int = 96
    token_cost_reference: float = 1200.0
    latency_cost_reference: float = 1.0
    analysis_only: bool = False


def _load_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in {"", None}:
                    parsed[key] = None if value == "" else value
                    continue
                if key in {
                    "avg_em",
                    "avg_f1",
                    "avg_tokens",
                    "avg_wall_time",
                    "avg_tool_calls",
                    "tool_calls",
                    "avg_steps",
                    "redundant_tool_calls",
                    "efficiency",
                    "f1",
                    "token_usage",
                    "wall_time",
                    "lambda_cost",
                    "lambda_uncertainty",
                    "lambda_redundancy",
                    "token_cost_reference",
                    "latency_cost_reference",
                }:
                    parsed[key] = float(value)
                elif key in {"max_tool_calls", "token_budget", "num_samples"}:
                    parsed[key] = int(value) if value not in {"", None} else None
                elif key == "stop_step_distribution":
                    parsed[key] = ast.literal_eval(value)
                elif value in {"True", "False"}:
                    parsed[key] = value == "True"
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _sample_from_ids(examples: list[HotpotExample], sampled_ids_path: Path, sample_size: int, seed: int) -> list[HotpotExample]:
    if not sampled_ids_path.exists():
        return sample_examples(examples, sample_size=sample_size, seed=seed)
    payload = json.loads(sampled_ids_path.read_text(encoding="utf-8"))
    order = [str(x) for x in payload.get("ids", [])]
    by_id = {ex.qid: ex for ex in examples}
    sample = [by_id[qid] for qid in order if qid in by_id]
    if len(sample) != len(order):
        missing = len(order) - len(sample)
        raise ValueError(f"sampled_ids.json contains {missing} ids not found in dataset")
    return sample


def _row_lookup(rows: list[dict[str, Any]], *, run_id: str | None = None, method: str | None = None, experiment: str | None = None, max_tool_calls: int | None = None, cost_mode: str | None = None, redundancy_mode: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        if run_id is not None and row.get("run_id") != run_id:
            continue
        if method is not None and row.get("method") != method:
            continue
        if experiment is not None and row.get("experiment") != experiment:
            continue
        if max_tool_calls is not None and int(row.get("max_tool_calls")) != max_tool_calls:
            continue
        if cost_mode is not None and row.get("cost_mode", "step") != cost_mode:
            continue
        if redundancy_mode is not None and row.get("redundancy_mode", "exact") != redundancy_mode:
            continue
        return row
    return None


def _merge_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for rows in groups:
        for row in rows:
            run_id = str(row.get("run_id", ""))
            if not run_id:
                continue
            if run_id not in merged:
                order.append(run_id)
            merged[run_id] = row
    return [merged[run_id] for run_id in order]


def _method_label(row: dict[str, Any]) -> str:
    method = str(row.get("method", ""))
    run_id = str(row.get("run_id", ""))
    ablation_labels = {
        "ablation_full_policy": "full policy",
        "ablation_no_expected_gain": "-expected_gain",
        "ablation_no_uncertainty": "-uncertainty",
        "ablation_no_redundancy": "-redundancy",
        "ablation_no_stop_policy": "-stop",
    }
    for prefix, label in ablation_labels.items():
        if run_id.startswith(prefix):
            return label
    if method == "policy":
        cost_mode = row.get("cost_mode", "step")
        redundancy_mode = row.get("redundancy_mode", "exact")
        if cost_mode == "token":
            return "policy (token_cost)"
        if cost_mode == "latency":
            return "policy (latency_cost)"
        if redundancy_mode == "semantic":
            return "policy (semantic redundancy)"
        return "policy (step_cost)"
    if method == "workflow":
        return "workflow (minimal)"
    if method == "workflow-search-twice":
        return "workflow-search-twice"
    if method == "workflow-search-verify":
        return "workflow-search-verify"
    return method


def _table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        table.append(
            {
                "Method": _method_label(row),
                "F1": round(float(row["avg_f1"]), 4),
                "Tokens": round(float(row["avg_tokens"]), 1),
                "Wall Time": round(float(row["avg_wall_time"]), 3),
                "Efficiency": round(float(row["efficiency"]), 6),
                "Tool Calls": round(float(row["avg_tool_calls"]), 2),
                "Redundant Tool Calls": round(float(row["redundant_tool_calls"]), 2),
            }
        )
    return table


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_missing_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _latex_table(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{title}}}",
        "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(str(row.get(col, "")) for col in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bucket_label(index: int, bucket_count: int) -> str:
    left = index / bucket_count
    right = (index + 1) / bucket_count
    return f"[{left:.1f}, {right:.1f}{']' if index == bucket_count - 1 else ')'}"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _heuristic_analysis(trace_path: Path, bucket_count: int = 5) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not trace_path.exists():
        return [], {"expected_gain_f1_corr": 0.0, "uncertainty_f1_corr": 0.0}

    per_signal: dict[str, list[dict[str, float]]] = {"expected_gain": [], "uncertainty": []}
    gain_values: list[float] = []
    gain_f1: list[float] = []
    unc_values: list[float] = []
    unc_f1: list[float] = []

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            final_f1 = float(row.get("f1", 0.0))
            for event in row.get("trajectory", []):
                if event.get("type") != "model":
                    continue
                decision = event.get("decision") or {}
                if "expected_gain" not in decision or "uncertainty" not in decision:
                    continue
                continue_flag = 1.0 if decision.get("should_search") else 0.0
                gain = float(decision.get("expected_gain", 0.0))
                uncertainty = float(decision.get("uncertainty", 0.0))
                per_signal["expected_gain"].append({"value": gain, "continue": continue_flag, "final_f1": final_f1})
                per_signal["uncertainty"].append({"value": uncertainty, "continue": continue_flag, "final_f1": final_f1})
                gain_values.append(gain)
                gain_f1.append(final_f1)
                unc_values.append(uncertainty)
                unc_f1.append(final_f1)

    rows: list[dict[str, Any]] = []
    for signal_name, records in per_signal.items():
        for bucket_idx in range(bucket_count):
            left = bucket_idx / bucket_count
            right = (bucket_idx + 1) / bucket_count
            if bucket_idx == bucket_count - 1:
                bucket_records = [r for r in records if left <= r["value"] <= right]
            else:
                bucket_records = [r for r in records if left <= r["value"] < right]
            if not bucket_records:
                continue
            rows.append(
                {
                    "signal": signal_name,
                    "bucket": _bucket_label(bucket_idx, bucket_count),
                    "count": len(bucket_records),
                    "continue_rate": round(sum(r["continue"] for r in bucket_records) / len(bucket_records), 4),
                    "avg_final_f1": round(sum(r["final_f1"] for r in bucket_records) / len(bucket_records), 4),
                }
            )

    corr = {
        "expected_gain_f1_corr": round(_pearson(gain_values, gain_f1), 4),
        "uncertainty_f1_corr": round(_pearson(unc_values, unc_f1), 4),
    }
    return rows, corr


def _plot_heuristic_signals(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, signal in zip(axes, ("expected_gain", "uncertainty")):
        srows = [r for r in rows if r["signal"] == signal]
        xs = list(range(len(srows)))
        ax.plot(xs, [r["continue_rate"] for r in srows], marker="o", color="red" if signal == "expected_gain" else "green")
        ax.set_xticks(xs, [r["bucket"] for r in srows], rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Continue Rate")
        ax.set_xlabel(signal)
        ax.set_title(f"{signal} vs continue_rate")
        ax.grid(alpha=0.3)
    save_figure_all(fig, out_dir / "figure6_heuristic_signals")
    save_figure_all(fig, out_dir / "heuristic_signals")
    plt.close(fig)


def _run_new_cases(config: SupplementConfig, sample: list[HotpotExample], searcher: BM25Searcher, paths: dict[str, Path]) -> list[dict[str, Any]]:
    if config.analysis_only:
        return []

    model = OpenAIChatModel(
        model=config.primary_model,
        base_url=config.model_base_url,
        api_key=config.model_api_key,
        temperature=config.temperature,
        seed=config.seed,
    )

    cases = [
        RunCase(
            experiment="supplement_cost",
            run_id=f"supp_policy_token_cost_k{config.max_tool_calls}_{config.primary_model}",
            method="policy",
            model_name=config.primary_model,
            max_tool_calls=config.max_tool_calls,
            cost_mode="token",
            token_cost_reference=config.token_cost_reference,
        ),
        RunCase(
            experiment="supplement_cost",
            run_id=f"supp_policy_latency_cost_k{config.max_tool_calls}_{config.primary_model}",
            method="policy",
            model_name=config.primary_model,
            max_tool_calls=config.max_tool_calls,
            cost_mode="latency",
            latency_cost_reference=config.latency_cost_reference,
        ),
        RunCase(
            experiment="supplement_redundancy",
            run_id=f"supp_policy_semantic_k{config.max_tool_calls}_{config.primary_model}",
            method="policy",
            model_name=config.primary_model,
            max_tool_calls=config.max_tool_calls,
            redundancy_mode="semantic",
        ),
        RunCase(
            experiment="supplement_workflow",
            run_id=f"supp_workflow_search_twice_{config.primary_model}",
            method="workflow-search-twice",
            model_name=config.primary_model,
            max_tool_calls=2,
        ),
        RunCase(
            experiment="supplement_workflow",
            run_id=f"supp_workflow_search_verify_{config.primary_model}",
            method="workflow-search-verify",
            model_name=config.primary_model,
            max_tool_calls=2,
        ),
    ]

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for case in cases:
        trace_path = paths["traces"] / f"{case.run_id}.jsonl"
        summary, details = _run_case(
            case=case,
            sample=sample,
            searcher=searcher,
            model=model,
            trace_path=trace_path,
            topk=config.topk,
            max_completion_tokens=config.max_completion_tokens,
            max_decision_tokens=config.max_decision_tokens,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    _write_csv(paths["root"] / "supplement_results.csv", summary_rows)
    _write_csv(paths["root"] / "supplement_per_sample.csv", detail_rows)
    return summary_rows


def _compose_markdown(
    output_path: Path,
    figures_dir: Path,
    main_table: list[dict[str, Any]],
    cost_table: list[dict[str, Any]],
    workflow_table: list[dict[str, Any]],
    redundancy_table: list[dict[str, Any]],
    ablation_table: list[dict[str, Any]],
    heuristic_table: list[dict[str, Any]],
    heuristic_corr: dict[str, float],
) -> None:
    pending_cost = len(cost_table) < 6
    pending_workflow = len(workflow_table) < 3
    pending_redundancy = len(redundancy_table) < 2
    recommended_figures = [
        f"- 主文保留 `figure1_pareto_tokens` 或 `figure2_pareto_latency` 二选一，另一张放附录：`{figures_dir}`",
        f"- 若正文需要支撑 heuristic 解释，新增 `figure6_heuristic_signals`：`{figures_dir / 'figure6_heuristic_signals.png'}`",
        "- `workflow fairness` 和 `cost definition` 更适合用表，不建议再单独堆图。",
    ]

    status_lines = ["- 已完成：术语统一、README 说明、heuristic signal 分析、主结果表/ablation 表导出。"]
    pending_items: list[str] = []
    if pending_cost:
        pending_items.extend(["`policy(token_cost)`", "`policy(latency_cost)`"])
    if pending_redundancy:
        pending_items.append("`policy(semantic redundancy)`")
    if pending_workflow:
        pending_items.extend(["`workflow-search-twice`", "`workflow-search-verify`"])
    if pending_items:
        status_lines.append("- 待补齐实跑：" + "、".join(pending_items) + "。")
    else:
        status_lines.append("- 新增 5 个补充 case 已补齐。")

    lines = [
        "# 实验结果-实验增补表316",
        "",
        "## 当前状态",
        *status_lines,
        "",
        "## 术语统一",
        "- `step_cost` 统一表示 policy 内部的归一化步代价。",
        "- `token_usage` 和 `wall_time` 保留为评测指标，不作为当前默认 policy 的直接决策输入。",
        "- `expected_gain` 与 `uncertainty` 统一表述为 LLM self-estimated heuristic signals，并裁剪到 `[0,1]`，不是 calibrated probabilities。",
        "",
        "## 建议保留的图",
        *recommended_figures,
        "",
        "## 主结果表",
        _markdown_table(main_table, ["Method", "F1", "Tokens", "Wall Time", "Efficiency"]),
        "",
        "## 成本定义对照表",
        _markdown_table(cost_table, ["Method", "F1", "Tokens", "Wall Time", "Efficiency"]),
        "- 若当前只有 `policy(step_cost)` 和 baseline，说明真实成本版本尚未实跑。" if pending_cost else "- `step_cost / token_cost / latency_cost` 三种版本已齐。",
        "",
        "## Workflow Fairness 表",
        _markdown_table(workflow_table, ["Method", "F1", "Tokens", "Wall Time", "Efficiency", "Tool Calls"]),
        "- 若当前只有 `workflow (minimal)`，说明两个 fixed workflow baseline 尚未实跑。" if pending_workflow else "- 三个 fixed workflow baseline 已齐。",
        "",
        "## Redundancy 对照表",
        _markdown_table(redundancy_table, ["Method", "F1", "Tokens", "Wall Time", "Redundant Tool Calls"]),
        "- 若当前只有 `policy(step_cost)`，说明 `semantic redundancy` 版本尚未实跑。" if pending_redundancy else "- `exact / semantic` redundancy 对照已齐。",
        "",
        "## Ablation 表",
        _markdown_table(ablation_table, ["Method", "F1", "Tokens", "Wall Time", "Efficiency"]),
        "",
        "## Heuristic Signal Analysis 表",
        _markdown_table(heuristic_table, ["signal", "bucket", "count", "continue_rate", "avg_final_f1"]),
        "",
        "## Heuristic Signal 粗相关",
        f"- expected_gain vs final F1 Pearson: `{heuristic_corr['expected_gain_f1_corr']}`",
        f"- uncertainty vs final F1 Pearson: `{heuristic_corr['uncertainty_f1_corr']}`",
        "",
        "## 结论摘要",
        "- 成本定义对照用于回答 `step_cost proxy` 是否与真实 token/latency 成本趋势一致。",
        "- semantic redundancy 对照用于证明 redundancy 项不是只靠 exact match 撑着。",
        "- workflow-search-twice 与 workflow-search-verify 用于提升 baseline 公平性，避免 reviewer 认为 workflow baseline 过弱。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_supplement(config: SupplementConfig) -> None:
    paths = _ensure_dirs(Path(config.output_dir))
    base_dir = Path(config.base_results_dir)
    _write_json(paths["root"] / "supplement_config.json", asdict(config))

    examples = load_hotpot_dev(config.hotpot_dev_path)
    sample = _sample_from_ids(
        examples=examples,
        sampled_ids_path=base_dir / "sampled_ids.json",
        sample_size=config.sample_size,
        seed=config.seed,
    )
    _write_json(paths["root"] / "sampled_ids.json", {"sample_size": len(sample), "ids": [ex.qid for ex in sample]})

    if config.corpus_jsonl_path:
        corpus = load_passages_jsonl(config.corpus_jsonl_path)
    else:
        corpus = build_corpus_from_contexts(examples)
    searcher = BM25Searcher(corpus)

    base_rows = _load_results_csv(base_dir / "results.csv")
    existing_supp_rows = _load_results_csv(paths["root"] / "supplement_results.csv")
    new_rows = _run_new_cases(config=config, sample=sample, searcher=searcher, paths=paths)
    supplement_rows = _merge_rows(existing_supp_rows, new_rows)
    all_rows = base_rows + supplement_rows

    main_rows_raw = [
        _row_lookup(all_rows, experiment="main", method="direct", max_tool_calls=config.max_tool_calls),
        _row_lookup(all_rows, experiment="main", method="workflow", max_tool_calls=config.max_tool_calls),
        _row_lookup(all_rows, experiment="supplement_workflow", method="workflow-search-twice"),
        _row_lookup(all_rows, experiment="supplement_workflow", method="workflow-search-verify"),
        _row_lookup(all_rows, experiment="main", method="threshold", max_tool_calls=config.max_tool_calls),
        _row_lookup(all_rows, experiment="main", method="react", max_tool_calls=config.max_tool_calls),
        _row_lookup(all_rows, experiment="main", method="policy", max_tool_calls=config.max_tool_calls, cost_mode="step", redundancy_mode="exact"),
    ]
    main_table = _table_rows([row for row in main_rows_raw if row])

    cost_rows_raw = [
        _row_lookup(all_rows, experiment="main", method="workflow", max_tool_calls=config.max_tool_calls),
        _row_lookup(all_rows, experiment="main", method="react", max_tool_calls=config.max_tool_calls),
        _row_lookup(all_rows, experiment="main", method="threshold", max_tool_calls=config.max_tool_calls),
        _row_lookup(all_rows, experiment="main", method="policy", max_tool_calls=config.max_tool_calls, cost_mode="step", redundancy_mode="exact"),
        _row_lookup(all_rows, experiment="supplement_cost", method="policy", cost_mode="token"),
        _row_lookup(all_rows, experiment="supplement_cost", method="policy", cost_mode="latency"),
    ]
    cost_table = _table_rows([row for row in cost_rows_raw if row])

    workflow_rows_raw = [
        _row_lookup(all_rows, experiment="main", method="workflow", max_tool_calls=config.max_tool_calls),
        _row_lookup(all_rows, experiment="supplement_workflow", method="workflow-search-twice"),
        _row_lookup(all_rows, experiment="supplement_workflow", method="workflow-search-verify"),
    ]
    workflow_table = _table_rows([row for row in workflow_rows_raw if row])

    redundancy_rows_raw = [
        _row_lookup(all_rows, experiment="main", method="policy", max_tool_calls=config.max_tool_calls, cost_mode="step", redundancy_mode="exact"),
        _row_lookup(all_rows, experiment="supplement_redundancy", method="policy", redundancy_mode="semantic"),
    ]
    redundancy_table = _table_rows([row for row in redundancy_rows_raw if row])

    ablation_rows_raw = [
        _row_lookup(all_rows, run_id=f"ablation_full_policy_{config.primary_model}"),
        _row_lookup(all_rows, run_id=f"ablation_no_expected_gain_{config.primary_model}"),
        _row_lookup(all_rows, run_id=f"ablation_no_uncertainty_{config.primary_model}"),
        _row_lookup(all_rows, run_id=f"ablation_no_redundancy_{config.primary_model}"),
        _row_lookup(all_rows, run_id=f"ablation_no_stop_policy_{config.primary_model}"),
    ]
    ablation_table = _table_rows([row for row in ablation_rows_raw if row])

    heuristic_rows, heuristic_corr = _heuristic_analysis(
        Path(config.base_results_dir) / "traces" / f"main_policy_k{config.max_tool_calls}_{config.primary_model}.jsonl"
    )
    _plot_heuristic_signals(heuristic_rows, paths["figures"])

    _write_csv(paths["root"] / "table_main_results.csv", main_table)
    _write_csv(paths["root"] / "cost_definition_comparison.csv", cost_table)
    _write_csv(paths["root"] / "workflow_fairness.csv", workflow_table)
    _write_csv(paths["root"] / "redundancy_comparison.csv", redundancy_table)
    _write_csv(paths["root"] / "table_ablation.csv", ablation_table)
    _write_csv(paths["root"] / "heuristic_signal_analysis.csv", heuristic_rows)
    _write_json(paths["root"] / "heuristic_signal_correlation.json", heuristic_corr)

    tables_dir = paths["tables"]
    _latex_table(tables_dir / "table_main_results.tex", main_table, ["Method", "F1", "Tokens", "Wall Time", "Efficiency"], "Main Results")
    _latex_table(
        tables_dir / "table_cost_definition.tex",
        cost_table,
        ["Method", "F1", "Tokens", "Wall Time", "Efficiency"],
        "Cost Definition Comparison",
    )
    _latex_table(
        tables_dir / "table_workflow_fairness.tex",
        workflow_table,
        ["Method", "F1", "Tokens", "Wall Time", "Efficiency", "Tool Calls"],
        "Workflow Fairness",
    )
    _latex_table(tables_dir / "table_ablation.tex", ablation_table, ["Method", "F1", "Tokens", "Wall Time", "Efficiency"], "Policy Ablation")
    _latex_table(
        tables_dir / "table_heuristic_signal.tex",
        heuristic_rows,
        ["signal", "bucket", "count", "continue_rate", "avg_final_f1"],
        "Heuristic Signal Analysis",
    )

    _compose_markdown(
        output_path=paths["root"] / "实验结果-实验增补表316.md",
        figures_dir=paths["figures"],
        main_table=main_table,
        cost_table=cost_table,
        workflow_table=workflow_table,
        redundancy_table=redundancy_table,
        ablation_table=ablation_table,
        heuristic_table=heuristic_rows,
        heuristic_corr=heuristic_corr,
    )
