#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_policy_pilot.visualization import (
    annotate_scatter,
    method_color,
    method_marker,
    pareto_frontier,
    save_figure_all,
    setup_plot_style,
)


def load_main_rows(results_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("experiment") != "main":
                continue
            rows.append(
                {
                    "run_id": row["run_id"],
                    "method": row["method"],
                    "max_tool_calls": int(row["max_tool_calls"]),
                    "avg_tokens": float(row["avg_tokens"]),
                    "avg_wall_time": float(row["avg_wall_time"]),
                    "avg_f1": float(row["avg_f1"]),
                }
            )
    return rows


def _style_axes(ax: plt.Axes, *, x_key: str, rows: list[dict[str, object]]) -> None:
    x_values = [float(r[x_key]) for r in rows]
    y_values = [float(r["avg_f1"]) for r in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.18, x_max * 0.08)
    y_pad = max((y_max - y_min) * 0.18, 0.015)
    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    ax.grid(alpha=0.28, linestyle="--", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_pareto(rows: list[dict[str, object]], *, x_key: str, output_base: Path, title: str, xlabel: str) -> None:
    setup_plot_style()
    frontier = pareto_frontier(rows, x_key, "avg_f1")

    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    for row in rows:
        edge = "#111111" if row["run_id"] in frontier else "white"
        ax.scatter(
            float(row[x_key]),
            float(row["avg_f1"]),
            c=method_color(str(row["method"])),
            marker=method_marker(int(row["max_tool_calls"])),
            s=170,
            edgecolors=edge,
            linewidths=1.4,
            alpha=0.96,
            zorder=3,
        )

    annotate_scatter(
        ax,
        rows,
        x_key,
        "avg_f1",
        label_fn=lambda row: f'{row["method"]}@{row["max_tool_calls"]}',
    )
    _style_axes(ax, x_key=x_key, rows=rows)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1")
    ax.set_title(title)
    save_figure_all(fig, output_base)
    plt.close(fig)


def main() -> None:
    root = Path("/home/inspur-02/lby/outputs/expanded")
    figures_dir = root / "figures"
    rows = load_main_rows(root / "results.csv")
    draw_pareto(
        rows,
        x_key="avg_tokens",
        output_base=figures_dir / "figure1_pareto_tokens",
        title="Quality-Cost Pareto Frontier (Tokens)",
        xlabel="Average Tokens",
    )
    draw_pareto(
        rows,
        x_key="avg_wall_time",
        output_base=figures_dir / "figure2_pareto_latency",
        title="Quality-Cost Pareto Frontier (Wall Time)",
        xlabel="Average Wall Time (s)",
    )


if __name__ == "__main__":
    main()
