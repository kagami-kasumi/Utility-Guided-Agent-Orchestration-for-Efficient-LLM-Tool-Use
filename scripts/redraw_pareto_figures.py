#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_policy_pilot.visualization import (
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

    if x_key == "avg_tokens":
        x_pad = max((x_max - x_min) * 0.08, x_max * 0.025)
    else:
        x_pad = max((x_max - x_min) * 0.09, x_max * 0.03)
    y_pad = max((y_max - y_min) * 0.12, 0.012)

    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    ax.grid(alpha=0.28, linestyle="--", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _initial_label_offset(
    row: dict[str, object], *, x_span: float, y_span: float
) -> tuple[float, float]:
    method = str(row["method"])
    tool_calls = int(row["max_tool_calls"])

    x_unit = max(x_span * 0.018, 0.012 if x_span < 10 else 12.0)
    y_unit = max(y_span * 0.018, 0.0035)

    x_sign_by_method = {
        "direct": -1.0,
        "threshold": 1.0,
        "workflow": 1.0,
        "react": -1.0,
        "policy": 1.0,
    }
    y_sign_by_calls = {1: -1.0, 3: 1.0, 5: 1.0}

    dx = x_unit * x_sign_by_method.get(method, 1.0)
    dy = y_unit * y_sign_by_calls.get(tool_calls, 1.0)

    if method in {"workflow", "react"} and tool_calls == 5:
        dx *= 1.4
    if method == "policy" and tool_calls == 1:
        dx *= -0.85
        dy *= 0.9
    if method == "direct":
        direct_offsets = {
            1: (0.45, -1.0),
            3: (1.1, 0.95),
            5: (-1.2, 1.1),
        }
        scale_x, scale_y = direct_offsets.get(tool_calls, (-1.0, 1.0))
        dx = x_unit * scale_x
        dy = y_unit * scale_y
    if method == "workflow":
        workflow_offsets = {
            1: (1.0, -0.8),
            3: (0.8, 1.1),
            5: (1.65, 0.95),
        }
        scale_x, scale_y = workflow_offsets.get(tool_calls, (1.0, 1.0))
        dx = x_unit * scale_x
        dy = y_unit * scale_y

    return dx, dy


def _display_gap_to_bbox(bbox, x: float, y: float) -> float:
    dx = max(bbox.x0 - x, 0.0, x - bbox.x1)
    dy = max(bbox.y0 - y, 0.0, y - bbox.y1)
    return math.hypot(dx, dy)


def _nudge_labels_off_markers(
    fig: plt.Figure,
    ax: plt.Axes,
    texts: list[plt.Text],
    anchor_points: list[tuple[float, float]],
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for _ in range(6):
        moved = False
        for text, (x, y) in zip(texts, anchor_points):
            bbox = text.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
            px, py = ax.transData.transform((x, y))
            gap = _display_gap_to_bbox(bbox, px, py)
            if gap >= 12.0:
                continue

            tx, ty = text.get_position()
            tx_disp, ty_disp = ax.transData.transform((tx, ty))
            vx = tx_disp - px
            vy = ty_disp - py
            if abs(vx) < 1.0 and abs(vy) < 1.0:
                vx, vy = 14.0, 10.0
            norm = math.hypot(vx, vy)
            step = max(14.0 - gap, 8.0)
            new_disp = (
                tx_disp + step * vx / norm,
                ty_disp + step * vy / norm,
            )
            new_data = ax.transData.inverted().transform(new_disp)
            text.set_position((float(new_data[0]), float(new_data[1])))
            moved = True

        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _draw_selective_connectors(
    fig: plt.Figure,
    ax: plt.Axes,
    texts: list[plt.Text],
    anchor_points: list[tuple[float, float]],
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for text, (x, y) in zip(texts, anchor_points):
        bbox = text.get_window_extent(renderer=renderer).expanded(1.02, 1.04)
        px, py = ax.transData.transform((x, y))
        gap = _display_gap_to_bbox(bbox, px, py)
        if gap < 20.0:
            continue

        nearest_disp = (
            min(max(px, bbox.x0), bbox.x1),
            min(max(py, bbox.y0), bbox.y1),
        )
        nearest_data = ax.transData.inverted().transform(nearest_disp)
        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (float(nearest_data[0]), float(nearest_data[1])),
                arrowstyle="-",
                lw=1.0,
                color="#6A6A6A",
                alpha=0.9,
                shrinkA=8,
                shrinkB=2,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
        )


def draw_pareto(rows: list[dict[str, object]], *, x_key: str, output_base: Path, title: str, xlabel: str) -> None:
    setup_plot_style()
    frontier_ids = pareto_frontier(rows, x_key, "avg_f1")
    fig, ax = plt.subplots(figsize=(10.2, 6.7))

    _style_axes(ax, x_key=x_key, rows=rows)
    x_values = [float(r[x_key]) for r in rows]
    y_values = [float(r["avg_f1"]) for r in rows]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)

    texts = []
    anchor_points: list[tuple[float, float]] = []

    for row in rows:
        is_frontier = row["run_id"] in frontier_ids
        edge = "#111111" if is_frontier else "white"
        x, y = float(row[x_key]), float(row["avg_f1"])

        ax.scatter(
            x, y,
            c=method_color(str(row["method"])),
            marker=method_marker(int(row["max_tool_calls"])),
            s=170,
            edgecolors=edge,
            linewidths=1.4,
            alpha=0.96,
            zorder=3,
        )

        label = f'{row["method"]}@{row["max_tool_calls"]}'
        dx, dy = _initial_label_offset(row, x_span=x_span, y_span=y_span)
        t = ax.text(
            x + dx,
            y + dy,
            label,
            fontsize=11,
            fontweight="normal",
            bbox={
                "boxstyle": "round,pad=0.12",
                "fc": "white",
                "ec": "none",
                "alpha": 0.88,
            },
            zorder=4,
        )
        texts.append(t)
        anchor_points.append((x, y))

    adjust_text(
        texts,
        x=[x for x, _ in anchor_points],
        y=[y for _, y in anchor_points],
        ax=ax,
        avoid_self=True,
        expand=(1.32, 1.48),
        force_text=(0.54, 0.76),
        force_static=(0.48, 0.58),
        force_pull=(0.03, 0.04),
        max_move=(48, 30),
        only_move={"text": "xy", "static": "xy", "pull": "xy", "explode": "xy"},
    )
    _nudge_labels_off_markers(fig, ax, texts, anchor_points)
    _draw_selective_connectors(fig, ax, texts, anchor_points)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1")
    ax.set_title(title)
    save_figure_all(fig, output_base)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "outputs" / "expanded"
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
