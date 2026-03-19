from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

METHOD_COLORS = {
    "direct": "blue",
    "threshold": "green",
    "workflow": "gray",
    "workflow-search-twice": "gray",
    "workflow-search-verify": "gray",
    "react": "orange",
    "policy": "red",
}

MARKERS_BY_BUDGET = {
    1: "o",
    3: "^",
    5: "s",
}


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "black")


def method_marker(max_tool_calls: int | None) -> str:
    if max_tool_calls is None:
        return "D"
    return MARKERS_BY_BUDGET.get(max_tool_calls, "D")


def save_figure_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")


def pareto_frontier(rows: list[dict[str, Any]], cost_key: str, quality_key: str) -> set[str]:
    front: set[str] = set()
    for p in rows:
        dominated = False
        for q in rows:
            if p["run_id"] == q["run_id"]:
                continue
            if (
                q[cost_key] <= p[cost_key]
                and q[quality_key] >= p[quality_key]
                and (q[cost_key] < p[cost_key] or q[quality_key] > p[quality_key])
            ):
                dominated = True
                break
        if not dominated:
            front.add(p["run_id"])
    return front


def annotate_scatter(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    label_fn,
) -> None:
    if not rows:
        return

    ordered = sorted(rows, key=lambda row: (float(row[x_key]), float(row[y_key])))
    x_values = [float(row[x_key]) for row in ordered]
    y_values = [float(row[y_key]) for row in ordered]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    dx = max(0.02 * x_span, 0.015 * max(x_values), 0.02)
    dy = max(0.035 * y_span, 0.004)

    offsets = [
        (dx, dy),
        (dx, -dy),
        (-1.35 * dx, dy),
        (-1.35 * dx, -dy),
        (0.55 * dx, 1.65 * dy),
        (0.55 * dx, -1.65 * dy),
    ]

    for idx, row in enumerate(ordered):
        x = float(row[x_key])
        y = float(row[y_key])
        ox, oy = offsets[idx % len(offsets)]
        ha = "left" if ox >= 0 else "right"
        ax.annotate(
            label_fn(row),
            xy=(x, y),
            xytext=(x + ox, y + oy),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": "#B8BEC3",
                "lw": 0.8,
                "alpha": 0.95,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#8A949A",
                "lw": 0.8,
                "shrinkA": 3,
                "shrinkB": 4,
            },
            zorder=5,
        )
