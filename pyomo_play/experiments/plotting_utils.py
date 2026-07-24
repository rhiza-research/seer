"""Publication-ready plotting helpers for solver-scaling experiment results."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---- palette (validated categorical palette, light chart surface) ----
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
FONT_FAMILY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]

# human-readable labels for common experiment result columns
AXIS_LABELS = {
    "n_zones": "Zones",
    "n_years": "Years",
    "n_hours": "Hours per year",
    "n_variables": "Variables",
    "n_constraints": "Constraints",
    "solve_time": "Solve time (s)",
    "total_cost": "Total cost ($)",
}


def _to_dataframe(data):
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


def plot_solve_time_scaling(
    *,
    results,
    x="n_variables",
    y="solve_time",
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
    log_x=False,
    log_y=True,
    save_path=None,
):
    """Plot a scaling curve of `y` (default: solve time) against `x` (default: problem size).

    `results` is either a single DataFrame/list-of-dicts (one series — e.g. straight from
    experiments/single_zone.py's `results` list), or a dict of {series_label: DataFrame/
    list-of-dicts} to overlay multiple series (e.g. continuous vs discrete builds).
    """
    series = (
        {label: _to_dataframe(df) for label, df in results.items()}
        if isinstance(results, dict)
        else {None: _to_dataframe(results)}
    )

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    else:
        fig = ax.figure

    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    for (label, df), color in zip(series.items(), CATEGORICAL_PALETTE):
        df = df.sort_values(x)
        ax.plot(
            df[x], df[y],
            color=color, linewidth=2, solid_capstyle="round",
            marker="o", markersize=8, markerfacecolor=color,
            markeredgecolor=CHART_SURFACE, markeredgewidth=1,
            label=label,
        )

    if log_x:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if log_y:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax.set_xlabel(xlabel or AXIS_LABELS.get(x, x), color=SECONDARY_INK, fontsize=11, fontfamily=FONT_FAMILY)
    ax.set_ylabel(ylabel or AXIS_LABELS.get(y, y), color=SECONDARY_INK, fontsize=11, fontfamily=FONT_FAMILY)
    if title:
        ax.set_title(title, color=PRIMARY_INK, fontsize=13, fontweight="bold", fontfamily=FONT_FAMILY, pad=12)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color=GRIDLINE, linewidth=0.8)

    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(AXIS_LINE)
            spine.set_linewidth(0.8)

    ax.tick_params(colors=MUTED_INK, labelsize=9.5)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)

    if len(series) > 1:
        ax.legend(frameon=False, loc="best", fontsize=9.5, labelcolor=SECONDARY_INK)

    if created_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, facecolor=CHART_SURFACE, bbox_inches="tight")
        return fig, ax

    return ax
