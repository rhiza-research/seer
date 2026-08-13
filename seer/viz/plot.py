"""Plotting helpers for the standardized seer output tables (see seer/models/README.md):
generator_capacity / storage_capacity (plot_capacity / plot_storage) and dispatch
(plot_dispatch / plot_dispatch_window).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# ---- tech colors, loosely following IEA style conventions ----
TECH_COLORS = {
    "coal": "#4d4d4d",
    "gas": "#e67e22",
    "ccgt": "#e67e22",
    "ice": "#a04000",
    "oil": "#a04000",
    "hfo": "#a04000",
    "lfo": "#ba6136",
    "nuclear": "#8e44ad",
    "hydro": "#2980b9",
    "wind": "#5dade2",
    "pv": "#f1c40f",
    "solar": "#f1c40f",
    "battery": "#27ae60",
    "storage": "#27ae60",
    "import": "#95a5a6",
    "importtransmission": "#95a5a6",
}
_FALLBACK_PALETTE = plt.get_cmap("tab20").colors

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # standard (non-leap) year


def tech_color(tech):
    """Consistent color for a tech; falls back to a deterministic pick from a qualitative
    palette for anything not in TECH_COLORS."""
    key = str(tech).lower()
    if key in TECH_COLORS:
        return TECH_COLORS[key]
    return _FALLBACK_PALETTE[hash(key) % len(_FALLBACK_PALETTE)]


def _get_ax(ax):
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=(10, 4))
    return ax


def _label_year_clusters(ax, cluster_centers, cluster_starts):
    """Bold year label under each year's cluster, with a separator before it (skipped for the
    first cluster). cluster_centers/cluster_starts: {year: x position}."""
    for i, (year, start) in enumerate(cluster_starts.items()):
        if i > 0:
            ax.axvline(start, color="grey", linewidth=0.5, linestyle="--")
        ax.text(
            cluster_centers[year], -0.12, str(year),
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", va="top", fontsize=9, fontweight="bold",
        )


def _label_time_axis(ax, year_lengths, time_group):
    """year_lengths: {year: number of x-positions in that year}, in plotted order. Adds
    month + year labels for hour/day granularity, or one label per year for "year"."""
    offset = 0
    cluster_centers, cluster_starts = {}, {}
    ticks, tick_labels = [], []

    units_per_month = None if time_group == "year" else (
        DAYS_IN_MONTH if time_group == "day" else [days * 24 for days in DAYS_IN_MONTH]
    )

    for year, length in year_lengths.items():
        cluster_starts[year] = offset
        cluster_centers[year] = offset + length / 2

        if units_per_month is not None:
            month_starts = np.concatenate([[0], np.cumsum(units_per_month)])
            for month_index, start in enumerate(month_starts[:-1]):
                if start < length:
                    ticks.append(offset + start)
                    tick_labels.append(MONTH_LABELS[month_index])

        offset += length

    ax.set_xlim(0, offset)
    if units_per_month is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])

    _label_year_clusters(ax, cluster_centers, cluster_starts)


def _signed_stackplot(ax, x, series, colors):
    """Stacked area chart that handles mixed-sign series (e.g. storage: positive when
    discharging, negative when charging) correctly.

    ax.stackplot itself just cumulatively sums whatever's passed, in order, with no regard for
    sign - so a negative series pulls the running baseline below zero, distorting every series
    stacked on top of it even though their own values never went negative. Here, at each x, every
    series instead contributes to one of two independent baselines depending on the sign of its
    own value there: a positive stack growing up from 0, and a negative stack growing down from
    0 - so e.g. generators (always >= 0) stay stacked above 0 regardless of what storage is doing.

    series: {name: array-like}, in stacking order. colors: {name: color}.
    """
    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))
    for name, values in series.items():
        values = np.asarray(values, dtype=float)
        pos_part = np.clip(values, 0, None)
        neg_part = np.clip(values, None, 0)
        color = colors[name]

        if pos_part.any():
            ax.fill_between(x, pos_bottom, pos_bottom + pos_part, color=color, label=name)
            pos_bottom += pos_part
        if neg_part.any():
            ax.fill_between(x, neg_bottom + neg_part, neg_bottom, color=color, label=None if pos_part.any() else name)
            neg_bottom += neg_part


def _plot_grouped_stacked_capacity(df, ylabel, space_group, ax):
    """Shared implementation for plot_capacity/plot_storage: a bar per year (or, if space_group
    is set, one bar per group within each year), stacked by tech."""
    ax = _get_ax(ax)

    index_cols = ["year"] + ([space_group] if space_group else [])
    pivot = df.pivot_table(index=index_cols, columns="tech", values="capacity", aggfunc="sum", fill_value=0.0)

    years = sorted(df["year"].unique())
    groups = sorted(df[space_group].unique()) if space_group else [None]
    techs = sorted(df["tech"].unique())
    n_groups = len(groups)

    cluster_centers, cluster_starts, xtick_positions, xtick_labels = {}, {}, [], []
    for yi, year in enumerate(years):
        base_x = yi * (n_groups + 1)  # one slot of padding between year clusters
        cluster_starts[year] = base_x - 0.5
        cluster_centers[year] = base_x + (n_groups - 1) / 2

        for gi, group in enumerate(groups):
            bar_x = base_x + gi
            key = (year, group) if space_group else year
            row = pivot.loc[key] if key in pivot.index else pd.Series(0.0, index=techs)

            bottom = 0.0
            for tech in techs:
                value = row.get(tech, 0.0)
                label = tech if (yi == 0 and gi == 0) else None
                ax.bar(bar_x, value, width=0.9, bottom=bottom, color=tech_color(tech), label=label)
                bottom += value

            if space_group:
                xtick_positions.append(bar_x)
                xtick_labels.append(str(group))

    if space_group:
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    _label_year_clusters(ax, cluster_centers, cluster_starts)

    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=7)
    return ax


def plot_capacity(df, space_group="zone", ax=None):
    """Generator capacity (generator_capacity schema), aggregated by tech, as a stacked bar per
    year. space_group="zone" additionally groups bars by zone within each year; None aggregates
    across zones into a single stacked bar per year."""
    ax = _plot_grouped_stacked_capacity(df, "Capacity (MW)", space_group, ax)
    ax.set_title("Generator capacity")
    return ax


def plot_storage(df, space_group="zone", ax=None):
    """Storage capacity (storage_capacity schema, MWh) - same layout as plot_capacity."""
    ax = _plot_grouped_stacked_capacity(df, "Storage capacity (MWh)", space_group, ax)
    ax.set_title("Storage capacity")
    return ax


def plot_dispatch(df, gen_group="tech", time_group="hour", ax=None):
    """Dispatch (dispatch schema) as a filled stacked area chart over time.

    gen_group="tech" aggregates dispatch by tech; None shows each individual unit (no legend, to
    avoid clutter). time_group sums dispatch within each "hour" (no aggregation), "day", or
    "year" before plotting.
    """
    ax = _get_ax(ax)
    df = df.copy()
    if time_group == "day":
        df["hour"] = (df["hour"] - 1) // 24 + 1
    elif time_group == "year":
        df["hour"] = 1

    series_col = "tech" if gen_group == "tech" else "name"
    name_tech = df[["name", "tech"]].drop_duplicates().set_index("name")["tech"]

    year_lengths = df.groupby("year")["hour"].nunique().to_dict()
    full_index = pd.MultiIndex.from_tuples(
        [(year, hour) for year, length in year_lengths.items() for hour in range(1, length + 1)],
        names=["year", "hour"],
    )
    pivot = (
        df.groupby(["year", "hour", series_col])["dispatch"].sum()
        .unstack(series_col).reindex(full_index, fill_value=0.0)
    )

    series_names = list(pivot.columns)
    colors = {name: tech_color(name if gen_group == "tech" else name_tech[name]) for name in series_names}

    x = list(range(len(pivot)))
    _signed_stackplot(ax, x, {name: pivot[name].values for name in series_names}, colors)
    _label_time_axis(ax, year_lengths, time_group)

    ax.set_ylabel("Dispatch (MW)")
    if gen_group == "tech":
        ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Dispatch")
    return ax


def _random_window(hours_in_year, window, rng):
    """Pick a random (start_hour, end_hour) window (1-indexed, inclusive) within a year of
    hours_in_year hours (which may be 8760 or 8784, see seer/models/README.md's hour convention)."""
    if window == "year":
        return 1, hours_in_year

    if window == "month":
        month_starts = np.concatenate([[0], np.cumsum([days * 24 for days in DAYS_IN_MONTH])])
        month_index = rng.integers(0, 12)
        return int(month_starts[month_index]) + 1, int(min(month_starts[month_index + 1], hours_in_year))

    block = {"week": 24 * 7, "day": 24}.get(window)
    if block is None:
        raise ValueError(f"Unknown window: {window!r} (expected 'day', 'week', 'month', or 'year')")

    block_index = rng.integers(0, hours_in_year // block)
    start = block_index * block + 1
    return start, start + block - 1


def plot_dispatch_window(df, window="day", group="tech", ax=None, rng=None):
    """Dispatch (dispatch schema) for a single randomly chosen day/week/month/year, as a filled
    area chart - useful for eyeballing dispatch shape without averaging it away.

    window: "day" (24h), "week" (168h), "month" (calendar-length), or "year" (the whole year),
    chosen at random within a randomly chosen model year.
    group="tech" stacks dispatch by tech (summed across zones/units of that tech); None
    aggregates across the whole model into a single total dispatch area.
    """
    rng = rng or np.random.default_rng()
    ax = _get_ax(ax)

    year = rng.choice(sorted(df["year"].unique()))
    year_df = df[df["year"] == year]
    start_hour, end_hour = _random_window(int(year_df["hour"].max()), window, rng)
    window_df = year_df[year_df["hour"].between(start_hour, end_hour)]

    if group == "tech":
        pivot = window_df.groupby(["hour", "tech"])["dispatch"].sum().unstack("tech").sort_index().fillna(0.0)
        series_names = list(pivot.columns)
        colors = {name: tech_color(name) for name in series_names}
        _signed_stackplot(ax, pivot.index, {name: pivot[name].values for name in series_names}, colors)
        ax.legend(loc="upper left", fontsize=7)
    else:
        totals = window_df.groupby("hour")["dispatch"].sum().sort_index()
        ax.fill_between(totals.index, totals.values, color=tech_color("total"))

    ax.set_xlabel("Hour of year")
    ax.set_ylabel("Dispatch (MW)")
    group_label = "by tech" if group == "tech" else "aggregate"
    ax.set_title(f"Dispatch sample: {window} in {year} ({group_label})")
    return ax
