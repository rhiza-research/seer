"""Plots of a seer-schema input dataset (see seer/schema/README.md), for per-country records.

Each function takes the dataset's tables as DataFrames and returns a matplotlib Figure.
Time-series x axes are the dataset's own `times.csv` timepoints in order (e.g. a few
representative days per month), labeled by month and day.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plot import MONTH_LABELS, TECH_STACK_ORDER, tech_color

FIGSIZE = (6.8, 2.7)  # sized for a portrait A4 page's text width


def set_report_style():
    sns.set_theme(style="white", context="paper", font_scale=0.95)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def zone_colors(zones):
    palette = sns.color_palette("deep", max(len(zones), 1))
    return {zone: palette[i] for i, zone in enumerate(zones)}


def _time_axis(ax, times):
    """Ticks at the first hour of each modeled day, with light day separators. `times` is
    times.csv; days are labeled by month and, where times.csv has weights, the weight of that
    day ("Jan x1", "Jan x30" -- e.g. a peak day and a typical day), else by day of month."""
    weights = times["weight"].tolist() if "weight" in times.columns else None
    times = times["time"].tolist()
    day_starts = [i for i, t in enumerate(times) if i == 0 or t[:5] != times[i - 1][:5]]
    month = [MONTH_LABELS[int(times[i][:2]) - 1] for i in day_starts]
    if weights is not None:
        labels = [f"{m} \u00d7{weights[i]:g}" for m, i in zip(month, day_starts)]
    else:
        labels = [f"{m} {int(times[i][3:5])}" for m, i in zip(month, day_starts)]
    ax.set_xticks(day_starts)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    for i in day_starts[1:]:
        ax.axvline(i - 0.5, color="0.9", linewidth=0.6, zorder=0)
    ax.set_xlim(-0.5, len(times) - 0.5)


def plot_demand_trajectory(demand_forecast):
    """Annual peak demand by zone."""
    peak = demand_forecast[demand_forecast["type"] == "peak"]
    zones = sorted(peak["zone"].unique())
    colors = zone_colors(zones)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for zone in zones:
        z = peak[peak["zone"] == zone].sort_values("year")
        ax.plot(z["year"], z["value"], marker="o", markersize=3, linewidth=1.5, color=colors[zone], label=zone)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Demand trajectory (annual peak)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Peak demand (MW)")
    ax.set_ylim(bottom=0)
    if len(zones) > 1:
        ax.legend(title="Zone")
    fig.tight_layout()
    return fig


def plot_demand_profiles(demand_profile, times):
    """Normalized hourly demand profile for every modeled timepoint, by zone. Zones with
    identical profiles are drawn as one line labeled with all of them, so none look missing."""
    time_order = times["time"].tolist()
    position = {t: i for i, t in enumerate(time_order)}
    zones = sorted(demand_profile["zone"].unique())
    colors = zone_colors(zones)
    wide = demand_profile.pivot(index="time", columns="zone", values="value").reindex(time_order)
    groups = []  # [zones sharing one curve], in zone order
    for zone in zones:
        for group in groups:
            if np.allclose(wide[zone].values, wide[group[0]].values, equal_nan=True):
                group.append(zone)
                break
        else:
            groups.append([zone])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(time_order))
    for group in groups:
        label = group[0] if len(group) == 1 else f"{', '.join(group)} (identical)"
        ax.plot(x, wide[group[0]].values, linewidth=1, color=colors[group[0]], label=label)
    _time_axis(ax, times)
    ax.set_title("Hourly demand profiles (modeled days)")
    ax.set_ylabel("Demand (fraction of annual peak)")
    ax.set_ylim(bottom=0)
    if len(zones) > 1:
        ax.legend(title="Zone", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    return fig


def starting_capacity(generator_constants, storage_constants, first_year):
    """Existing capacity in service in `first_year`, MW, by (zone, tech) -- generators plus
    storage power capacity."""
    def in_service(df):
        start = df["start_year"].astype(float)
        return df[~df["expandable"].astype(bool) & (start <= first_year) & (first_year < start + df["life"])]

    gens = in_service(generator_constants)[["zone", "tech", "capacity"]]
    sto = in_service(storage_constants)[["zone", "tech", "capacity_mw"]].rename(columns={"capacity_mw": "capacity"})
    return pd.concat([gens, sto]).groupby(["zone", "tech"])["capacity"].sum().unstack("tech").fillna(0.0)


def _stack_key(tech):
    key = str(tech).lower()
    return (TECH_STACK_ORDER.index(key), key) if key in TECH_STACK_ORDER else (len(TECH_STACK_ORDER), key)


def build_options(generator_constants, storage_constants):
    """Build options by tech: count, and summed total build limit (MW for generators, MWh for
    storage; NaN if any option of that tech has no limit)."""
    def by_tech(df, limit_col, unit):
        df = df[df["expandable"].astype(bool)]
        g = df.groupby("tech")[limit_col]
        return pd.DataFrame({"count": g.size(), "limit": g.sum(min_count=1).where(g.count() == g.size()), "unit": unit})

    return pd.concat([
        by_tech(generator_constants, "total_build_limit", "MW"),
        by_tech(storage_constants, "total_build_limit_mwh", "MWh"),
    ])


def _tech_pie(ax, values, labels, title, show_counts=False):
    """Pie of `values` (indexed by tech, in stack order), `labels` in a legend below. Wedges of
    5% or more are labeled with their share, or with the value itself if `show_counts`."""
    total = values.sum()
    wedge_label = lambda pct: "" if pct < 5 else (f"{pct * total / 100:.0f}" if show_counts else f"{pct:.0f}%")
    wedges, _, _ = ax.pie(
        values.values, colors=[tech_color(t) for t in values.index], startangle=90, counterclock=False,
        autopct=wedge_label, pctdistance=0.72,
        textprops={"fontsize": 7, "color": "white", "weight": "bold"},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.legend(wedges, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=7)
    ax.set_title(title, loc="center")
    ax.set_aspect("equal")


def plot_capacity_mix(generator_constants, storage_constants, first_year):
    """Two pies by tech: existing capacity in `first_year` (MW, summed over zones), and build
    options by count -- not by build limit, which is often missing or an "unlimited"
    placeholder (e.g. Botswana's 10,000 MW); the legend gives each tech's summed limit."""
    cap = starting_capacity(generator_constants, storage_constants, first_year).sum()
    cap = cap[cap > 0]
    cap = cap[sorted(cap.index, key=_stack_key)]
    opts = build_options(generator_constants, storage_constants)
    opts = opts.loc[sorted(opts.index, key=_stack_key)]

    n_rows = max(len(cap), len(opts), 1)
    fig, (ax_cap, ax_opt) = plt.subplots(1, 2, figsize=(FIGSIZE[0], 2.6 + 0.14 * n_rows))
    total = cap.sum()
    _tech_pie(ax_cap, cap, [f"{t}  {mw:,.0f} MW ({mw / total:.0%})" for t, mw in cap.items()],
              f"Starting capacity ({first_year}): {total:,.0f} MW")
    if len(opts):
        limit = lambda r: "no limit" if pd.isna(r["limit"]) else f"limit {r['limit']:,.0f} {r['unit']}"
        _tech_pie(ax_opt, opts["count"],
                  [f"{t}  x{int(r['count'])} ({limit(r)})" for t, r in opts.iterrows()],
                  f"Build options: {int(opts['count'].sum())}", show_counts=True)
    else:
        ax_opt.set_axis_off()
        ax_opt.set_title("No build options", loc="center")
    fig.tight_layout()
    return fig


def plot_re_profiles(renewable_profiles, generator_constants, times):
    """Every renewable profile used by a generator, one panel per tech (in that tech's color).
    Thin lines are individual profiles, colored lines each zone's mean for that tech (zones
    distinguished by line style when there are several)."""
    time_order = times["time"].tolist()
    position = {t: i for i, t in enumerate(time_order)}
    used = generator_constants.dropna(subset=["profile_id"]).drop_duplicates("profile_id")[["profile_id", "zone", "tech"]]
    profiles = renewable_profiles.drop(columns=["zone", "tech"], errors="ignore").merge(used, on="profile_id")
    profiles["x"] = profiles["time"].map(position)

    zones = sorted(profiles["zone"].unique())
    techs = sorted(profiles["tech"].unique(), key=_stack_key)
    styles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(len(techs), 1, figsize=(FIGSIZE[0], 1.25 * len(techs) + 0.7), squeeze=False, sharex=True)
    for ax, tech in zip(axes[:, 0], techs):
        t = profiles[profiles["tech"] == tech]
        color = tech_color(tech)
        for _, p in t.groupby("profile_id"):
            p = p.sort_values("x")
            ax.plot(p["x"], p["value"], color="0.6", linewidth=0.3, alpha=0.35)
        for i, zone in enumerate(zones):
            z = t[t["zone"] == zone]
            if z.empty:
                continue
            mean = z.groupby("x")["value"].mean()
            label = f"{zone} mean" if len(zones) > 1 else "mean"
            ax.plot(mean.index, mean.values, color=color, linewidth=1.5, linestyle=styles[i % len(styles)], label=label)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(f"{tech}  (n={t['profile_id'].nunique()})", fontsize=8.5, color=color)
        if len(zones) > 1:
            ax.legend(loc="upper right", fontsize=8)
    axes[0, 0].figure.suptitle("Renewable profiles by tech", x=0.01, ha="left", fontsize=9.5, weight="bold")
    fig.supylabel("Capacity factor", fontsize=8.5)
    _time_axis(axes[-1, 0], times)
    fig.tight_layout()
    return fig
