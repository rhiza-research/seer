"""Summaries and plots of solved model outputs (see seer/models/README.md), for per-country
records -- computed the same way for every backend from its output tables plus the inputs, so
pyomo and PyPSA results are directly comparable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..models.utils.time_utils import build_snapshot_calendar, compute_time_weights, load_time_weights, load_times
from .inputs import FIGSIZE, _stack_key
from .plot import _tech_stack_key, draw_dispatch_year, tech_color

OUTPUT_TABLES = ["generator_capacity", "storage_capacity", "dispatch"]
OPTIONAL_OUTPUT_TABLES = ["lost_load"]  # missing from results solved before it was added


def load_results(results_dir):
    """{table name: DataFrame} for one model's output folder."""
    results_dir = Path(results_dir)
    tables = {name: pd.read_csv(results_dir / f"{name}.csv") for name in OUTPUT_TABLES}
    for name in OPTIONAL_OUTPUT_TABLES:
        if (results_dir / f"{name}.csv").exists():
            tables[name] = pd.read_csv(results_dir / f"{name}.csv")
    return tables


def hour_weights(data_path, years):
    """{(year, hour): real hours that modeled hour stands for} -- the same weights the models use."""
    data_path = str(data_path).rstrip("/")
    calendar = build_snapshot_calendar(years, load_times(data_path), load_time_weights(data_path))
    return compute_time_weights(calendar)


def annual_demand(data_path, years):
    """{year: weighted demand (MWh) summed over zones} -- peak x profile x the hours each
    timepoint stands for, as the models see it."""
    data_path = Path(data_path)
    peak = pd.read_csv(data_path / "demand_forecast.csv")
    peak = peak[peak["type"] == "peak"]
    profile = pd.read_csv(data_path / "demand_profile.csv").merge(pd.read_csv(data_path / "times.csv"), on="time")
    zone_energy = (profile["value"] * profile["weight"]).groupby(profile["zone"]).sum()  # MWh per MW of peak
    peak = peak[peak["year"].isin(years)]
    return (peak["value"] * peak["zone"].map(zone_energy)).groupby(peak["year"]).sum().to_dict()


def capacity_built(capacity, constants, name_col):
    """Capacity added each year by build options: the year-on-year increase in each build
    option's capacity (the first modeled year counts in full). Returns (zone, tech, year, built)."""
    expandable = set(constants.loc[constants["expandable"].astype(bool), name_col])
    cap = capacity[capacity[name_col].isin(expandable)].sort_values([name_col, "year"])
    cap = cap.assign(built=cap.groupby(name_col)["capacity"].diff().fillna(cap["capacity"]).clip(lower=0))
    return cap.groupby(["zone", "tech", "year"], as_index=False)["built"].sum()


def summarize(results, data_path):
    """Totals for one model's results: built capacity (generators MW; storage MW and MWh),
    undiscounted capital and marginal cost over the run, weighted dispatch (GWh) by tech, and
    weighted lost load (GWh) with its undiscounted cost at settings.csv's `voll` (None if the
    results have no lost_load table)."""
    data_path = Path(data_path)
    gens = pd.read_csv(data_path / "generator_constants.csv")
    sto = pd.read_csv(data_path / "storage_constants.csv")
    gen_costs = pd.read_csv(data_path / "generator_costs.csv")
    sto_costs = pd.read_csv(data_path / "storage_costs.csv")
    years = sorted(results["dispatch"]["year"].unique())
    weights = hour_weights(data_path, years)

    def cost_lookup(costs, id_col, cost_type):
        table = costs[costs["cost_type"] == cost_type].set_index(id_col)
        return {(name, int(year)): table.at[name, year] for name in table.index for year in table.columns if year.isdigit()}

    # capacity built + capital cost
    gen_built = capacity_built(results["generator_capacity"], gens, "generator")
    per_gen = results["generator_capacity"].merge(gens[["generator", "expandable"]], on="generator")
    per_gen = per_gen[per_gen["expandable"].astype(bool)].sort_values(["generator", "year"])
    per_gen["built"] = per_gen.groupby("generator")["capacity"].diff().fillna(per_gen["capacity"]).clip(lower=0)
    capital = cost_lookup(gen_costs, "generator", "capital")
    gen_capex = sum(row.built * capital.get((row.generator, row.year), 0.0) for row in per_gen.itertuples())

    per_sto = results["storage_capacity"].merge(sto[["storage", "expandable", "duration_hours"]], on="storage")
    per_sto = per_sto[per_sto["expandable"].astype(bool)].sort_values(["storage", "year"])
    per_sto["built"] = per_sto.groupby("storage")["capacity"].diff().fillna(per_sto["capacity"]).clip(lower=0)
    sto_capital = cost_lookup(sto_costs, "storage", "capital")
    sto_capex = sum(row.built * sto_capital.get((row.storage, row.year), 0.0) for row in per_sto.itertuples())
    sto_built_mw = per_sto.assign(built=per_sto["built"] / per_sto["duration_hours"]).groupby(
        ["zone", "tech", "year"], as_index=False)["built"].sum()

    # weighted dispatch + marginal cost
    dispatch = results["dispatch"].copy()
    dispatch["hours"] = [weights[(y, h)] for y, h in zip(dispatch["year"], dispatch["hour"])]
    dispatch["energy"] = dispatch["dispatch"] * dispatch["hours"]  # MWh
    marginal = cost_lookup(gen_costs, "generator", "marginal")
    gen_names = set(gens["generator"])
    is_gen = dispatch["name"].isin(gen_names)
    marginal_cost = sum(
        e * marginal.get((n, int(y)), 0.0)
        for n, y, e in zip(dispatch.loc[is_gen, "name"], dispatch.loc[is_gen, "year"], dispatch.loc[is_gen, "energy"])
    )

    lost_load_gwh = lost_load_cost = lost_by_year = None
    if "lost_load" in results:
        lost = results["lost_load"]
        hours = [weights[(y, h)] for y, h in zip(lost["year"], lost["hour"])]
        lost_by_year = (lost["lost_load"] * hours).groupby(lost["year"]).sum().to_dict()  # MWh
        lost_mwh = float((lost["lost_load"] * hours).sum())
        lost_load_gwh = lost_mwh / 1e3
        lost_load_cost = lost_mwh * float(pd.read_csv(data_path / "settings.csv").set_index("parameters")["value"]["voll"])

    return {
        "years": years,
        "capital_cost": gen_capex + sto_capex,
        "lost_load_gwh": lost_load_gwh,
        "lost_load_cost": lost_load_cost,
        "lost_load_by_year": lost_by_year,
        "marginal_cost": marginal_cost,
        "dispatch_gwh": (dispatch.groupby("tech")["energy"].sum() / 1e3).to_dict(),
        "built_mw": pd.concat([gen_built, sto_built_mw]).groupby("tech")["built"].sum().to_dict(),
        "built_by_year": pd.concat([gen_built, sto_built_mw], ignore_index=True),
        "storage_built_mwh": per_sto.groupby("tech")["built"].sum().to_dict(),
    }


def comparison_rows(summaries, runs):
    """Rows [label, value per model..., difference] comparing model summaries side by side.
    `summaries`/`runs`: {model: summarize(...)} / {model: run.json dict}."""
    models = list(summaries)
    rows = []

    def add(label, values, fmt):
        vals = [values.get(m) for m in models]
        nums = [v for v in vals if v is not None]
        diff = ""
        if len(nums) == 2 and max(abs(nums[0]), abs(nums[1])) > 1e-9:
            diff = f"{(nums[1] - nums[0]) / max(abs(nums[0]), 1e-9) * 100:+.1f}%" if abs(nums[0]) > 1e-9 else "--"
        rows.append([label] + ["" if v is None else fmt(v) for v in vals] + [diff])

    add("Objective (discounted, $M)", {m: runs[m]["objective"] / 1e6 for m in models if m in runs}, lambda v: f"{v:,.1f}")
    add("Solve time (s)", {m: runs[m]["solve_seconds"] for m in models if m in runs}, lambda v: f"{v:,.0f}")
    add("Capital cost ($M)", {m: s["capital_cost"] / 1e6 for m, s in summaries.items()}, lambda v: f"{v:,.1f}")
    add("Marginal cost ($M)", {m: s["marginal_cost"] / 1e6 for m, s in summaries.items()}, lambda v: f"{v:,.1f}")
    lost = {m: s for m, s in summaries.items() if s.get("lost_load_gwh") is not None}
    if lost:
        add("Lost load (GWh)", {m: s["lost_load_gwh"] for m, s in lost.items()}, lambda v: f"{v:,.1f}")
        add("Lost load cost at VOLL ($M)", {m: s["lost_load_cost"] / 1e6 for m, s in lost.items()}, lambda v: f"{v:,.1f}")
    # per-tech rows grouped under a full-width label row (a 1-element row)
    rows.append(["Dispatch (GWh)"])
    techs = sorted({t for s in summaries.values() for t in s["dispatch_gwh"]}, key=_stack_key)
    for tech in techs:
        values = {m: s["dispatch_gwh"].get(tech, 0.0) for m, s in summaries.items()}
        if any(abs(v) > 0.05 for v in values.values()):
            add(tech, values, lambda v: f"{v:,.0f}")
    rows.append(["Built (MW)"])
    n_before = len(rows)
    built_techs = sorted({t for s in summaries.values() for t in s["built_mw"]}, key=_stack_key)
    for tech in built_techs:
        values = {m: s["built_mw"].get(tech, 0.0) for m, s in summaries.items()}
        if any(v > 0.05 for v in values.values()):
            add(tech, values, lambda v: f"{v:,.0f}")
    if len(rows) == n_before:
        rows.append(["none"] + ["0"] * len(models) + [""])
    return rows


# above this many zones, capacity-built bars are summed over zones rather than drawn side by side
MAX_GROUPED_ZONES = 20


def _draw_capacity_built(ax, built_by_year, years, techs, max_zones=MAX_GROUPED_ZONES):
    """Stacked (by tech) and grouped (by zone, side by side within each year) bars on `ax`.
    With more than `max_zones` zones the bars would be too thin to see, so zones are summed."""
    if built_by_year["zone"].nunique() > max_zones:
        built_by_year = built_by_year.groupby(["tech", "year"], as_index=False)["built"].sum().assign(zone="")
    zones = sorted(built_by_year["zone"].unique()) or [""]
    group_width = 0.8
    bar_width = group_width / len(zones)
    for zi, zone in enumerate(zones):
        x = np.arange(len(years)) - group_width / 2 + bar_width * (zi + 0.5)
        bottom = np.zeros(len(years))
        z = built_by_year[built_by_year["zone"] == zone]
        for tech in techs:
            values = z[z["tech"] == tech].set_index("year")["built"].reindex(years, fill_value=0.0).values
            ax.bar(x, values, bottom=bottom, width=bar_width * 0.92, color=tech_color(tech))
            bottom += values
        if len(zones) > 1:
            for xi in x:
                ax.text(xi, 0, zone, rotation=90, ha="center", va="bottom", fontsize=6, color="white")
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([str(y) for y in years], fontsize=7)


def plot_capacity_built(built_by_case, years, title="Capacity built by year"):
    """Capacity added each year, MW: one row per case, one panel per model
    ({case: {model: built_by_year}}, e.g. upfront vs annualized capex), shared y axis and one
    legend. Panels are titled "<model>, <case>" (just the model if there's a single unnamed case)."""
    years = list(years)
    all_built = pd.concat([b for case in built_by_case.values() for b in case.values()], ignore_index=True)
    techs = sorted([t for t, g in all_built.groupby("tech") if g["built"].sum() > 0.05], key=_stack_key)
    n_cols = max(len(case) for case in built_by_case.values())
    fig, axes = plt.subplots(len(built_by_case), n_cols, figsize=(FIGSIZE[0], FIGSIZE[1] * (0.4 + 0.6 * len(built_by_case))),
                             sharey=True, squeeze=False)
    for row, (case, built_by_model) in enumerate(built_by_case.items()):
        for ax, (model, built) in zip(axes[row], built_by_model.items()):
            _draw_capacity_built(ax, built, years, techs)
            ax.set_title(f"{model}, {case}" if case else model, fontsize=8.5, loc="left")
            if not techs:
                ax.text(0.5, 0.5, "No capacity built", transform=ax.transAxes, ha="center", va="center", color="0.5")
                ax.set_ylim(0, 1)
        for ax in axes[row][len(built_by_model):]:
            ax.set_visible(False)  # model not solved for this case
        axes[row][0].set_ylabel("Capacity added (MW)")
    if techs:
        handles = [plt.Rectangle((0, 0), 1, 1, color=tech_color(t)) for t in techs]
        fig.legend(handles, techs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(techs), fontsize=7.5, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, weight="bold")
    fig.tight_layout()
    return fig


def plot_lost_load_share(summaries_by_case, demand_by_year, title="Lost load by year"):
    """Annual lost load as % of demand, bars by year: one row per case, one panel per model
    ({case: {model: summary}}, as plot_capacity_built), shared y axis."""
    years = sorted(demand_by_year)
    n_cols = max(len(case) for case in summaries_by_case.values())
    fig, axes = plt.subplots(len(summaries_by_case), n_cols, figsize=(FIGSIZE[0], FIGSIZE[1] * (0.4 + 0.5 * len(summaries_by_case))),
                             sharey=True, squeeze=False)
    for row, (case, summaries) in enumerate(summaries_by_case.items()):
        for ax, (model, summary) in zip(axes[row], summaries.items()):
            lost = summary.get("lost_load_by_year") or {}
            share = [100 * lost.get(y, 0.0) / demand_by_year[y] for y in years]
            bars = ax.bar(np.arange(len(years)), share, width=0.7, color="#c0392b")
            ax.bar_label(bars, labels=[f"{v:.1f}%" for v in share], fontsize=6.5, padding=1)
            ax.set_xticks(np.arange(len(years)))
            ax.set_xticklabels([str(y) for y in years], fontsize=7)
            ax.set_title(f"{model}, {case}" if case else model, fontsize=8.5, loc="left")
        axes[row][0].set_ylabel("Lost load (% of demand)")
    axes[0][0].set_ylim(top=axes[0][0].get_ylim()[1] * 1.12)  # room for the labels
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, weight="bold")
    fig.tight_layout()
    return fig


def plot_dispatch(dispatch_by_model, title="Hourly dispatch by year"):
    """Hourly dispatch by tech: one row per year, one column per model ({model: dispatch}),
    all panels on a shared y axis with one legend."""
    all_dispatch = pd.concat(dispatch_by_model.values(), ignore_index=True)
    years = sorted(all_dispatch["year"].unique())
    active = all_dispatch.groupby("tech")["dispatch"].apply(lambda x: x.abs().max() > 1e-6)
    techs = sorted(active.index[active], key=_tech_stack_key)
    colors = {tech: tech_color(tech) for tech in techs}
    models = list(dispatch_by_model)
    fig, axes = plt.subplots(len(years), len(models), figsize=(FIGSIZE[0], 1.45 * len(years)),
                             sharey=True, squeeze=False)
    for col, model in enumerate(models):
        dispatch = dispatch_by_model[model]
        for row, year in enumerate(years):
            ax = axes[row][col]
            draw_dispatch_year(ax, dispatch[dispatch["year"] == year], techs, colors)
            ax.set_title(f"{model} {year}", fontsize=8, loc="left")
            ax.tick_params(labelsize=6.5)
            if col == 0:
                ax.set_ylabel("MW", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[tech]) for tech in techs]
    fig.legend(handles, techs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(techs), fontsize=7.5, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, weight="bold")
    fig.tight_layout()
    return fig
