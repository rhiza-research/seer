"""What a config would make, without making it: runs, scenarios, and each perturbation's ranges (run.py --plan)."""
import json

import numpy as np
import pandas as pd

from ..schema import SeerInstance
from ..scenarios.profiles import HYDRO_TECHS, profile_categories
from ..scenarios.scenario_parameters import (PARAMETERS, SCENARIO_DATA, get_parameter_levels, get_tech_parameter,
                                             get_vom_trajectory, parse)
from .design import build_entries, scenarios


def runs_table(config):
    """A markdown table of config's runs: scenarios, iters, perturbations, solve."""
    rows = []
    for run in config["runs"]:
        n_scenarios = len(scenarios(run.get("scenario")))
        rows.append({"run": run["name"], "sweep_mode": (run.get("scenario") or {}).get("sweep_mode", "-"),
                     "scenarios": n_scenarios, "iters": run.get("iters", 1),
                     "perturbations": n_scenarios * run.get("iters", 1),
                     "perturbation": json.dumps(run["perturbation"]) if run.get("perturbation") else "-",
                     "solve": bool(run.get("solve", False))})
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] if rows else []
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in row.values()) + " |" for row in rows])


def print_plan(config, max_scenarios=20):
    entries = build_entries(config)
    print(f"{config['country']}: base {config['base']}, cost_dataset {config['cost_dataset']}\n")
    print(runs_table(config))
    print(f"\ntotal: {len(entries)} perturbations, {sum(e['solve'] for e in entries)} solved\n")
    for run in config["runs"]:
        named = [e for e in entries if e["run"] == run["name"]]
        names = list(dict.fromkeys(e["scenario_name"] for e in named))
        if names == ["base"]:
            continue
        print(f"{run['name']}: {len(names)} scenarios")
        for name in names[:max_scenarios]:
            print(f"  {next(e['id'] for e in named if e['scenario_name'] == name)}  {name}")
        if len(names) > max_scenarios:
            print(f"  ... and {len(names) - max_scenarios} more")
    base = SeerInstance.read(config["base"])
    for spec in dict.fromkeys(json.dumps(run["perturbation"]) for run in config["runs"] if run.get("perturbation")):
        around = [r["name"] for r in config["runs"] if json.dumps(r.get("perturbation")) == spec]
        print(f"\nperturbation {spec} (runs {', '.join(around)}): ranges around the base values "
              f"(around each scenario's levels instead, for runs with a scenario)")
        print(perturbation_ranges(base, json.loads(spec), config["cost_dataset"]).to_string(index=False))


# ---------------------------------------------------------------- perturbation ranges


def perturbation_ranges(instance, perturbation, cost_dataset):
    """Per parameter (and item) the perturbation scales: base value, reference low / high, and draw range."""
    options = parse(None, perturbation, cost_dataset=cost_dataset)
    years = [int(y) for y in instance.years["year"]]
    rows = []
    for parameter, item, base, unit in base_values(instance, years):
        scale = options.scale(parameter, item)
        if not scale:
            continue
        if base is None:
            rows.append({"parameter": parameter, "item": item or "", "unit": unit, "note": "no effect: nothing to perturb"})
            continue
        levels = get_parameter_levels(parameter, item, years, cost_dataset)
        low, high = (float(np.min([np.ravel(v)[-1] for v in levels.values()])),
                     float(np.max([np.ravel(v)[-1] for v in levels.values()])))
        if parameter == "capex_start":
            low, high = low / 1000, high / 1000  # $/MW -> $/kW (storage: $/MWh -> $/kWh)
        lo, hi = base - scale * (base - min(low, base)), base + scale * (max(high, base) - base)
        note = "one-sided: base below low" if base < low else "one-sided: base above high" if base > high else ""
        rows.append({"parameter": parameter, "item": item or "", "unit": unit, "scale": scale, "base": base,
                     "ref_low": low, "ref_high": high, "draw_low": lo, "draw_high": hi, "note": note})
    table = pd.DataFrame(rows, columns=["parameter", "item", "unit", "scale", "base", "ref_low", "ref_high",
                                        "draw_low", "draw_high", "note"])
    return table.fillna("").map(lambda v: f"{v:.4g}" if isinstance(v, float) else v)


def base_values(instance, years):
    """(parameter, item, base value or None if nothing to act on, unit) for every parameter perturb can scale."""
    settings = instance.settings.set_index("parameters")["value"]
    for parameter, unit in [("dr", "fraction"), ("voll", "$/MWh")]:
        yield parameter, None, float(settings[parameter]), unit

    growth = []
    for _, group in instance.demand_forecast.groupby(["zone", "type"]):
        group = group.sort_values("year")
        span = group["year"].iloc[-1] - group["year"].iloc[0]
        if span:
            growth.append((group["value"].iloc[-1] / group["value"].iloc[0]) ** (1 / span) - 1)
    yield "demand_growth", None, float(np.median(growth)) if growth else None, "/yr (median over zones)"

    day = instance.demand_profile.assign(day=instance.demand_profile["time"].str[:5])
    keys = [c for c in ["zone", "year", "day"] if c in day.columns]
    load_factor = day.groupby(keys)["value"].agg(lambda v: v.mean() / v.max())
    yield "demand_profile", None, float(load_factor.mean()), "daily load factor (mean over days)"

    profiles = instance.renewable_profiles.groupby("profile_id")["value"].mean()
    category = profile_categories(instance)
    for cat in SCENARIO_DATA["re_profile"]:
        cfs = [cf for pid, cf in profiles.items() if category.get(pid) == cat]
        yield "re_profile", cat, float(np.mean(cfs)) if cfs else None, "capacity factor (mean over profiles)"

    gens = instance.generator_constants
    hydro = gens.loc[gens["tech"].isin(HYDRO_TECHS) & gens["isre"].astype(bool), "profile_id"]
    rows = instance.renewable_profiles["profile_id"].isin(hydro)
    yield "hydro", None, float(instance.renewable_profiles.loc[rows, "value"].mean()) if rows.any() else None, "capacity factor"

    for parameter, item, base in cost_bases(instance, years):
        unit = {"capex_start": "$/kW (storage $/kWh)", "fuel_start": "$/MMBtu"}.get(parameter, f"ratio {years[-1]} / {years[0]}")
        yield parameter, item, base, unit


def cost_bases(instance, years):
    """(parameter, tech or fuel, base) for capex and fuel starts and pathways: medians over the units."""
    values = {}
    for costs, constants, id_col in [(instance.generator_costs, instance.generator_constants, "generator"),
                                     (instance.storage_costs, instance.storage_constants, "storage")]:
        year_cols = [c for c in costs.columns if str(c).isdigit()]
        units = costs.merge(constants, on=id_col)
        for _, unit in units.iterrows():
            curve = unit[year_cols].to_numpy(dtype=float)
            if unit["cost_type"] == "capital" and unit["expandable"] and curve[0] > 0:
                if get_parameter_levels("capex_start", unit["tech"], years) is not None:
                    values.setdefault(("capex_start", unit["tech"]), []).append(curve[0] / 1000)
                    values.setdefault(("capex_pathway", unit["tech"]), []).append(curve[-1] / curve[0])
            if id_col == "generator" and unit["cost_type"] == "marginal":
                heat_rate = get_tech_parameter(unit["tech"], "heat_rate")
                if heat_rate is None or get_parameter_levels("fuel_start", unit["fuel"], years) is None:
                    continue
                fuel = np.clip((curve - get_vom_trajectory(unit["tech"], years)) / heat_rate, 1e-6, None)
                values.setdefault(("fuel_start", unit["fuel"]), []).append(fuel[0])
                values.setdefault(("fuel_pathway", unit["fuel"]), []).append(fuel[-1] / fuel[0])
    for (parameter, item), v in sorted(values.items()):
        yield parameter, item, float(np.median(v))
    for parameter in [p for p in PARAMETERS if p.startswith(("capex", "fuel"))]:
        if not any(p == parameter for p, _ in values):
            yield parameter, None, None
