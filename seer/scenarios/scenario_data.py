"""Scenario reference data: scenario_data.yaml and the technology / fuel tables in reference_data,
and parameter_levels -- the one place that knows where each parameter's low / mid / high live.
"""
from functools import cache
from pathlib import Path

import pandas as pd
import yaml

DATA_DIR = Path(__file__).parent / "reference_data"
SCENARIO_DATA = yaml.safe_load((DATA_DIR / "scenario_data.yaml").read_text())


@cache
def load_table(name):
    return pd.read_csv(DATA_DIR / name)


def tech_parameter(tech, column):
    params = load_table("technology/tech_parameters.csv").set_index("tech")
    return params.loc[tech, column] if tech in params.index and pd.notna(params.loc[tech, column]) else None


def table_levels(table, key, parameter, years):
    """(start, pathway) low / mid / high for key's parameter in a technology or fuel table (table:
    "tech" or "fuel"); None if absent. start is the value in years[0] (the table's base-year value
    moved along the mid pathway), pathway the ratio to years[0], held flat outside the table's years."""
    starts, pathways = {"tech": ("technology/tech_cost_starts.csv", "technology/tech_cost_pathways.csv"),
                        "fuel": ("fuels/fuel_cost_starts.csv", "fuels/fuel_cost_pathways.csv")}[table]
    starts, pathways = load_table(starts), load_table(pathways)
    rows = starts[(starts[table] == key) & (starts["parameter"] == parameter)]
    if rows.empty:
        return None
    path_rows = pathways[(pathways[table] == key) & (pathways["parameter"] == parameter)].set_index("scenario")
    table_years = [int(c) for c in path_rows.columns if c.isdigit()]
    pathway = {}
    for level in ["low", "mid", "high"]:
        curve = pd.Series(path_rows.loc[level, [str(y) for y in table_years]].to_numpy(dtype=float), index=table_years)
        pathway[level] = curve.reindex(sorted(set(table_years) | set(years))).interpolate(method="index").ffill().bfill().loc[years].to_numpy()
    start = {level: rows.iloc[0][level] * pathway["mid"][0] for level in ["low", "mid", "high"]}
    return start, {level: p / p[0] for level, p in pathway.items()}


def parameter_levels(parameter, key=None, years=None):
    """{"low", "mid", "high"} for parameter (and key: a tech, fuel or solar / wind), in model units;
    None if key has no reference. Cost parameters are trajectories over years."""
    if parameter in ("dr", "voll", "demand_growth", "demand_profile"):
        return {level: SCENARIO_DATA[parameter][level] for level in ["low", "mid", "high"]}
    if parameter == "re_profile":
        return SCENARIO_DATA["re_profile"].get(key)
    if parameter == "hydro":
        return {level: tech_parameter("hydro", f"cf_{level}") for level in ["low", "mid", "high"]}
    if parameter in ("capex_start", "capex_pathway"):
        levels = table_levels("tech", key, "capex", years)
        if levels is None:
            return None
        per_mw = 1000 / (tech_parameter(key, "duration_hours") or 1)  # $/kW -> $/MW, or $/MWh of energy for storage
        return {level: v * per_mw for level, v in levels[0].items()} if parameter == "capex_start" else levels[1]
    if parameter in ("fuel_start", "fuel_pathway"):
        levels = table_levels("fuel", key, "price", years)
        return None if levels is None else levels[0] if parameter == "fuel_start" else levels[1]
    raise ValueError(f"unknown parameter {parameter!r}")
