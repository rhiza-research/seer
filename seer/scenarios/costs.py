"""Sample and resample cost trajectories (generator_costs.csv, storage_costs.csv) from
reference_data/technology and reference_data/fuels (built by scripts/build_cost_ranges.py,
scripts/build_fuel_prices.py and scripts/build_tech_parameters.py).

A trajectory is start x pathway, and the two are sampled independently: start is the value in the
first modeled year, pathway its ratio to that year. Each has its own low / mid / high reference.
"""
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import sample_value

REFERENCE_DIR = Path(__file__).parent / "reference_data"


@cache
def load_table(name):
    return pd.read_csv(REFERENCE_DIR / name)


def tech_parameter(tech, column):
    params = load_table("technology/tech_parameters.csv").set_index("tech")
    return params.loc[tech, column] if tech in params.index and pd.notna(params.loc[tech, column]) else None


def reference(costs, pathways, key_col, key, parameter, years):
    """({"low", "mid", "high"} start values in years[0], {"low", "mid", "high"} pathways over years
    as ratios to years[0]); None if there's no reference for key. The tables' base-year start is
    moved to years[0] along the mid pathway; pathways are held flat outside the tables' years."""
    rows = costs[(costs[key_col] == key) & (costs["parameter"] == parameter)]
    if rows.empty:
        return None
    path_rows = pathways[(pathways[key_col] == key) & (pathways["parameter"] == parameter)].set_index("scenario")
    table_years = [int(c) for c in path_rows.columns if c.isdigit()]
    pathway = {}
    for level in ["low", "mid", "high"]:
        curve = pd.Series(path_rows.loc[level, [str(y) for y in table_years]].to_numpy(dtype=float), index=table_years)
        curve = curve.reindex(sorted(set(table_years) | set(years))).interpolate(method="index").ffill().bfill().loc[years]
        pathway[level] = curve.to_numpy()
    start = {level: rows.iloc[0][level] * pathway["mid"][0] for level in ["low", "mid", "high"]}
    return start, {level: p / p[0] for level, p in pathway.items()}


def sample_trajectory(start_level, pathway_level, own, ref, rng, scale):
    """start x pathway: each part sampled at its level around own's start / pathway (random), or kept
    as own's if its level is None."""
    start_ref, pathway_ref = ref
    start = own[0] if start_level is None else sample_value(start_level, own[0], start_ref, rng, scale)
    pathway = own / own[0] if pathway_level is None else sample_value(pathway_level, own / own[0], pathway_ref, rng, scale)
    return start * pathway / pathway[0]  # a pathway is 1 in the first year; its noise only changes its shape


def per_key_rng(rng):
    """key -> an rng seeded once per key from rng, so every row sharing a key (tech, fuel) gets the
    same draw."""
    seeds = {}

    def rng_for(key):
        seeds.setdefault(key, int(rng.integers(2**63)))
        return np.random.default_rng(seeds[key])
    return rng_for


def year_columns(df):
    return [c for c in df.columns if str(c).isdigit()]


class CapexSampler:
    """Sample build options' capital cost trajectories ($/MW, or $/MWh of energy for storage)."""

    to_perturb = [("generator_costs", "generator_constants", "generator"), ("storage_costs", "storage_constants", "storage")]

    def reference(self, tech, years):
        ref = reference(load_table("technology/tech_costs.csv"), load_table("technology/tech_cost_pathways.csv"), "tech", tech, "capex", years)
        if ref is None:
            return None
        per_mw = 1000 / (tech_parameter(tech, "duration_hours") or 1)  # $/kW -> $/MW, or $/MWh for storage
        return {level: v * per_mw for level, v in ref[0].items()}, ref[1]

    def sample(self, start_level, pathway_level, tech, years, rng, scale=1.0):
        """Capital cost trajectory for a new build option of tech (random draws around mid)."""
        ref = self.reference(tech, years)
        return sample_trajectory(start_level, pathway_level, ref[0]["mid"] * ref[1]["mid"], ref, rng, scale)

    def resample(self, seer_instance, rng, start_level=None, pathway_level=None, scale=1.0):
        """Return a copy of seer_instance with every build option's capital cost trajectory resampled
        (random draws around the instance's own start / pathway; a None level keeps the instance's
        own), one draw per tech. Existing units' capital (0) is untouched."""
        out = seer_instance.copy()
        rng_for = per_key_rng(rng)
        for costs_name, constants_name, id_col in self.to_perturb:
            df = getattr(out, costs_name)
            constants = getattr(out, constants_name).set_index(id_col)
            year_cols = year_columns(df)
            years = [int(c) for c in year_cols]
            for i, row in df[df["cost_type"] == "capital"].iterrows():
                unit = constants.loc[row[id_col]]
                if not unit["expandable"]:
                    continue
                ref = self.reference(unit["tech"], years)
                if ref is None:
                    warnings.warn(f"no capex reference for tech {unit['tech']!r} -- leaving {row[id_col]!r} unchanged")
                    continue
                own = row[year_cols].to_numpy(dtype=float)
                df.loc[i, year_cols] = sample_trajectory(start_level, pathway_level, own, ref, rng_for(unit["tech"]), scale)
            setattr(out, costs_name, df)
        return out


class MarginalSampler:
    """Sample thermal generators' marginal cost trajectories ($/MWh): heat_rate x fuel price + vom,
    with the fuel price's start and pathway sampled."""

    to_perturb = "generator_costs"

    def reference(self, tech, fuel, years):
        """(heat_rate, vom over years, fuel price reference); None without a heat rate and fuel price."""
        heat_rate = tech_parameter(tech, "heat_rate")
        price = reference(load_table("fuels/fuel_prices.csv"), load_table("fuels/fuel_price_pathways.csv"), "fuel", fuel, "price", years)
        if heat_rate is None or price is None:
            return None
        vom = reference(load_table("technology/tech_costs.csv"), load_table("technology/tech_cost_pathways.csv"), "tech", tech, "vom", years)
        vom = np.zeros(len(years)) if vom is None else vom[0]["mid"] * vom[1]["mid"]
        return heat_rate, vom, price

    def sample(self, start_level, pathway_level, tech, fuel, years, rng, scale=1.0):
        """Marginal cost trajectory for tech burning fuel (random draws around mid)."""
        heat_rate, vom, price_ref = self.reference(tech, fuel, years)
        price = sample_trajectory(start_level, pathway_level, price_ref[0]["mid"] * price_ref[1]["mid"], price_ref, rng, scale)
        return heat_rate * price + vom

    def resample(self, seer_instance, rng, start_level=None, pathway_level=None, scale=1.0):
        """Return a copy of seer_instance with every thermal generator's marginal cost trajectory
        resampled: its implied fuel price ((marginal - vom) / heat_rate) is resampled (random draws
        around it; a None level keeps its own) and recombined. Generators with no heat rate or fuel
        price reference (renewables, hydro, ...) are untouched. One draw per fuel."""
        out = seer_instance.copy()
        rng_for = per_key_rng(rng)
        df = getattr(out, self.to_perturb)
        constants = out.generator_constants.set_index("generator")
        year_cols = year_columns(df)
        years = [int(c) for c in year_cols]
        for i, row in df[df["cost_type"] == "marginal"].iterrows():
            unit = constants.loc[row["generator"]]
            ref = self.reference(unit["tech"], unit["fuel"], years)
            if ref is None:
                continue
            heat_rate, vom, price_ref = ref
            own_price = np.clip((row[year_cols].to_numpy(dtype=float) - vom) / heat_rate, 1e-6, None)
            df.loc[i, year_cols] = heat_rate * sample_trajectory(start_level, pathway_level, own_price, price_ref, rng_for(unit["fuel"]), scale) + vom
        setattr(out, self.to_perturb, df)
        return out
