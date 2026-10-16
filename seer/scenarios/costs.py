"""Cost scenarios: capex and marginal-cost trajectories, from reference tables in
seer/scenarios/reference_data/cost_assumptions (built by scripts/build_cost_ranges.py and
scripts/build_fuel_prices.py).

CapexPerturbation: sample a starting-cost multiplier and a decline trajectory independently per
tech, then rescale every generator/storage unit's capex trajectory from its own existing starting
value -- never from reference_data's absolute dollar figures, which belong to a different
(US/global-average) context than any given instance's real costs.

MarginalPerturbation: unlike capex, seer's marginal_cost is one undifferentiated $/MWh number
with no stored fuel/VOM split. We estimate the fuel-driven share as heat_rate[tech] (MMBtu/MWh,
from costs.csv) x fuel_price[fuel] ($/MMBtu, from fuel_prices.csv), treat the rest of the existing
marginal_cost as a flat non-fuel residual, and perturb only the fuel share -- sampling a starting-
price multiplier and a price trajectory independently per fuel (not per tech, since a commodity's
price doesn't depend on which plant burns it; cached across techs sharing a fuel).

Techs with no heat_rate entry (renewables, storage, CHP, etc. -- anything without a real fuel
commodity) are left with marginal_cost unperturbed; this only ever touches thermal generators
burning gas/oil/coal.

Both perturbations share the same core (sample_scenario, below): sample a start multiplier +
decline-trajectory ratio for a key (tech or fuel) via one of three modes, then rescale a cost-table
value from its own start-year level.
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .base import Perturbation

REFERENCE_DIR = Path(__file__).parent / "reference_data" / "cost_assumptions"

TRAJECTORY_SCENARIOS = ["low", "mid", "high"]


def load_capex_tables(reference_dir=REFERENCE_DIR):
    return (
        pd.read_csv(reference_dir / "costs.csv"),
        pd.read_csv(reference_dir / "decline_curves.csv"),
    )


def load_fuel_tables(reference_dir=REFERENCE_DIR):
    return (
        pd.read_csv(reference_dir / "costs.csv"),
        pd.read_csv(reference_dir / "fuel_prices.csv"),
        pd.read_csv(reference_dir / "decline_curves_fuel.csv"),
    )


def trajectory_curve(decline_df, key_col, key, value_col, scenario, years, start_year):
    """One scenario's decline curve for `key` (a tech or fuel), indexed by year and filled onto
    every year in `years` plus start_year -- interpolated between the table's own years, held
    flat beyond either end."""
    curve = decline_df[(decline_df[key_col] == key) & (decline_df["scenario"] == scenario)].set_index("year")[value_col].sort_index()
    all_years = sorted(set(curve.index) | set(years) | {start_year})
    return curve.reindex(all_years).interpolate(method="index").ffill().bfill()


def sample_scenario(low_mult, high_mult, decline_df, key_col, key, value_col, years, start_year, rng, mode="uniform", scenario=None):
    """Sample a start-year multiplier and a year-by-year ratio-to-start-year trajectory for `key`
    (a tech or fuel), from the same low/mid/high scenario data (low_mult/1.0/high_mult for the
    start value, the low/mid/high curves in `decline_df` for the trajectory), via one of three
    modes -- the piece shared by every cost perturbation (and, later, cost creation) that rescales
    a value from its own start-year level:

      "uniform" (default): draw one quantile u ~ Uniform(0, 1); the start multiplier AND the
        trajectory curve are each interpolated between their own low and high anchors at that
        SAME u, so a low draw is a low start cost with a low-decline trajectory together, not an
        inconsistent mix of a continuous price with an independently-random trajectory shape.
      "random_scenario": pick one of low/mid/high uniformly at random, and use THAT scenario's
        own start value and its own trajectory curve.
      "scenario": use the caller-supplied `scenario` ("low"/"mid"/"high") explicitly --
        deterministic, so every tech/fuel in one perturbation gets the same named scenario.

    Returns (start_multiplier, info, ratio) where ratio[start_year] == 1.0 and `info` records what
    was drawn (`{"mode": "uniform", "quantile": u}` or `{"mode": mode, "scenario": scenario}`). If
    `key` has no decline curve at all, ratio is flat (all 1.0) in every mode."""
    key_decline = decline_df[decline_df[key_col] == key]

    if mode == "uniform":
        u = float(rng.uniform(0.0, 1.0))
        start_mult = low_mult + u * (high_mult - low_mult)
        info = {"mode": "uniform", "quantile": u}
        if key_decline.empty:
            return float(start_mult), info, {y: 1.0 for y in years}
        low_curve = trajectory_curve(key_decline, key_col, key, value_col, "low", years, start_year)
        high_curve = trajectory_curve(key_decline, key_col, key, value_col, "high", years, start_year)
        curve = (1 - u) * low_curve + u * high_curve
        return float(start_mult), info, (curve / curve.loc[start_year]).to_dict()

    if mode == "random_scenario":
        scenario = str(rng.choice(TRAJECTORY_SCENARIOS))
    elif mode != "scenario":
        raise ValueError(f"mode must be one of 'uniform', 'random_scenario', 'scenario', got {mode!r}")
    elif scenario not in TRAJECTORY_SCENARIOS:
        raise ValueError(f"scenario must be one of {TRAJECTORY_SCENARIOS}, got {scenario!r}")

    start_mult = {"low": low_mult, "mid": 1.0, "high": high_mult}[scenario]
    info = {"mode": mode, "scenario": scenario}
    if key_decline.empty:
        return float(start_mult), info, {y: 1.0 for y in years}
    curve = trajectory_curve(key_decline, key_col, key, value_col, scenario, years, start_year)
    return float(start_mult), info, (curve / curve.loc[start_year]).to_dict()


def perturb_capex(cost_table, constants_table, id_col, ranges_df, decline_df, start_year, rng, mode="uniform", scenario=None):
    cost_table = cost_table.copy()
    year_cols = [c for c in cost_table.columns if c not in (id_col, "cost_type")]
    years = [int(c) for c in year_cols]
    capex_rows = cost_table["cost_type"] == "capital"
    start_col = next(c for c in year_cols if int(c) == start_year)
    ranges = ranges_df[(ranges_df["parameter"] == "capex") & ranges_df["method"].isin(["scenario", "multiplier"])].set_index("tech")

    sampled = {}
    for tech in constants_table["tech"].dropna().unique():
        if tech not in ranges.index:
            warnings.warn(f"CapexPerturbation: no capex range data for tech {tech!r} -- leaving its capex unperturbed.")
            continue
        if tech not in decline_df["tech"].values:
            warnings.warn(f"CapexPerturbation: no decline curve for tech {tech!r} -- holding capex trajectory flat at the perturbed start value.")

        low_mult, high_mult = ranges.loc[tech, ["low_mult", "high_mult"]]
        start_mult, info, ratio = sample_scenario(low_mult, high_mult, decline_df, "tech", tech, "capex_fraction", years, start_year, rng, mode, scenario)
        sampled[tech] = {"start_multiplier": start_mult, **info}

        ids = constants_table.loc[constants_table["tech"] == tech, id_col]
        rows = capex_rows & cost_table[id_col].isin(ids)
        if not rows.any():
            continue

        new_start_values = cost_table.loc[rows, start_col] * start_mult
        for year_col in year_cols:
            cost_table.loc[rows, year_col] = new_start_values * ratio[int(year_col)]

    return cost_table, sampled


@dataclass
class CapexPerturbation(Perturbation):
    reference_dir: Path = REFERENCE_DIR
    mode: str = "uniform"  # "uniform" / "random_scenario" / "scenario" -- see sample_scenario
    scenario: str | None = None  # required ("low"/"mid"/"high") when mode == "scenario"
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        ranges, decline = load_capex_tables(self.reference_dir)
        start_year = int(min(out.years["year"]))

        out.generator_costs, gen_sampled = perturb_capex(
            out.generator_costs, out.generator_constants, "generator", ranges, decline, start_year, rng, self.mode, self.scenario
        )
        out.storage_costs, sto_sampled = perturb_capex(
            out.storage_costs, out.storage_constants, "storage", ranges, decline, start_year, rng, self.mode, self.scenario
        )
        self.sampled_ = {"generator": gen_sampled, "storage": sto_sampled}
        return out

    def params(self):
        return {"mode": self.mode, "scenario": self.scenario, "sampled": self.sampled_}


def perturb_marginal(cost_table, constants_table, id_col, costs_df, fuel_prices, fuel_decline, start_year, rng, mode="uniform", scenario=None):
    cost_table = cost_table.copy()
    year_cols = [c for c in cost_table.columns if c not in (id_col, "cost_type")]
    years = [int(c) for c in year_cols]
    start_col = next(c for c in year_cols if int(c) == start_year)
    marginal_rows = cost_table["cost_type"] == "marginal"

    heat_rates = costs_df[costs_df["parameter"] == "heat_rate"].set_index("tech")["central_value"]
    fuel_ranges = fuel_prices[fuel_prices["parameter"] == "price"].set_index("fuel")
    fuel_by_tech = constants_table.drop_duplicates("tech").set_index("tech")["fuel"]

    sampled = {}
    for tech in constants_table["tech"].dropna().unique():
        if tech not in heat_rates.index:
            continue
        fuel = fuel_by_tech.get(tech)
        if fuel not in fuel_ranges.index:
            warnings.warn(f"MarginalPerturbation: no fuel price data for fuel {fuel!r} (tech {tech!r}) -- leaving its marginal_cost unperturbed.")
            continue

        if fuel not in sampled:
            low_mult, high_mult = fuel_ranges.loc[fuel, ["low_mult", "high_mult"]]
            start_mult, info, ratio = sample_scenario(low_mult, high_mult, fuel_decline, "fuel", fuel, "price_fraction", years, start_year, rng, mode, scenario)
            sampled[fuel] = {"start_multiplier": start_mult, "ratio": ratio, **info}
        start_mult, ratio = sampled[fuel]["start_multiplier"], sampled[fuel]["ratio"]

        ids = constants_table.loc[constants_table["tech"] == tech, id_col]
        rows = marginal_rows & cost_table[id_col].isin(ids)
        if not rows.any():
            continue

        # fuel share of the EXISTING marginal_cost, estimated at reference-data's own
        # (unperturbed) central price -- the rest is held as a flat residual (O&M etc.), never
        # touched. The perturbed fuel cost (which start_mult/ratio actually act on) is a separate
        # quantity, so start_mult visibly shifts the start-year total too, not just future years.
        fuel_cost_estimate = heat_rates.loc[tech] * float(fuel_ranges.loc[fuel, "central_value"])
        perturbed_fuel_cost_start = fuel_cost_estimate * start_mult
        residual = (cost_table.loc[rows, start_col] - fuel_cost_estimate).clip(lower=0)

        for year_col in year_cols:
            cost_table.loc[rows, year_col] = residual + perturbed_fuel_cost_start * ratio[int(year_col)]

    return cost_table, {fuel: {k: v[k] for k in v if k != "ratio"} for fuel, v in sampled.items()}


@dataclass
class MarginalPerturbation(Perturbation):
    reference_dir: Path = REFERENCE_DIR
    mode: str = "uniform"  # "uniform" / "random_scenario" / "scenario" -- see sample_scenario
    scenario: str | None = None  # required ("low"/"mid"/"high") when mode == "scenario"
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        costs_df, fuel_prices, fuel_decline = load_fuel_tables(self.reference_dir)
        start_year = int(min(out.years["year"]))

        out.generator_costs, gen_sampled = perturb_marginal(
            out.generator_costs, out.generator_constants, "generator", costs_df, fuel_prices, fuel_decline, start_year, rng, self.mode, self.scenario
        )
        self.sampled_ = {"generator": gen_sampled}
        return out

    def params(self):
        return {"mode": self.mode, "scenario": self.scenario, "sampled": self.sampled_}
