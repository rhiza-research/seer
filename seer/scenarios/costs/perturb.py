"""Cost perturbations.

CostPerturbation (capex): sample a starting-cost multiplier and a decline trajectory
independently per tech (from seer/scenarios/reference_data/cost_assumptions), then rescale every
generator/storage unit's capex trajectory from its own existing starting value -- never from
reference_data's absolute dollar figures, which belong to a different (US/global-average)
context than any given instance's real costs.

MarginalCostPerturbation: unlike capex, seer's marginal_cost is one undifferentiated
$/MWh number with no stored fuel/VOM split. We estimate the fuel-driven share as
heat_rate[tech] (MMBtu/MWh, from costs.csv) x fuel_price[fuel] ($/MMBtu, from fuel_prices.csv),
treat the rest of the existing marginal_cost as a flat non-fuel residual, and perturb only the
fuel share -- sampling a starting-price multiplier and a price trajectory independently per
fuel (not per tech, since a commodity's price doesn't depend on which plant burns it).

Techs with no heat_rate entry (renewables, storage, CHP, etc. -- anything without a real fuel
commodity) are left with marginal_cost unperturbed; this only ever touches thermal generators
burning gas/oil/coal.
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from ..base import Perturbation
from .reference import REFERENCE_DIR, TRAJECTORY_SCENARIOS, load_capex_tables, load_fuel_tables, trajectory_curve


def perturb_capex(cost_table, constants_table, id_col, ranges_df, decline_df, start_year, rng):
    cost_table = cost_table.copy()
    year_cols = [c for c in cost_table.columns if c not in (id_col, "cost_type")]
    capex_rows = cost_table["cost_type"] == "capex"
    start_col = next(c for c in year_cols if int(c) == start_year)

    ranges = ranges_df[(ranges_df["parameter"] == "capex") & ranges_df["method"].isin(["scenario", "multiplier"])].set_index("tech")

    sampled = {}
    for tech in constants_table["tech"].dropna().unique():
        if tech not in ranges.index:
            warnings.warn(f"CostPerturbation: no capex range data for tech {tech!r} -- leaving its capex unperturbed.")
            continue

        low_mult, high_mult = ranges.loc[tech, ["low_mult", "high_mult"]]
        start_mult = rng.uniform(min(low_mult, high_mult), max(low_mult, high_mult))

        tech_decline = decline_df[decline_df["tech"] == tech]
        scenario = None
        if tech_decline.empty:
            warnings.warn(f"CostPerturbation: no decline curve for tech {tech!r} -- holding capex trajectory flat at the perturbed start value.")
        else:
            scenario = rng.choice(TRAJECTORY_SCENARIOS)
            curve = trajectory_curve(tech_decline, "tech", tech, "capex_fraction", scenario, [int(c) for c in year_cols], start_year)
            fraction_at_start = curve.loc[start_year]

        sampled[tech] = {"start_multiplier": float(start_mult), "trajectory_scenario": scenario}

        ids = constants_table.loc[constants_table["tech"] == tech, id_col]
        rows = capex_rows & cost_table[id_col].isin(ids)
        if not rows.any():
            continue

        new_start_values = cost_table.loc[rows, start_col] * start_mult
        for year_col in year_cols:
            ratio = curve.loc[int(year_col)] / fraction_at_start if scenario is not None else 1.0
            cost_table.loc[rows, year_col] = new_start_values * ratio

    return cost_table, sampled


@dataclass
class CostPerturbation(Perturbation):
    reference_dir: Path = REFERENCE_DIR
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        ranges, decline = load_capex_tables(self.reference_dir)
        start_year = int(min(out.years["year"]))

        out.generator_costs, gen_sampled = perturb_capex(
            out.generator_costs, out.generator_constants, "generator", ranges, decline, start_year, rng
        )
        out.storage_costs, sto_sampled = perturb_capex(
            out.storage_costs, out.storage_constants, "storage", ranges, decline, start_year, rng
        )
        self.sampled_ = {"generator": gen_sampled, "storage": sto_sampled}
        return out

    def params(self):
        return {"sampled": self.sampled_}


def perturb_marginal_cost(cost_table, constants_table, id_col, costs_df, fuel_prices, fuel_decline, years, start_year, rng):
    cost_table = cost_table.copy()
    year_cols = [c for c in cost_table.columns if c not in (id_col, "cost_type")]
    marginal_rows = cost_table["cost_type"] == "marginal_cost"

    heat_rates = costs_df[(costs_df["parameter"] == "heat_rate")].set_index("tech")["central_value"]
    fuel_ranges = fuel_prices[fuel_prices["parameter"] == "price"].set_index("fuel")

    sampled = {}
    fuel_by_tech = constants_table.drop_duplicates("tech").set_index("tech")["fuel"]

    for tech in constants_table["tech"].dropna().unique():
        if tech not in heat_rates.index:
            continue
        fuel = fuel_by_tech.get(tech)
        if fuel not in fuel_ranges.index:
            warnings.warn(f"MarginalCostPerturbation: no fuel price data for fuel {fuel!r} (tech {tech!r}) -- leaving its marginal_cost unperturbed.")
            continue

        if fuel not in sampled:
            low_mult, high_mult = fuel_ranges.loc[fuel, ["low_mult", "high_mult"]]
            start_mult = rng.uniform(min(low_mult, high_mult), max(low_mult, high_mult))
            scenario = rng.choice(TRAJECTORY_SCENARIOS)
            curve = trajectory_curve(fuel_decline, "fuel", fuel, "price_fraction", scenario, [int(c) for c in year_cols], start_year)
            sampled[fuel] = {"start_multiplier": float(start_mult), "trajectory_scenario": scenario, "curve": curve}

        start_mult, scenario, curve = sampled[fuel]["start_multiplier"], sampled[fuel]["trajectory_scenario"], sampled[fuel]["curve"]
        fraction_at_start = curve.loc[start_year]

        ids = constants_table.loc[constants_table["tech"] == tech, id_col]
        rows = marginal_rows & cost_table[id_col].isin(ids)
        if not rows.any():
            continue

        heat_rate = heat_rates.loc[tech]
        # fuel share of the EXISTING marginal_cost, estimated at reference-data's own
        # (unperturbed) central price -- this is what gets held out as "residual" below. The
        # perturbed fuel cost (which the multiplier/trajectory actually act on) is separate, so
        # the sampled start_multiplier visibly shifts the start-year total too, not just future
        # years -- at start_year the trajectory ratio is 1.0 by construction, so if the
        # perturbed and unperturbed fuel costs were the same value, start_multiplier would
        # cancel out of the total entirely.
        fuel_cost_estimate = heat_rate * float(fuel_ranges.loc[fuel, "central_value"])
        perturbed_fuel_cost_start = fuel_cost_estimate * start_mult

        start_col = next(c for c in year_cols if int(c) == start_year)
        orig_start = cost_table.loc[rows, start_col]
        residual = (orig_start - fuel_cost_estimate).clip(lower=0)

        for year_col in year_cols:
            ratio = curve.loc[int(year_col)] / fraction_at_start
            cost_table.loc[rows, year_col] = residual + perturbed_fuel_cost_start * ratio

    return cost_table, {fuel: {"start_multiplier": v["start_multiplier"], "trajectory_scenario": v["trajectory_scenario"]} for fuel, v in sampled.items()}


@dataclass
class MarginalCostPerturbation(Perturbation):
    reference_dir: Path = REFERENCE_DIR
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        costs_df, fuel_prices, fuel_decline = load_fuel_tables(self.reference_dir)
        years = sorted(int(y) for y in out.years["year"])
        start_year = years[0]

        out.generator_costs, gen_sampled = perturb_marginal_cost(
            out.generator_costs, out.generator_constants, "generator", costs_df, fuel_prices, fuel_decline, years, start_year, rng
        )
        self.sampled_ = {"generator": gen_sampled}
        return out

    def params(self):
        return {"sampled": self.sampled_}
