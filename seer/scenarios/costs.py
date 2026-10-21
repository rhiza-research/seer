"""Sample and resample cost trajectories (generator_costs.csv, storage_costs.csv) from reference
tables in reference_data/cost_assumptions (built by scripts/build_cost_ranges.py and
scripts/build_fuel_prices.py).

Each reference is a low / mid / high trajectory over the modeled years: central_value x the level's
multiplier (1 for mid) x the level's decline curve, relative to the first modeled year.
"""
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import sample_value

REFERENCE_DIR = Path(__file__).parent / "reference_data" / "cost_assumptions"
BATTERY_HOURS = 4.0  # costs.csv's battery storage capex is $/kW of a 4-hour system (NREL ATB)


@cache
def load_table(name):
    return pd.read_csv(REFERENCE_DIR / name)


def reference_trajectory(row, decline, key_col, value_col, years):
    """{"low", "mid", "high"}: arrays over years, for one costs.csv / fuel_prices.csv row."""
    ref = {}
    for level, mult in [("low", row["low_mult"]), ("mid", 1.0), ("high", row["high_mult"])]:
        curve = decline[(decline[key_col] == row[key_col]) & (decline["scenario"] == level)].set_index("year")[value_col]
        if curve.empty:
            curve = pd.Series(1.0, index=years)  # no decline curve: flat
        curve = curve.reindex(sorted(set(curve.index) | set(years))).interpolate(method="index").ffill().bfill().loc[years]
        ref[level] = row["central_value"] * mult * (curve / curve.iloc[0]).to_numpy()
    return ref


def year_columns(df):
    return [c for c in df.columns if str(c).isdigit()]


class CapexSampler:
    """Sample build options' capital cost trajectories ($/MW, or $/MWh for storage)."""

    to_perturb = [("generator_costs", "generator_constants", "generator"), ("storage_costs", "storage_constants", "storage")]

    def reference(self, tech, years, per_mwh=False):
        capex = load_table("costs.csv")
        rows = capex[(capex["tech"] == tech) & (capex["parameter"] == "capex") & (capex["unit"] == "$/kW")]
        if rows.empty:
            return None
        ref = reference_trajectory(rows.iloc[0], load_table("decline_curves.csv"), "tech", "capex_fraction", years)
        return {level: v * 1000 / (BATTERY_HOURS if per_mwh else 1) for level, v in ref.items()}  # $/kW -> $/MW (or $/MWh)

    def sample(self, level, tech, years, rng, scale=1.0, per_mwh=False):
        """Capital cost trajectory for a new build option of tech (random draws around mid)."""
        ref = self.reference(tech, years, per_mwh)
        return sample_value(level, ref["mid"], ref, rng, scale)

    def resample(self, seer_instance, rng, level, scale=1.0):
        """Return a copy of seer_instance with every build option's capital cost trajectory resampled
        (random draws around the instance's own trajectory). Existing units' capital (0) is untouched."""
        out = seer_instance.copy()
        for costs_name, constants_name, id_col in self.to_perturb:
            df = getattr(out, costs_name)
            constants = getattr(out, constants_name).set_index(id_col)
            year_cols = year_columns(df)
            years = [int(c) for c in year_cols]
            for i, row in df[df["cost_type"] == "capital"].iterrows():
                unit = constants.loc[row[id_col]]
                if not unit["expandable"]:
                    continue
                ref = self.reference(unit["tech"], years, per_mwh=costs_name == "storage_costs")
                if ref is None:
                    warnings.warn(f"no $/kW capex reference for tech {unit['tech']!r} -- leaving {row[id_col]!r} unchanged")
                    continue
                df.loc[i, year_cols] = sample_value(level, row[year_cols].to_numpy(dtype=float), ref, rng, scale)
            setattr(out, costs_name, df)
        return out


class MarginalSampler:
    """Sample thermal generators' marginal cost trajectories ($/MWh): heat_rate x fuel price + vom."""

    to_perturb = "generator_costs"

    def reference(self, tech, fuel, years):
        capex = load_table("costs.csv").set_index(["tech", "parameter"])
        prices = load_table("fuel_prices.csv")
        price_rows = prices[(prices["fuel"] == fuel) & (prices["parameter"] == "price")]
        if (tech, "heat_rate") not in capex.index or price_rows.empty:
            return None
        heat_rate = capex.loc[(tech, "heat_rate"), "central_value"]
        vom = capex.loc[(tech, "vom"), "central_value"] if (tech, "vom") in capex.index and capex.loc[(tech, "vom"), "unit"] == "$/MWh" else 0.0
        price = reference_trajectory(price_rows.iloc[0], load_table("decline_curves_fuel.csv"), "fuel", "price_fraction", years)
        return {level: heat_rate * p + vom for level, p in price.items()}

    def sample(self, level, tech, fuel, years, rng, scale=1.0):
        """Marginal cost trajectory for tech burning fuel (random draws around mid)."""
        ref = self.reference(tech, fuel, years)
        return sample_value(level, ref["mid"], ref, rng, scale)

    def resample(self, seer_instance, rng, level, scale=1.0):
        """Return a copy of seer_instance with every thermal generator's marginal cost trajectory
        resampled (random draws around the instance's own trajectory). Generators with no heat rate
        or fuel price reference (renewables, hydro, ...) are untouched."""
        out = seer_instance.copy()
        df = getattr(out, self.to_perturb)
        constants = out.generator_constants.set_index("generator")
        year_cols = year_columns(df)
        years = [int(c) for c in year_cols]
        for i, row in df[df["cost_type"] == "marginal"].iterrows():
            unit = constants.loc[row["generator"]]
            ref = self.reference(unit["tech"], unit["fuel"], years)
            if ref is None:
                continue
            df.loc[i, year_cols] = sample_value(level, row[year_cols].to_numpy(dtype=float), ref, rng, scale)
        setattr(out, self.to_perturb, df)
        return out
