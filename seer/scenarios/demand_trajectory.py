"""Sample and resample demand_forecast.csv's annual peak trajectory, by its growth rate (low / mid /
high from reference_data/scenarios.yaml).
"""
import numpy as np
import pandas as pd

from .spec import REFERENCE, sample_value


def scale_demand_growth(demand_forecast, exponent):
    """new_value[year] = start_value * (value[year] / start_value) ** exponent, per (zone, type)
    trajectory. exponent=1 leaves it unchanged; >1 grows (or shrinks) more sharply than the
    original trajectory implied, <1 more gently -- this scales whatever growth shape the
    original data already had, rather than refitting a constant annual rate and discarding it.
    """
    demand_forecast = demand_forecast.copy()
    for _, group in demand_forecast.groupby(["zone", "type"]):
        start_year = group["year"].min()
        start_value = group.loc[group["year"] == start_year, "value"].iloc[0]
        ratio = group["value"] / start_value
        demand_forecast.loc[group.index, "value"] = start_value * ratio**exponent
    return demand_forecast


class DemandTrajectorySampler:
    """Sample demand_forecast.csv's annual peak trajectory, parametrized by its growth rate."""

    to_perturb = "demand_forecast"
    ref = REFERENCE["demand_trajectory"]["growth_rate"]

    def sample(self, level, start_value, years, rng, scale=1.0):
        """Peak trajectory from start_value in years[0], growing at a growth rate drawn at level
        (random draws around mid)."""
        growth_rate = sample_value(level, self.ref["mid"], self.ref, rng, scale)
        return pd.Series({y: start_value * (1 + growth_rate) ** (y - years[0]) for y in years})

    def resample(self, seer_instance, rng, level, scale=1.0):
        """Return a copy of seer_instance with each zone's peak trajectory regrown at a growth rate
        drawn at level (random draws around the zone's own rate), keeping the trajectory's shape."""
        out = seer_instance.copy()
        df = getattr(out, self.to_perturb)
        for _, group in df.groupby(["zone", "type"]):
            group = group.sort_values("year")
            n_years = group["year"].iloc[-1] - group["year"].iloc[0]
            # Step 1: the trajectory's own average growth rate
            growth_rate = (group["value"].iloc[-1] / group["value"].iloc[0]) ** (1 / n_years) - 1
            # Step 2: draw the new growth rate
            new_growth_rate = sample_value(level, growth_rate, self.ref, rng, scale)
            # Step 3: rescale the trajectory's growth, keeping its shape
            exponent = np.log(1 + new_growth_rate) / np.log(1 + growth_rate)
            df.loc[group.index, "value"] = scale_demand_growth(group, exponent)["value"]
        setattr(out, self.to_perturb, df)
        return out

