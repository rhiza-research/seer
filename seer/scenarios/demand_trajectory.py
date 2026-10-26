"""Sample and resample demand_forecast.csv's annual peak trajectory, by its growth rate (target
demand-growth; low / mid / high from reference_data/scenario_data.yaml).
"""
import numpy as np
import pandas as pd

from .scenario_options import SCENARIO_DATA, sample_value


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
    """Sample demand_forecast.csv's annual peak trajectory, parametrized by its growth rate (target
    demand-growth)."""

    to_perturb = "demand_forecast"
    target = "demand-growth"
    levels = SCENARIO_DATA["demand_trajectory"]["growth_rate"]

    def sample(self, start_value, years, options, rng):
        """Peak trajectory from start_value in years[0], growing at a rate set by options (from mid)."""
        growth_rate = sample_value(self.levels["mid"], self.levels, rng, options.level(self.target), options.scale(self.target))
        return pd.Series({y: start_value * (1 + growth_rate) ** (y - years[0]) for y in years})

    def resample(self, seer_instance, options, rng):
        """Return a copy of seer_instance with each zone's peak trajectory regrown at a rate set by
        options (from the zone's own average rate), keeping the trajectory's shape."""
        out = seer_instance.copy()
        if not options.active(self.target):
            return out
        df = getattr(out, self.to_perturb)
        for _, group in df.groupby(["zone", "type"]):
            group = group.sort_values("year")
            n_years = group["year"].iloc[-1] - group["year"].iloc[0]
            if n_years == 0:
                continue  # a single year has no growth to change
            # Step 1: the trajectory's own average growth rate
            growth_rate = (group["value"].iloc[-1] / group["value"].iloc[0]) ** (1 / n_years) - 1
            # Step 2: draw the new growth rate
            new_growth_rate = sample_value(growth_rate, self.levels, rng, options.level(self.target), options.scale(self.target))
            # Step 3: rescale the trajectory's growth, keeping its shape -- a flat one has no shape to keep
            if abs(growth_rate) < 1e-9:
                values = group["value"].iloc[0] * (1 + new_growth_rate) ** (group["year"] - group["year"].iloc[0])
            else:
                values = scale_demand_growth(group, np.log(1 + new_growth_rate) / np.log(1 + growth_rate))["value"]
            df.loc[group.index, "value"] = values
        setattr(out, self.to_perturb, df)
        return out
