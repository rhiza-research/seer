"""Sample and resample demand_forecast.csv's annual peak trajectory, by its growth rate. Parameters:
demand_growth.
"""
import numpy as np
import pandas as pd


def sample(peak, years, options, rng=None):
    """Peak trajectory from peak in years[0], growing at a rate per options."""
    growth_rate = options.sample("demand_growth", rng=rng)
    return pd.Series({y: peak * (1 + growth_rate) ** (y - years[0]) for y in years})


def resample(instance, options, rng=None):
    """Copy of instance with each zone's peak trajectory regrown at a rate per options (from the
    zone's own average rate), keeping the trajectory's shape."""
    out = instance.copy()
    if not options.active("demand_growth"):
        return out
    df = out.demand_forecast
    for _, group in df.groupby(["zone", "type"]):
        group = group.sort_values("year")
        years = group["year"] - group["year"].iloc[0]
        if years.iloc[-1] == 0:
            continue  # a single year has no growth to change

        # Step 1 - the trajectory's own average growth rate, and the new one
        growth_rate = (group["value"].iloc[-1] / group["value"].iloc[0]) ** (1 / years.iloc[-1]) - 1
        new_growth_rate = options.sample("demand_growth", initial=growth_rate, rng=rng)

        # Step 2 - regrow, scaling the trajectory's own shape (a flat one has none: grow it smoothly)
        ratio = group["value"] / group["value"].iloc[0]
        if abs(growth_rate) < 1e-9:
            ratio = (1 + new_growth_rate) ** years
        else:
            ratio = ratio ** (np.log(1 + new_growth_rate) / np.log(1 + growth_rate))
        df.loc[group.index, "value"] = group["value"].iloc[0] * ratio
    return out
