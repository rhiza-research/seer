"""Sample and resample each zone's annual peak demand trajectory, by its growth rate.

Parameter targets: demand_growth
Seer schema files: demand_forecast.csv
"""
import numpy as np
import pandas as pd


def sample(first_peak, years, scenario_options):
    """Peak trajectory from first_peak (the peak in years[0]), growing at a rate per scenario_options."""
    growth_rate = scenario_options.sample("demand_growth")
    return pd.Series({y: first_peak * (1 + growth_rate) ** (y - years[0]) for y in years})


def resample(instance, scenario_options):
    """Copy of instance with each zone's peak trajectory regrown at a rate per scenario_options (from the
    zone's own average rate), keeping the trajectory's shape."""
    out = instance.copy()
    if not scenario_options.active("demand_growth"):
        return out
    df = out.demand_forecast
    for _, group in df.groupby(["zone", "type"]):
        group = group.sort_values("year")
        years = group["year"] - group["year"].iloc[0]
        if years.iloc[-1] == 0:
            continue  # a single year has no growth to change

        # Step 1 - the trajectory's own average growth rate, and the new one
        growth_rate = (group["value"].iloc[-1] / group["value"].iloc[0]) ** (1 / years.iloc[-1]) - 1
        new_growth_rate = scenario_options.sample("demand_growth", initial=growth_rate)

        # Step 2 - regrow, scaling the trajectory's own shape (a flat one has none: grow it smoothly)
        ratio = group["value"] / group["value"].iloc[0]
        if abs(growth_rate) < 1e-9:
            ratio = (1 + new_growth_rate) ** years
        else:
            ratio = ratio ** (np.log(1 + new_growth_rate) / np.log(1 + growth_rate))
        df.loc[group.index, "value"] = group["value"].iloc[0] * ratio
    return out
