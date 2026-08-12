"""Generates the seer schema files `demandforecast.csv` and `demandprofile.csv`.
seed: annual hourly demand shape that is perturbed per zone and time point.
"""

import pandas as pd

from .noise import noise_multiplier


def build_demand(zones, peak_demand_mw, years, growth_rate_mean, noise_pct,
                  seed_path="seed_data/demand_seed.csv"):
    """Generates demand parameters:
    - demandforecast: peak-demand growth per zone (growth_rate_mean scaled by noise)
    - demandprofile: normalized hourly load shape from the seed, independently noised per
      zone and timepoint

    peak_demand_mw (a scenario input, one value per zone) is the starting point.
    """
    assert len(peak_demand_mw) == len(zones), "peak_demand_mw must have exactly one entry per zone"

    base_year = years[0]
    growth_rate = growth_rate_mean * noise_multiplier(noise_pct, size=len(zones))

    forecast_rows = [
        {"zone": zone, "type": "peak", "year": year, "value": round(start * (1 + growth) ** (year - base_year), 3)}
        for zone, start, growth in zip(zones, peak_demand_mw, growth_rate)
        for year in years
    ]

    shape = pd.read_csv(seed_path)
    frames = []
    for zone in zones:
        noise = noise_multiplier(noise_pct, size=len(shape))
        frames.append(shape.assign(zone=zone, value=(shape["value"] * noise).round(6)))
    profile_df = pd.concat(frames, ignore_index=True)[["zone", "season", "day", "hour", "value"]]

    return pd.DataFrame(forecast_rows), profile_df
