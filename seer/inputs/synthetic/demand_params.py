"""Generates the seer schema files `demandforecast.csv` and `demandprofile.csv`.
seed: annual hourly demand shape that is perturbed per zone and time point.
"""

import pandas as pd

from .noise import noise_multiplier


def build_demand(zones, peak_demand_mw, years, growth_rate_mean, noise_pct,
                  seed_path="seed_data/demand_seed.csv"):
    """Generates demand parameters:
    - demandforecast: peak demand by year by zone
    - demandprofile: normalized hourly load shape

    peak_demand_mw is the expected peak demand per zone; growth_rate_mean is the expected annual growth rate.
    both the peak demand, growth rate, and load shape are perturbed by noise.
    """
    assert len(peak_demand_mw) == len(zones), "peak_demand_mw must have exactly one entry per zone"

    # determine the annual growth rate, perturbed by noise
    growth_rate = growth_rate_mean * noise_multiplier(noise_pct, size=len(zones))

    base_year = years[0]
    forecast_rows = [
        {"zone": zone, "type": "peak", "year": year, "value": round(start * (1 + growth) ** (year - base_year), 3)}
        for zone, start, growth in zip(zones, peak_demand_mw, growth_rate)
        for year in years
    ]

    shape = pd.read_csv(seed_path)
    frames = []
    for zone in zones:
        # perturb the hourly load shape by noise
        noise = noise_multiplier(noise_pct, size=len(shape))
        frames.append(shape.assign(zone=zone, value=(shape["value"] * noise)))
    profile_df = pd.concat(frames, ignore_index=True)[["zone", "season", "day", "hour", "value"]]

    return pd.DataFrame(forecast_rows), profile_df
