"""Generates the seer schema files `demandforecast.csv` and `demandprofile.csv`.

The "seed" is `seed_data/demand_seed.csv`: an annual hourly load shape (season, day, hour,
value - same layout as demandprofile.csv, minus the zone column), so it can be edited without
touching code. Every zone shares this profile; only the peak-demand forecast varies by zone.
"""

import numpy as np
import pandas as pd


def build_demand(zones, peak_demand_mw, years, growth_rate_mean, growth_rate_std,
                  seed_path="seed_data/demand_seed.csv"):
    """Generates demand parameters:
    - demandforecast: randomized peak-demand growth per zone
    - demandprofile: normalized hourly load shape, from the seed; shared across zones and years.

    peak_demand_mw (a scenario input, one value per zone) is the starting point; growth off it
    is randomized around growth_rate_mean/growth_rate_std.
    """
    assert len(peak_demand_mw) == len(zones), "peak_demand_mw must have exactly one entry per zone"

    base_year = years[0]
    growth_rate = np.random.normal(growth_rate_mean, growth_rate_std, size=len(zones))

    forecast_rows = [
        {"zone": zone, "type": "peak", "year": year, "value": round(start * (1 + growth) ** (year - base_year), 3)}
        for zone, start, growth in zip(zones, peak_demand_mw, growth_rate)
        for year in years
    ]

    shape = pd.read_csv(seed_path)
    profile_df = pd.concat(
        [shape.assign(zone=zone) for zone in zones], ignore_index=True
    )[["zone", "season", "day", "hour", "value"]]

    return pd.DataFrame(forecast_rows), profile_df
