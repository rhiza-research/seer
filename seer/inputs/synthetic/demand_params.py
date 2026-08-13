"""Generates the seer schema files `demandforecast.csv` and `demandprofile.csv`.
seed: annual hourly demand shape that is perturbed per zone and time point.
"""

import pandas as pd

from .noise import noise_multiplier


def build_demand_forecast(zones, peak_demand_mw, years, growth_rate):
    """Build demandforecast: peak demand by year by zone.

    peak_demand_mw is the starting (first modeled year) peak demand per zone; growth_rate is
    each zone's already-realized annual growth rate (randomization happens in generate_data.py,
    not here - this is a deterministic projection from a given growth rate).
    """
    assert len(peak_demand_mw) == len(zones), "peak_demand_mw must have exactly one entry per zone"
    assert len(growth_rate) == len(zones), "growth_rate must have exactly one entry per zone"

    base_year = years[0]
    rows = [
        {"zone": zone, "type": "peak", "year": year, "value": round(start * (1 + growth) ** (year - base_year), 3)}
        for zone, start, growth in zip(zones, peak_demand_mw, growth_rate)
        for year in years
    ]

    return pd.DataFrame(rows)


def build_demand_profile(zones, noise_pct, seed_path="seed_data/demand_seed.csv"):
    """Build demandprofile: the seed's normalized hourly load shape, independently perturbed
    per zone and timepoint by noise_pct.
    """
    shape = pd.read_csv(seed_path)
    frames = []
    for zone in zones:
        # perturb the hourly load shape by noise
        noise = noise_multiplier(noise_pct, size=len(shape))
        frames.append(shape.assign(zone=zone, value=(shape["value"] * noise)))

    return pd.concat(frames, ignore_index=True)[["zone", "season", "day", "hour", "value"]]
