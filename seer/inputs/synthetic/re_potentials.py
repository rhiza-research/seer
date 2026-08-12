"""Generates the seer schema file `renewable_profiles.csv`.

The "seed" is `seed_data/re_seed.csv`: a full annual capacity-factor curve per tech (tech,
month, day, hour, value - same layout as renewable_profiles.csv, minus the zone column), so
it can be edited without touching code. Each zone gets its own noised copy of this shared
curve, rather than an identical copy.
"""

import pandas as pd

from .noise import noise_multiplier


def build_renewable_profiles(zones, noise_pct, techs=None, seed_path="seed_data/re_seed.csv"):
    """Build renewable_profiles: an annual hourly capacity-factor curve per (zone, tech), each
    a noised copy of the seed curve for that tech.

    One multiplicative noise draw per (zone, tech), applied across that tech's whole curve.
    Result is clipped to [0, 1] (capacity factors can't go negative or above nameplate).
    """
    seed = pd.read_csv(seed_path)
    if techs is None:
        techs = seed["tech"].unique().tolist()
    seed = seed[seed["tech"].isin(techs)]

    frames = []
    for zone in zones:
        for tech, tech_curve in seed.groupby("tech"):
            multiplier = noise_multiplier(noise_pct)
            frames.append(tech_curve.assign(
                zone=zone,
                value=(tech_curve["value"] * multiplier).clip(0, 1).round(6),
            ))

    return pd.concat(frames, ignore_index=True)[["zone", "tech", "month", "day", "hour", "value"]]
