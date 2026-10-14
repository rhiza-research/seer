"""Demand perturbations: growth-rate intensity (demand_forecast.csv) and load-curve shape
(demand_profile.csv). Growth is driven by the dataclass's own sampled bounds, since there's no
external source of "how much might a country's demand growth rate vary". Load-curve shape draws
from the real-profile library (library.py) instead.
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...models.time_utils import parse_time_string
from ..base import Perturbation
from .library import LIBRARY_DIR, compute_features, load_library, sample_profile


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


def jitter_and_resample_load_curve(
    demand_profile, metadata, curves, rng,
    load_factor_jitter_std, peak_hour_jitter_std, jitter_mode, blend_alpha, k, blend_neighbors, noise_pct,
):
    """For each represented day (grouped by zone[, year], month, day parsed from the time
    string): compute its own (load_factor, peak_hour), jitter them, and draw a replacement curve
    from the profile library at the SAME hours that day already used -- rescaled back to the
    day's own original peak level (demand_profile's values are fractions of the zone's *annual*
    peak, not of that day's own peak, so a different day's relative importance within the year
    must be preserved), then blended with the day's original curve by blend_alpha (1.0 = full
    replacement, 0.0 = unchanged).

    jitter_mode="shared" draws one jitter pair for every day (a whole-system shift); "independent"
    draws a fresh jitter per day (added day-to-day volatility)."""
    demand_profile = demand_profile.copy()
    parsed = [parse_time_string(t) for t in demand_profile["time"]]
    demand_profile["_month"] = [p[0] for p in parsed]
    demand_profile["_day"] = [p[1] for p in parsed]
    # fractional hour-of-day, not truncated to whole hours -- sub-hourly data (e.g. a 20-minute
    # resolution country) would otherwise collapse several distinct timesteps onto one integer
    # hour and look like duplicate/out-of-range values
    demand_profile["_hour"] = [p[2] + p[3] / 60 + p[4] / 3600 for p in parsed]

    group_cols = [c for c in ["zone", "year"] if c in demand_profile.columns] + ["_month", "_day"]

    shared_jitter = None
    if jitter_mode == "shared":
        shared_jitter = (rng.normal(0, load_factor_jitter_std), rng.normal(0, peak_hour_jitter_std))

    sampled_days = []
    for key, group in demand_profile.groupby(group_cols, dropna=False):
        hours = group["_hour"].to_numpy(dtype=float)
        if (hours < 0).any() or (hours >= 24).any():
            warnings.warn(
                f"LoadCurvePerturbation: day {dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))!r} "
                f"has hour values outside [0, 24) ({hours.min():.0f}-{hours.max():.0f}) -- not real hours-of-day "
                "(a known encoding bug for at least one EPM-open country, see Senegal). Leaving it unperturbed "
                "rather than running circular hour logic on nonsense values."
            )
            continue
        values = group["value"].to_numpy(dtype=float)
        load_factor, peak_hour, day_peak = compute_features(hours, values)

        lf_jitter, peak_jitter = shared_jitter if shared_jitter is not None else (
            rng.normal(0, load_factor_jitter_std), rng.normal(0, peak_hour_jitter_std)
        )
        target_load_factor = float(np.clip(load_factor + lf_jitter, 0.01, 1.0))
        target_peak_hour = (peak_hour + peak_jitter) % 24

        replacement = sample_profile(metadata, curves, target_load_factor, target_peak_hour, hours, rng, k=k, blend=blend_neighbors, noise_pct=noise_pct)
        new_values = (1 - blend_alpha) * values + blend_alpha * replacement * day_peak
        demand_profile.loc[group.index, "value"] = new_values

        key_values = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        sampled_days.append({
            **key_values, "load_factor": load_factor, "peak_hour": peak_hour,
            "target_load_factor": target_load_factor, "target_peak_hour": float(target_peak_hour),
        })

    return demand_profile.drop(columns=["_month", "_day", "_hour"]), sampled_days


@dataclass
class DemandGrowthPerturbation(Perturbation):
    low: float = 0.5
    high: float = 1.5
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        exponent = float(rng.uniform(self.low, self.high))
        self.sampled_ = {"growth_exponent": exponent}
        out.demand_forecast = scale_demand_growth(out.demand_forecast, exponent)
        return out

    def params(self):
        return {"sampled": self.sampled_}


# jitter std at scale=1.0 -- chosen so a "typical" perturbation moves load_factor by a few
# hundredths and peak_hour by an hour or two; LoadCurvePerturbation.scale multiplies both
# together, so one number controls the overall perturbation size across both features at once.
BASE_LOAD_FACTOR_JITTER_STD = 0.05
BASE_PEAK_HOUR_JITTER_STD = 1.5


@dataclass
class LoadCurvePerturbation(Perturbation):
    library_dir: Path = LIBRARY_DIR
    scale: float = 1.0  # multiplies both features' jitter std together -- 0 = unperturbed, 2 = twice the default spread
    jitter_mode: str = "shared"  # "shared" (one draw for every day) or "independent" (per day)
    blend_alpha: float = 1.0  # 1.0 = fully replace each day's shape, 0.0 = unchanged
    k: int = 5
    blend_neighbors: bool = True
    noise_pct: float = 0.0
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        metadata, curves = load_library(self.library_dir)

        out.demand_profile, sampled_days = jitter_and_resample_load_curve(
            out.demand_profile, metadata, curves, rng,
            self.scale * BASE_LOAD_FACTOR_JITTER_STD, self.scale * BASE_PEAK_HOUR_JITTER_STD, self.jitter_mode,
            self.blend_alpha, self.k, self.blend_neighbors, self.noise_pct,
        )
        self.sampled_ = {"jitter_mode": self.jitter_mode, "days": sampled_days}
        return out

    def params(self):
        return {
            "scale": self.scale,
            "jitter_mode": self.jitter_mode,
            "blend_alpha": self.blend_alpha,
            "sampled": self.sampled_,
        }
