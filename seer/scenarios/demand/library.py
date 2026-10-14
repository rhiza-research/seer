"""Demand profile library: compute shape features (load_factor, peak_hour) on any
native-resolution daily curve, and sample/blend new curves from a library by matching those
features. Resampling to a specific output resolution (e.g. hourly) happens after sampling or
blending, not when the library is built -- so curves of any native resolution (5 points/day, 24,
half-hourly, ...) can sit in the same library and still be drawn at whatever resolution a given
model needs.

The caller is responsible for making sure every profile passed in is at a comparable scale (e.g.
not mixing individual-household traces with country-level system demand) -- this module doesn't
police that.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..daily_profiles import circular_hour_distance, resample_circular

LIBRARY_DIR = Path(__file__).parents[1] / "reference_data" / "demand_profiles"


def load_library(library_dir=LIBRARY_DIR):
    """(metadata, curves) tables written by scripts/build_demand_profile_library.py."""
    return (
        pd.read_csv(library_dir / "daily_profile_metadata.csv"),
        pd.read_csv(library_dir / "daily_profile_curves.csv"),
    )


def compute_features(hours, values):
    """(load_factor, peak_hour, peak) for one daily curve at its own native resolution.
    load_factor is mean/max (an arithmetic mean over however many points are given, not
    time-weighted); peak_hour is the hour-of-day of the curve's own maximum sample."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    peak = values.max()
    load_factor = values.mean() / peak
    peak_hour = float(hours[values.argmax()])
    return load_factor, peak_hour, peak


def feature_distance(load_factor_a, peak_hour_a, load_factor_b, peak_hour_b):
    """load_factor difference (already 0-1 scale) and circular peak-hour difference (scaled by
    /12, its max possible value) combined in quadrature, so neither feature dominates purely
    from its unit scale."""
    d_lf = load_factor_a - load_factor_b
    d_peak = circular_hour_distance(peak_hour_a, peak_hour_b) / 12.0
    return (d_lf**2 + d_peak**2) ** 0.5


def sample_profile(metadata, curves, target_load_factor, target_peak_hour, target_hours, rng, k=5, blend=True, noise_pct=0.0):
    """metadata: DataFrame [profile_id, load_factor, peak_hour]. curves: long-format DataFrame
    [profile_id, hour, value] at each profile's own native resolution. Returns an array of
    values at target_hours, peak-normalized to 1.0."""
    distances = metadata.apply(
        lambda r: feature_distance(target_load_factor, target_peak_hour, r["load_factor"], r["peak_hour"]), axis=1
    )
    nearest_idx = distances.nsmallest(k).index
    nearest_ids = metadata.loc[nearest_idx, "profile_id"]
    nearest_distances = distances.loc[nearest_idx].to_numpy()
    weights = 1.0 / (nearest_distances + 1e-6)
    weights /= weights.sum()

    resampled = [
        resample_circular(curves.loc[curves["profile_id"] == pid, "hour"], curves.loc[curves["profile_id"] == pid, "value"], target_hours)
        for pid in nearest_ids
    ]

    if blend:
        result = np.average(resampled, axis=0, weights=weights)
    else:
        result = resampled[rng.choice(len(resampled), p=weights)]

    if noise_pct:
        result = np.clip(result * (1 + rng.normal(0, noise_pct, size=result.shape)), 0, None)

    return result / result.max()
