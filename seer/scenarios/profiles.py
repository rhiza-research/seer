"""Daily-profile (load-curve / renewable-output shape) scenarios: perturbing or -- later --
creating a demand_profile.csv or renewable_profiles.csv from a library of real daily curves
(seer/scenarios/reference_data/{demand_profiles,renewable_profiles}), built by
scripts/build_demand_profile_library.py (and, later, an analogous renewable one).

The approach is the same for every signal: compute a small set of shape features on each
represented day, find the library day(s) whose own features are nearest a target, and use that to
replace or blend the original day's values. Perturbation jitters each day's OWN features as the
target (computed from the instance passed in); creation (not yet implemented) would instead take
target features directly from the caller -- the only difference is where the target features come
from, so both share everything below the "compute target features" step.

Shared machinery (below) is signal-agnostic: given a library (metadata + curves) and a distance
from each library row to a target, find nearest rows and resample/combine their curves onto a
given hour grid. Feature computation and distance are signal-specific and live in their own
section further down (demand; renewable, when added).
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Perturbation
from ..models.utils.time_utils import parse_time_string

# --- shared: periodic (24h) curve resampling, and library-wide nearest-neighbor lookup ---


def circular_hour_distance(h1, h2):
    d = abs(h1 - h2) % 24
    return min(d, 24 - d)


def resample_circular(hours, values, target_hours):
    """Evaluate the periodic (24h) curve defined by (hours, values) at arbitrary target_hours,
    via linear interpolation on a 3x-tiled (wrapped) copy of the curve."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(hours)
    hours, values = hours[order], values[order]
    extended_hours = np.concatenate([hours - 24, hours, hours + 24])
    extended_values = np.concatenate([values, values, values])
    return np.interp(target_hours, extended_hours, extended_values)


def load_library(library_dir):
    """(metadata, curves) tables written by a library-build script (e.g.
    scripts/build_demand_profile_library.py)."""
    return (
        pd.read_csv(Path(library_dir) / "daily_profile_metadata.csv"),
        pd.read_csv(Path(library_dir) / "daily_profile_curves.csv"),
    )


def nearest_profiles(metadata, distances, k=5):
    """profile_ids and inverse-distance sampling weights of the k library rows nearest a target,
    given a per-row `distances` Series aligned to `metadata`'s index."""
    nearest_idx = distances.nsmallest(k).index
    ids = metadata.loc[nearest_idx, "profile_id"]
    weights = 1.0 / (distances.loc[nearest_idx].to_numpy() + 1e-6)
    return ids, weights / weights.sum()


def sample_profile(curves, ids, weights, target_hours, rng, blend=True, noise_pct=0.0, normalize=True):
    """Resample each candidate profile (at its own native resolution) onto target_hours, then
    either weighted-average them (blend=True, demand's k-NN blend) or pick a single one at random
    by weight (blend=False, e.g. renewable's "use the single nearest real day as-is"). normalize
    rescales the result to peak=1 -- set False to keep a library's own absolute scale (e.g.
    renewable capacity-factor curves, which must not be rescaled)."""
    resampled = [
        resample_circular(curves.loc[curves["profile_id"] == pid, "hour"], curves.loc[curves["profile_id"] == pid, "value"], target_hours)
        for pid in ids
    ]
    result = np.average(resampled, axis=0, weights=weights) if blend else resampled[rng.choice(len(resampled), p=weights)]
    if noise_pct:
        result = np.clip(result * (1 + rng.normal(0, noise_pct, size=result.shape)), 0, None)
    return result / result.max() if normalize else result


# --- demand: load-curve shape, parametrized by (load_factor, peak_hour) ---

DEMAND_LIBRARY_DIR = Path(__file__).parent / "reference_data" / "demand_profiles"


def demand_features(hours, values):
    """(load_factor, peak_hour, peak) for one daily curve at its own native resolution.
    load_factor is mean/max (an arithmetic mean over however many points are given, not
    time-weighted); peak_hour is the hour-of-day of the curve's own maximum sample."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    peak = values.max()
    load_factor = values.mean() / peak
    peak_hour = float(hours[values.argmax()])
    return load_factor, peak_hour, peak


def demand_feature_distance(load_factor_a, peak_hour_a, load_factor_b, peak_hour_b):
    """load_factor difference (already 0-1 scale) and circular peak-hour difference (scaled by
    /12, its max possible value) combined in quadrature, so neither feature dominates purely
    from its unit scale."""
    d_lf = load_factor_a - load_factor_b
    d_peak = circular_hour_distance(peak_hour_a, peak_hour_b) / 12.0
    return (d_lf**2 + d_peak**2) ** 0.5


def sample_demand_profile(metadata, curves, target_load_factor, target_peak_hour, target_hours, rng, k=5, blend=True, noise_pct=0.0):
    """metadata: DataFrame [profile_id, load_factor, peak_hour]. curves: long-format DataFrame
    [profile_id, hour, value] at each profile's own native resolution. Returns an array of
    values at target_hours, peak-normalized to 1.0."""
    distances = metadata.apply(
        lambda r: demand_feature_distance(target_load_factor, target_peak_hour, r["load_factor"], r["peak_hour"]), axis=1
    )
    ids, weights = nearest_profiles(metadata, distances, k)
    return sample_profile(curves, ids, weights, target_hours, rng, blend=blend, noise_pct=noise_pct, normalize=True)


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
        load_factor, peak_hour, day_peak = demand_features(hours, values)

        lf_jitter, peak_jitter = shared_jitter if shared_jitter is not None else (
            rng.normal(0, load_factor_jitter_std), rng.normal(0, peak_hour_jitter_std)
        )
        target_load_factor = float(np.clip(load_factor + lf_jitter, 0.01, 1.0))
        target_peak_hour = (peak_hour + peak_jitter) % 24

        replacement = sample_demand_profile(metadata, curves, target_load_factor, target_peak_hour, hours, rng, k=k, blend=blend_neighbors, noise_pct=noise_pct)
        new_values = (1 - blend_alpha) * values + blend_alpha * replacement * day_peak
        demand_profile.loc[group.index, "value"] = new_values

        key_values = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        sampled_days.append({
            **key_values, "load_factor": load_factor, "peak_hour": peak_hour,
            "target_load_factor": target_load_factor, "target_peak_hour": float(target_peak_hour),
        })

    return demand_profile.drop(columns=["_month", "_day", "_hour"]), sampled_days


# jitter std at scale=1.0 -- chosen so a "typical" perturbation moves load_factor by a few
# hundredths and peak_hour by an hour or two; LoadCurvePerturbation.scale multiplies both
# together, so one number controls the overall perturbation size across both features at once.
BASE_LOAD_FACTOR_JITTER_STD = 0.05
BASE_PEAK_HOUR_JITTER_STD = 1.5


@dataclass
class LoadCurvePerturbation(Perturbation):
    library_dir: Path = DEMAND_LIBRARY_DIR
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


# --- renewable: solar/wind output shape, parametrized by capacity_factor alone ---
#
# Unlike demand, a renewable profile's shape is geography-pinned (sun position, local wind
# climate) rather than behaviorally transferable, so the only feature we match on is
# capacity_factor -- for solar this is essentially "how flattened the peak is" (cloud cover);
# for wind it's a weaker proxy for a genuinely more stochastic signal, but good enough to start.
# renewable_profiles.csv values are already an absolute capacity factor (not peak-relative like
# demand), so there's nothing to renormalize, and a matched library day's real values are used
# as-is -- never blended or rescaled, since forcing an exact mean match could push some hourly
# value above 1 (physically invalid) even when the mean itself is valid.

RENEWABLE_LIBRARY_DIR = Path(__file__).parent / "reference_data" / "renewable_profiles"


def renewable_category(tech):
    """solar/wind classification by tech-name substring, for grouping the renewable-profile
    library and picking a perturbation target's replacement pool. CSP is grouped with solar (its
    diurnal shape is still sun-driven, even though storage can shift its dispatch -- revisit if
    that turns out to matter for perturbation realism). Returns None (no shape library for it,
    e.g. geothermal) when neither matches."""
    t = str(tech).lower()
    if "solar" in t or "pv" in t or "csp" in t:
        return "solar"
    if "wind" in t:
        return "wind"
    return None


def renewable_features(values):
    """capacity_factor for one daily curve: the arithmetic mean of its own values (an estimate
    that gets coarser the fewer points a day has, same caveat as demand_features' load_factor).
    Already an absolute capacity factor, not peak-relative -- no peak to report or normalize by."""
    return float(np.mean(values))


def sample_renewable_profile(metadata, curves, target_capacity_factor, category, target_hours, rng):
    """metadata: DataFrame [profile_id, category, capacity_factor]. curves: long-format DataFrame
    [profile_id, hour, value] of absolute (not peak-normalized) capacity-factor values. Returns
    the single nearest-by-capacity_factor real library day of the same category, resampled onto
    target_hours and used as-is (no blending across neighbors, no rescaling)."""
    pool = metadata[metadata["category"] == category]
    distances = (pool["capacity_factor"] - target_capacity_factor).abs()
    ids, weights = nearest_profiles(pool, distances, k=1)
    return sample_profile(curves, ids, weights, target_hours, rng, blend=False, normalize=False)


# concentration at scale=1.0 -- a Beta(mu*kappa, (1-mu)*kappa) with this kappa keeps a "typical"
# perturbation within roughly a few hundredths of capacity_factor for most mu; RenewableProfile-
# Perturbation.scale divides it, so larger scale -> lower concentration -> wider spread.
BASE_CAPACITY_FACTOR_CONCENTRATION = 50.0


def sample_target_capacity_factor(capacity_factor, scale, rng):
    """Beta-distributed perturbed capacity factor centered on `capacity_factor`, bounded in (0, 1)
    by construction -- no clipping needed, unlike an unbounded additive/multiplicative jitter.
    scale <= 0 returns the original value unperturbed (Beta's concentration -> infinity in the
    limit, which rng.beta can't represent directly)."""
    if scale <= 0:
        return float(capacity_factor)
    mu = min(max(capacity_factor, 1e-3), 1 - 1e-3)
    kappa = BASE_CAPACITY_FACTOR_CONCENTRATION / scale
    return float(rng.beta(mu * kappa, (1 - mu) * kappa))


def jitter_and_resample_renewable_profile(renewable_profiles, metadata, curves, rng, scale):
    """For each represented day (grouped by profile_id, month, day parsed from the time string --
    profile_id already pins a single zone/tech/generator resource): compute its own
    capacity_factor, sample a perturbed target from a Beta distribution centered on it, and
    replace the day's values with the nearest real library day of the same category (solar/wind),
    resampled onto the SAME hours that day already used."""
    renewable_profiles = renewable_profiles.copy()
    parsed = [parse_time_string(t) for t in renewable_profiles["time"]]
    renewable_profiles["_month"] = [p[0] for p in parsed]
    renewable_profiles["_day"] = [p[1] for p in parsed]
    renewable_profiles["_hour"] = [p[2] + p[3] / 60 + p[4] / 3600 for p in parsed]

    sampled_days = []
    for (profile_id, month, day), group in renewable_profiles.groupby(["profile_id", "_month", "_day"], dropna=False):
        tech = group["tech"].iloc[0]
        category = renewable_category(tech)
        if category is None:
            continue  # no shape library for this tech (e.g. geothermal) -- leave unperturbed

        hours = group["_hour"].to_numpy(dtype=float)
        if (hours < 0).any() or (hours >= 24).any():
            warnings.warn(
                f"RenewableProfilePerturbation: profile {profile_id!r} day {month}-{day} has hour "
                f"values outside [0, 24) ({hours.min():.0f}-{hours.max():.0f}) -- leaving it unperturbed."
            )
            continue

        values = group["value"].to_numpy(dtype=float)
        capacity_factor = renewable_features(values)
        target_capacity_factor = sample_target_capacity_factor(capacity_factor, scale, rng)

        replacement = sample_renewable_profile(metadata, curves, target_capacity_factor, category, hours, rng)
        renewable_profiles.loc[group.index, "value"] = replacement

        sampled_days.append({
            "profile_id": profile_id, "tech": tech, "category": category, "month": month, "day": day,
            "capacity_factor": capacity_factor, "target_capacity_factor": target_capacity_factor,
        })

    return renewable_profiles.drop(columns=["_month", "_day", "_hour"]), sampled_days


@dataclass
class RenewableProfilePerturbation(Perturbation):
    library_dir: Path = RENEWABLE_LIBRARY_DIR
    scale: float = 1.0  # 0 = unperturbed; larger = wider spread around each day's own capacity_factor
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        metadata, curves = load_library(self.library_dir)
        out.renewable_profiles, sampled_days = jitter_and_resample_renewable_profile(
            out.renewable_profiles, metadata, curves, rng, self.scale,
        )
        self.sampled_ = {"days": sampled_days}
        return out

    def params(self):
        return {"scale": self.scale, "sampled": self.sampled_}
