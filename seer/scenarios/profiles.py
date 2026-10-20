"""Perturb profiles by resampling stochastically from profile libraries."""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Perturbation
from ..models.utils.time_utils import parse_time_string



def extract_daily_profiles(df, group_cols, value_col="value", min_points=2):
    """Yield (key, hours, values, day_rows) per represented day, grouped by group_cols + (month, day)."""
    parsed = [parse_time_string(t) for t in df["time"]]
    df = df.copy()
    df["_month"] = [p[0] for p in parsed]
    df["_day"] = [p[1] for p in parsed]
    df["_hour"] = [p[2] + p[3] / 60 + p[4] / 3600 for p in parsed]  # fractional hour, keeps sub-hourly data distinct

    for raw_key, day_rows in df.groupby(group_cols + ["_month", "_day"], dropna=False):
        key = dict(zip(group_cols + ["month", "day"], raw_key if isinstance(raw_key, tuple) else (raw_key,)))
        hours = day_rows["_hour"].to_numpy(dtype=float)
        if (hours < 0).any() or (hours >= 24).any():
            warnings.warn(f"day {key!r} has hour values outside [0, 24) -- skipping.")  # known time-encoding bug in some EPM-open data
            continue
        if day_rows["_hour"].duplicated().any() or len(day_rows) < min_points:
            continue  # not a genuine single day
        yield key, hours, day_rows[value_col].to_numpy(dtype=float), day_rows


def hour_distance(h1, h2):
    """Circular distance between two hours-of-day, wrapping at 24 (so 23 and 1 are 2 apart)."""
    d = abs(h1 - h2) % 24
    return min(d, 24 - d)


def time_align(hours, values, target_hours):
    """Regrid a daily profile (hours, values) onto target_hours, treating the day as a 24h loop
    (so interpolation wraps smoothly across midnight instead of hitting a hard edge)."""
    order = np.argsort(hours)
    hours, values = hours[order], values[order]
    looped_hours = np.concatenate([hours - 24, hours, hours + 24])  # three copies: wraps interpolation across midnight
    looped_values = np.concatenate([values, values, values])
    return np.interp(target_hours, looped_hours, looped_values)


def load_profile_library(library_dir):
    """Reshape a library's (metadata.csv, curves.csv) into one DataFrame with hours/values array columns."""
    library_dir = Path(library_dir)
    metadata = pd.read_csv(library_dir / "daily_profile_metadata.csv").set_index("profile_id")
    profiles = pd.read_csv(library_dir / "daily_profile_curves.csv").groupby("profile_id")
    metadata["hours"] = profiles["hour"].apply(lambda s: s.to_numpy(dtype=float))
    metadata["values"] = profiles["value"].apply(lambda s: s.to_numpy(dtype=float))
    return metadata.reset_index()


def _jitter_feature(value, offset, circular, bounds):
    jittered = value + offset
    if circular:
        return jittered % 24
    return float(np.clip(jittered, *bounds)) if bounds else jittered


class ProfilePerturbation(Perturbation):
    """Shared perturb pipeline; subclasses set TABLE/GROUP_COLS/FEATURES(/FILTER_COL) and
    extract_features (+ optionally filter_value/rescale) -- see LoadProfilePerturbation/ReProfilePerturbation below.

    Per represented day: (1) read its own profile, (2) compute the profile's own features -- no
    regridding needed, since features are deliberately resolution-robust summaries, not full-
    resolution data -- (3) jitter those features, (4) match the jittered target against the
    library's features (computed once per apply() call, reused across every day) to find the
    nearest (or a distance-weighted blend of the k nearest) library profile(s), (5) time-align
    only that small matched set onto this day's own hours, (6) optionally rescale."""

    TABLE: str  # SeerInstance attribute to perturb
    GROUP_COLS: list  # columns (besides the parsed day) identifying one represented day
    FEATURES: list  # [(name, circular, jitter_std, bounds | None), ...]
    FILTER_COL: str | None = None  # library column restricting candidates, via filter_value

    def extract_features(self, hours, values) -> dict:
        raise NotImplementedError

    def filter_value(self, day_rows):
        return None

    def rescale(self, profile, hours, values):
        return profile

    def _sample_jitter(self, rng):
        return {name: rng.normal(0, std * self.scale) for name, _, std, _ in self.FEATURES}

    def _match_profile(self, hours, target_features, pool, pool_features, rng):
        # feature distance in quadrature; circular features wrapped and scaled by their max possible difference (12h)
        dist = sum(
            (pool_features[name].apply(lambda h: hour_distance(h, target_features[name]) / 12.0) ** 2) if circular
            else (pool_features[name] - target_features[name]) ** 2
            for name, circular, _, _ in self.FEATURES
        ) ** 0.5
        nearest = dist.nsmallest(self.k).index
        matched = [time_align(pool.loc[i, "hours"], pool.loc[i, "values"], hours) for i in nearest]
        if self.k == 1:
            return matched[0]  # deterministic: the single nearest real profile
        weights = 1.0 / (dist.loc[nearest].to_numpy() + 1e-6)
        return np.average(matched, axis=0, weights=weights / weights.sum())  # distance-weighted blend of the k nearest

    def apply(self, instance, rng):
        out = instance.copy()
        df = getattr(out, self.TABLE)
        library = load_profile_library(self.library_dir)
        # library features computed ONCE here (each at its own native resolution), reused for every day below
        library_features = pd.DataFrame(
            [self.extract_features(h, v) for h, v in zip(library["hours"], library["values"])], index=library.index
        )
        group_cols = [c for c in self.GROUP_COLS if c in df.columns]
        shared_jitter = self._sample_jitter(rng) if self.jitter_mode == "shared" else None  # one draw reused for every day

        sampled_days = []
        for key, hours, values, day_rows in extract_daily_profiles(df, group_cols):
            # Step 1+2: this day's own profile and its own features (hours, values already read above)
            features = self.extract_features(hours, values)

            # Step 3: jitter the features to get a perturbation target
            jitter = shared_jitter if shared_jitter is not None else self._sample_jitter(rng)
            target_features = {name: _jitter_feature(features[name], jitter[name], circular, bounds) for name, circular, _, bounds in self.FEATURES}

            # Step 4: match the target against the (already-featurized) library, optionally restricted to a pool
            pool = library if self.FILTER_COL is None else library[library[self.FILTER_COL] == self.filter_value(day_rows)]
            if pool.empty:
                continue  # e.g. renewable tech with no matching category library (geothermal etc.) -- leave unperturbed
            matched = self._match_profile(hours, target_features, pool, library_features.loc[pool.index], rng)

            # Step 5+6: time-align already happened inside _match_profile; rescale (identity by default) and write back
            df.loc[day_rows.index, "value"] = self.rescale(matched, hours, values)

            sampled_days.append({**key, **features, **{f"target_{n}": v for n, v in target_features.items()}})

        setattr(out, self.TABLE, df)
        self.sampled_ = {"jitter_mode": self.jitter_mode, "days": sampled_days}
        return out

    def params(self):
        return {"scale": self.scale, "k": self.k, "jitter_mode": self.jitter_mode, "sampled": self.sampled_}


# --- demand: load-curve shape, parametrized by (peakiness, peak_hour) ---

# jitter std at scale=1.0: a "typical" perturbation moves peakiness by a few hundredths, peak_hour by an hour or two
BASE_PEAKINESS_JITTER_STD = 0.05
BASE_PEAK_HOUR_JITTER_STD = 1.5


@dataclass
class LoadProfilePerturbation(ProfilePerturbation):
    """Perturb demand_profile.csv's daily shape (peakiness + peak_hour), rescaled back to the
    day's own peak so only the shape is borrowed, not the day's magnitude."""

    TABLE = "demand_profile"
    GROUP_COLS = ["zone", "year"]
    FEATURES = [
        ("peakiness", False, BASE_PEAKINESS_JITTER_STD, (0.01, 1.0)),
        ("peak_hour", True, BASE_PEAK_HOUR_JITTER_STD, None),
    ]

    library_dir: Path = Path(__file__).parent / "reference_data" / "demand_profiles"
    scale: float = 1.0  # multiplies both features' jitter std -- 0 = unperturbed, 2 = twice the default spread
    k: int = 5  # >1 blends the k nearest real profiles; 1 uses the single nearest as-is
    jitter_mode: str = "shared"  # "shared": one draw for every day; "independent": a fresh draw per day
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def extract_features(self, hours, values):
        return {"peakiness": float(values.mean() / values.max()), "peak_hour": float(hours[np.argmax(values)])}

    def rescale(self, profile, hours, values):
        return profile * values.max()  # library profile is peak-normalized; restore the day's own peak level


# --- renewable: solar/wind output shape, parametrized by capacity_factor alone ---

# jitter std at scale=1.0: a "typical" perturbation moves capacity_factor by a few hundredths
BASE_CAPACITY_FACTOR_JITTER_STD = 0.05


def renewable_category(tech):
    """solar/wind classification by tech-name substring; None (e.g. geothermal) has no shape library."""
    t = str(tech).lower()
    if "solar" in t or "pv" in t or "csp" in t:  # CSP grouped with solar -- diurnal shape is still sun-driven
        return "solar"
    if "wind" in t:
        return "wind"
    return None


@dataclass
class ReProfilePerturbation(ProfilePerturbation):
    """Perturb renewable_profiles.csv's daily shape (capacity_factor), matched within the same
    solar/wind category; used at its own absolute scale, no rescale (capacity_factor IS the
    quantity of interest, not a peak-relative shape like demand's)."""

    TABLE = "renewable_profiles"
    GROUP_COLS = ["profile_id"]
    FEATURES = [("capacity_factor", False, BASE_CAPACITY_FACTOR_JITTER_STD, (0.001, 0.999))]
    FILTER_COL = "category"

    library_dir: Path = Path(__file__).parent / "reference_data" / "renewable_profiles"
    scale: float = 1.0  # 0 = unperturbed; larger = wider spread around each day's own capacity_factor
    k: int = 1  # blending real profiles could push an hourly value above 1 (physically invalid)
    jitter_mode: str = "independent"
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def extract_features(self, hours, values):
        return {"capacity_factor": float(np.mean(values))}

    def filter_value(self, day_rows):
        return renewable_category(day_rows["tech"].iloc[0])
