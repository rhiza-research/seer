"""Sample profiles: `sample` matches one library profile to target features (for creating an
instance); `resample` sets / perturbs an existing instance's profiles per its ScenarioOptions targets
(demand-profile, re-profile, hydro), day by day from the library, or rescaled for hydro."""
import warnings
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.utils.time_utils import parse_time_string
from .costs import tech_parameter
from .scenario_options import SCENARIO_DATA, sample_value


class ProfileSampler:
    """Shared sample / resample pipeline over a library of real daily profiles."""

    to_perturb: str  # SeerInstance attribute resample perturbs
    target: str  # ScenarioOptions target
    FILTER_COL: str | None = None  # library column restricting candidates, via filter_value

    @cached_property
    def library(self):
        return load_profile_library(self.library_dir)

    @cached_property
    def library_features(self):
        return pd.DataFrame(
            [self.extract_features(t, v) for t, v in zip(self.library["times"], self.library["values"])], index=self.library.index
        )

    def sample(self, features, times, filter_value=None):
        """Return the library profile matching target features, evaluated at times (fractional
        hours of day). Returns None if no library profile matches filter_value. Not rescaled: a
        load profile comes back peak-normalized, a renewable profile as a capacity factor."""
        library_profiles = self.library if self.FILTER_COL is None else self.library[self.library[self.FILTER_COL] == filter_value]
        if library_profiles.empty:
            return None
        # distance between target and library features
        dist = self.feature_distance(features, self.library_features.loc[library_profiles.index])
        nearest = dist.nsmallest(self.k).index
        profiles = [time_align(library_profiles.loc[i, "times"], library_profiles.loc[i, "values"], times) for i in nearest]
        if self.k == 1:
            return profiles[0]  # deterministic: the single nearest real profile
        weights = 1.0 / (dist.loc[nearest].to_numpy() + 1e-6)
        return np.average(profiles, axis=0, weights=weights / weights.sum())

    def resample(self, seer_instance, options, rng):
        """Return a copy of seer_instance with its profiles set / perturbed per options' target."""
        out = seer_instance.copy()
        if not options.active(self.target):
            return out
        df = getattr(out, self.to_perturb)  # the seer instance attribute to perturb
        day_target = self.day_targets(df, options, rng)

        for key, times, values, day_rows in self.prepare_profiles(df):
            # Step 1: compute this day's own features
            features = self.extract_features(times, values)

            # Step 2: the day's target features, per options
            target_features = day_target(key, features)

            # Step 3: sample the target's profile from the library, restricted to rows matching this
            # day (e.g. renewable's solar/wind category; none for demand)
            matched = self.sample(target_features, times, self.filter_value(day_rows))
            if matched is None:
                continue  # e.g. renewable tech with no matching category library (geothermal etc.) -- leave unperturbed

            # Step 4: rescale (identity by default) and write back
            df.loc[day_rows.index, "value"] = self.rescale(matched, times, values)

        setattr(out, self.to_perturb, df)
        return out

    def day_targets(self, df, options, rng):
        """(day key, day features) -> target features, for every day in df."""
        raise NotImplementedError

    def prepare_profiles(self, df):
        """Yield (key, times, values, day_rows) per represented day in df -- see split_into_days below."""
        raise NotImplementedError

    def extract_features(self, times, values) -> dict:
        """Compute features for a profile."""
        raise NotImplementedError

    def feature_distance(self, target_features, library_features) -> pd.Series:
        """Distance from target_features to each row of library_features"""
        raise NotImplementedError

    def filter_value(self, day_rows):
        return None

    def rescale(self, profile, times, values):
        return profile


# --- demand: load-curve shape, parametrized by (load_factor, peak_hour) ---


@dataclass
class LoadProfileSampler(ProfileSampler):
    """Sample demand_profile.csv's daily shape (load factor = mean / max, and peak hour); resample
    rescales to each day's own peak. Target demand-profile: low / mid / high set the load factor
    (high = peakiest), a named shape sets both features, and the scale perturbs both."""

    to_perturb = "demand_profile"
    target = "demand-profile"
    data = SCENARIO_DATA["demand_profile"]

    library_dir: Path = Path(__file__).parent / "reference_data" / "profiles" / "demand"
    k: int = 5  # >1 blends the k nearest real profiles; 1 uses the single nearest as-is

    def start_features(self):
        """create's starting features: mid load factor at the reference peak hour."""
        return {"load_factor": self.data["load_factor"]["mid"], "peak_hour": float(self.data["peak_hour"])}

    def target_features(self, features, options, rng):
        """features set to options' level (or named shape) and perturbed by its scale."""
        level, scale = options.level(self.target), options.scale(self.target)
        if level in self.data["shapes"]:
            features, level = {k: float(v) for k, v in self.data["shapes"][level].items()}, None
        load_factor = float(np.clip(sample_value(features["load_factor"], self.data["load_factor"], rng, level, scale), 0.01, 1.0))
        shift = scale * self.data["peak_hour_shift"] * rng.uniform(-1, 1) if scale else 0.0
        return {"load_factor": load_factor, "peak_hour": (features["peak_hour"] + shift) % 24}

    def day_targets(self, df, options, rng):
        return lambda key, features: self.target_features(features, options, rng)  # a fresh draw per day

    def prepare_profiles(self, df):
        """demand_profile.csv: zone, time, value[, year] -- one day per (zone[, year])."""
        group_cols = [c for c in ["zone", "year"] if c in df.columns]
        return split_into_days(df, group_cols)

    def extract_features(self, times, values):
        return {"load_factor": float(values.mean() / values.max()), "peak_hour": float(times[np.argmax(values)])}

    @staticmethod
    def _hour_distance(h1, h2):
        """Circular distance between two hours-of-day, wrapping at 24 (so 23 and 1 are 2 apart)."""
        d = abs(h1 - h2) % 24
        return min(d, 24 - d)

    def feature_distance(self, target_features, library_features):
        load_factor_diff = library_features["load_factor"] - target_features["load_factor"]
        peak_hour_diff = library_features["peak_hour"].apply(lambda h: self._hour_distance(h, target_features["peak_hour"]))
        peak_hour_diff = peak_hour_diff / 12.0  # scale by its max possible value so it's comparable to load_factor_diff
        return (load_factor_diff ** 2 + peak_hour_diff ** 2) ** 0.5

    def rescale(self, profile, times, values):
        return profile * values.max()  # library profile is peak-normalized; restore the day's own peak level


# --- renewable: solar/wind output shape, parametrized by capacity_factor alone ---


def renewable_category(tech):
    """solar/wind classification by tech-name substring; None (e.g. geothermal) has no shape library."""
    t = str(tech).lower()
    if "solar" in t or "pv" in t or "csp" in t:  # CSP grouped with solar -- diurnal shape is still sun-driven
        return "solar"
    if "wind" in t:
        return "wind"
    return None


@dataclass
class ReProfileSampler(ProfileSampler):
    """Sample renewable_profiles.csv's daily capacity-factor shape; filter_value is the solar/wind
    category. Target re-profile: level and scale set each profile's annual capacity factor (per
    category, scenario_data.yaml's renewable_profiles), and every day moves by the same ratio."""

    to_perturb = "renewable_profiles"
    target = "re-profile"
    FILTER_COL = "category"
    data = SCENARIO_DATA["renewable_profiles"]

    library_dir: Path = Path(__file__).parent / "reference_data" / "profiles" / "renewable"
    k: int = 1  # blending real profiles could push an hourly value above 1 (physically invalid)

    def sample(self, features, times, filter_value=None):
        profile = super().sample(features, times, filter_value)
        return None if profile is None else np.clip(profile, 0.0, 1.0)  # some library days exceed 1 (e.g. CSP)

    def annual_capacity_factor(self, cf, category, options, rng):
        """An annual capacity factor set / perturbed per options (unchanged for a category without a reference)."""
        if category not in self.data:
            return cf
        return float(np.clip(sample_value(cf, self.data[category], rng, options.level(self.target), options.scale(self.target)), 0.001, 0.999))

    def day_targets(self, df, options, rng):
        profiles = df.groupby("profile_id").agg(cf=("value", "mean"), tech=("tech", "first"))
        ratio = {pid: self.annual_capacity_factor(p.cf, renewable_category(p.tech), options, rng) / p.cf if p.cf > 0 else 1.0
                 for pid, p in profiles.iterrows()}
        return lambda key, features: {"capacity_factor": float(np.clip(features["capacity_factor"] * ratio[key["profile_id"]], 0.001, 0.999))}

    def prepare_profiles(self, df):
        """renewable_profiles.csv: profile_id, tech, zone, time, value -- one day per profile_id."""
        return split_into_days(df, ["profile_id"])

    def extract_features(self, times, values):
        return {"capacity_factor": float(np.mean(values))}

    def feature_distance(self, target_features, library_features):
        return (library_features["capacity_factor"] - target_features["capacity_factor"]).abs()

    def filter_value(self, day_rows):
        return renewable_category(day_rows["tech"].iloc[0])


# --- low-level utility shared by both prepare_profiles implementations above ---


def split_into_days(df, group_cols, value_col="value", min_points=2):
    """Yield (key, times, values, day_rows) per represented day: group by group_cols + parsed (month, day)."""
    parsed = [parse_time_string(t) for t in df["time"]]
    df = df.copy()
    df["_month"] = [p[0] for p in parsed]
    df["_day"] = [p[1] for p in parsed]
    df["_time"] = [p[2] + p[3] / 60 + p[4] / 3600 for p in parsed]  # fractional hour-of-day, keeps sub-hourly data distinct

    for raw_key, day_rows in df.groupby(group_cols + ["_month", "_day"], dropna=False):
        key = dict(zip(group_cols + ["month", "day"], raw_key if isinstance(raw_key, tuple) else (raw_key,)))
        times = day_rows["_time"].to_numpy(dtype=float)
        if (times < 0).any() or (times >= 24).any():
            warnings.warn(f"day {key!r} has hour values outside [0, 24) -- skipping.")
            continue
        if day_rows["_time"].duplicated().any() or len(day_rows) < min_points:
            continue  # not a genuine single day
        yield key, times, day_rows[value_col].to_numpy(dtype=float), day_rows


def time_align(times, values, target_times):
    """Interpolate a daily profile onto target_times, wrapping across midnight (a 24h loop)."""
    order = np.argsort(times)
    times, values = times[order], values[order]
    looped_times = np.concatenate([times - 24, times, times + 24])  # three copies: wraps interpolation across midnight
    looped_values = np.concatenate([values, values, values])
    return np.interp(target_times, looped_times, looped_values)


@cache
def load_profile_library(library_dir):
    """Reshape a library's (metadata.csv, curves.csv) into one DataFrame with times/values array columns."""
    library_dir = Path(library_dir)
    metadata = pd.read_csv(library_dir / "daily_profile_metadata.csv").set_index("profile_id")
    profiles = pd.read_csv(library_dir / "daily_profile_curves.csv").groupby("profile_id")
    metadata["times"] = profiles["hour"].apply(lambda s: s.to_numpy(dtype=float))
    metadata["values"] = profiles["value"].apply(lambda s: s.to_numpy(dtype=float))
    return metadata.reset_index()


# --- hydro: flat availability at an annual capacity factor ---


class HydroProfileSampler:
    """Sample hydro and ror's availability (target hydro): a flat profile at an annual capacity factor,
    with low / mid / high from technology/tech_parameters.csv (cf_low / cf_mid / cf_high). One draw is
    shared by every hydro and ror profile."""

    to_perturb = "renewable_profiles"
    target = "hydro"
    TECHS = ["hydro", "ror"]

    def levels(self):
        return {level: tech_parameter("hydro", f"cf_{level}") for level in ["low", "mid", "high"]}

    def sample(self, options, rng):
        """Annual capacity factor per options, starting from mid."""
        levels = self.levels()
        return float(sample_value(levels["mid"], levels, rng, options.level(self.target), options.scale(self.target)))

    def resample(self, seer_instance, options, rng):
        """Return a copy of seer_instance with its hydro and ror profiles rescaled to a capacity factor
        set per options (starting from their current mean), keeping each profile's shape."""
        out = seer_instance.copy()
        if not options.active(self.target):
            return out
        gens = out.generator_constants
        ids = gens.loc[gens["tech"].isin(self.TECHS) & gens["isre"].astype(bool), "profile_id"].dropna().unique()
        df = getattr(out, self.to_perturb)
        rows = df["profile_id"].isin(ids)
        if not rows.any():
            warnings.warn("no hydro / ror generators with a profile -- hydro left unchanged")
            return out
        start = df.loc[rows, "value"].mean()
        cf = sample_value(start, self.levels(), rng, options.level(self.target), options.scale(self.target))
        df.loc[rows, "value"] = (df.loc[rows, "value"] * cf / start).clip(upper=1.0)
        setattr(out, self.to_perturb, df)
        return out
