"""Sample and resample daily demand and renewable (solar, wind, hydro) profiles.

Parameter targets: demand_profile, re_profile (item: solar / wind), hydro
Seer schema files: demand_profile.csv, renewable_profiles.csv
"""
import warnings
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.utils.time_utils import parse_time_string
from .scenario_parameters import SCENARIO_DATA, get_parameter_levels, sample_value

HOURS = np.arange(24.0)
HYDRO_TECHS = ["hydro", "ror"]


# --- demand profile: daily shape by load factor (mean / max) and peak hour ---


def sample_demand_profile(times, scenario_options):
    """One zone's demand profile over times: a library day per representative day, per scenario_options."""
    start = {"load_factor": SCENARIO_DATA["demand_profile"]["mid"], "peak_hour": float(SCENARIO_DATA["demand_profile"]["peak_hour"])}
    return np.concatenate([LOAD.sample(demand_profile_target(start, scenario_options), HOURS) for _ in range(len(times) // len(HOURS))])


def resample_demand_profile(instance, scenario_options):
    """Copy of instance with every demand day re-matched to target features per scenario_options, from its own."""
    if not scenario_options.active("demand_profile"):
        return instance.copy()
    return LOAD.resample(instance, lambda day, features: demand_profile_target(features, scenario_options))


def demand_profile_target(features, scenario_options):
    """features set to scenario_options' level or shape, then perturbed: load factor within its levels, peak
    hour shifted by up to perturbation x peak_hour_shift."""
    data, level, scale = SCENARIO_DATA["demand_profile"], scenario_options.level("demand_profile"), scenario_options.scale("demand_profile")
    if level in data["shapes"]:
        features, level = {k: float(v) for k, v in data["shapes"][level].items()}, None
    load_factor = float(np.clip(sample_value("demand_profile", features["load_factor"], level, scale, rng=scenario_options.rng), 0.01, 1.0))
    shift = scale * data["peak_hour_shift"] * scenario_options.rng.uniform(-1, 1) if scale else 0.0
    return {"load_factor": load_factor, "peak_hour": (features["peak_hour"] + shift) % 24}


# --- renewable profiles: solar / wind days by capacity factor ---


def sample_re_profile(capacity_factor, category, times, scenario_options):
    """A solar / wind profile over times at an annual capacity factor set / perturbed per scenario_options
    (from capacity_factor): one library day, repeated."""
    if get_parameter_levels("re_profile", category) is not None:
        capacity_factor = float(np.clip(scenario_options.sample("re_profile", initial=capacity_factor, item=category), 0.001, 0.999))
    return np.tile(RE.sample({"capacity_factor": capacity_factor}, HOURS, category), len(times) // len(HOURS))


def resample_re_profile(instance, scenario_options):
    """Copy of instance with each solar / wind profile's annual capacity factor set / perturbed per
    scenario_options, every day moving by the same ratio and re-matched to the library."""
    if not scenario_options.active("re_profile"):
        return instance.copy()

    # Step 1 - each profile's annual capacity factor, and the ratio to its new one
    profiles = instance.renewable_profiles.groupby("profile_id").agg(cf=("value", "mean"), tech=("tech", "first"))
    ratio = {}
    for pid, p in profiles.iterrows():
        category = renewable_category(p.tech)
        if p.cf > 0 and get_parameter_levels("re_profile", category) is not None:
            ratio[pid] = float(np.clip(scenario_options.sample("re_profile", initial=p.cf, item=category), 0.001, 0.999)) / p.cf

    # Step 2 - move every day by its profile's ratio
    return RE.resample(instance, lambda day, features: {
        "capacity_factor": float(np.clip(features["capacity_factor"] * ratio.get(day["profile_id"], 1.0), 0.001, 0.999))})


def renewable_category(tech):
    """solar/wind classification by tech-name substring; None (e.g. geothermal) has no shape library."""
    t = str(tech).lower()
    if "solar" in t or "pv" in t or "csp" in t:  # CSP grouped with solar -- diurnal shape is still sun-driven
        return "solar"
    if "wind" in t:
        return "wind"
    return None


# --- hydro: flat availability at a capacity factor, one draw for every hydro and ror profile ---


def sample_hydro(scenario_options):
    """Hydro / ror annual capacity factor per scenario_options."""
    return float(scenario_options.sample("hydro"))


def resample_hydro(instance, scenario_options):
    """Copy of instance with hydro / ror profiles rescaled to a capacity factor per scenario_options (from their
    current mean), keeping each profile's shape."""
    out = instance.copy()
    if not scenario_options.active("hydro"):
        return out
    gens, df = out.generator_constants, out.renewable_profiles
    rows = df["profile_id"].isin(gens.loc[gens["tech"].isin(HYDRO_TECHS) & gens["isre"].astype(bool), "profile_id"])
    if not rows.any():
        warnings.warn("no hydro / ror generators with a profile -- hydro left unchanged")
        return out
    initial = df.loc[rows, "value"].mean()
    df.loc[rows, "value"] = (df.loc[rows, "value"] * scenario_options.sample("hydro", initial=initial) / initial).clip(upper=1.0)
    return out


# --- library matching pipeline, shared by demand and solar / wind ---


class ProfileSampler:
    """Match target features to real days in a profile library."""

    to_perturb: str  # SeerInstance attribute
    FILTER_COL: str | None = None  # library column restricting candidates, via filter_value

    @cached_property
    def library(self):
        return load_profile_library(self.library_dir)

    @cached_property
    def library_features(self):
        return pd.DataFrame([self.extract_features(t, v) for t, v in zip(self.library["times"], self.library["values"])], index=self.library.index)

    def sample(self, features, times, filter_value=None):
        """The library profile matching features at times (hours of day); None if nothing matches filter_value."""
        library = self.library if self.FILTER_COL is None else self.library[self.library[self.FILTER_COL] == filter_value]
        if library.empty:
            return None
        dist = self.feature_distance(features, self.library_features.loc[library.index])
        nearest = dist.nsmallest(self.k).index
        profiles = [time_align(library.loc[i, "times"], library.loc[i, "values"], times) for i in nearest]
        if self.k == 1:
            return profiles[0]
        weights = 1.0 / (dist.loc[nearest].to_numpy() + 1e-6)
        return np.average(profiles, axis=0, weights=weights / weights.sum())

    def resample(self, instance, day_target):
        """Copy of instance with every day re-matched to day_target(day, day features)."""
        out = instance.copy()
        df = getattr(out, self.to_perturb)
        for day, times, values, day_rows in self.prepare_profiles(df):
            matched = self.sample(day_target(day, self.extract_features(times, values)), times, self.filter_value(day_rows))
            if matched is not None:
                df.loc[day_rows.index, "value"] = self.rescale(matched, times, values)
        return out

    def filter_value(self, day_rows):
        return None

    def rescale(self, profile, times, values):
        return profile


@dataclass
class LoadProfileSampler(ProfileSampler):
    to_perturb = "demand_profile"
    library_dir: Path = Path(__file__).parent / "reference_data" / "profiles" / "demand"
    k: int = 5  # blends the k nearest real days

    def prepare_profiles(self, df):
        return split_into_days(df, [c for c in ["zone", "year"] if c in df.columns])

    def extract_features(self, times, values):
        return {"load_factor": float(values.mean() / values.max()), "peak_hour": float(times[np.argmax(values)])}

    def feature_distance(self, target, library_features):
        hour_diff = library_features["peak_hour"].apply(lambda h: min(abs(h - target["peak_hour"]) % 24, 24 - abs(h - target["peak_hour"]) % 24))
        return ((library_features["load_factor"] - target["load_factor"]) ** 2 + (hour_diff / 12.0) ** 2) ** 0.5

    def rescale(self, profile, times, values):
        return profile * values.max()  # library days are peak-normalized; restore the day's own peak


@dataclass
class ReProfileSampler(ProfileSampler):
    to_perturb = "renewable_profiles"
    FILTER_COL = "category"
    library_dir: Path = Path(__file__).parent / "reference_data" / "profiles" / "renewable"
    k: int = 1  # blending could push an hourly value above 1

    def sample(self, features, times, filter_value=None):
        profile = super().sample(features, times, filter_value)
        return None if profile is None else np.clip(profile, 0.0, 1.0)  # some library days exceed 1 (e.g. CSP)

    def prepare_profiles(self, df):
        return split_into_days(df, ["profile_id"])

    def extract_features(self, times, values):
        return {"capacity_factor": float(np.mean(values))}

    def feature_distance(self, target, library_features):
        return (library_features["capacity_factor"] - target["capacity_factor"]).abs()

    def filter_value(self, day_rows):
        return renewable_category(day_rows["tech"].iloc[0])


LOAD, RE = LoadProfileSampler(), ReProfileSampler()


# --- utilities ---


def split_into_days(df, group_cols, value_col="value", min_points=2):
    """Yield (day, times, values, day_rows) per represented day -- day: {group_cols..., month, day} -- grouping by
    group_cols + parsed (month, day)."""
    parsed = [parse_time_string(t) for t in df["time"]]
    df = df.copy()
    df["_month"] = [p[0] for p in parsed]
    df["_day"] = [p[1] for p in parsed]
    df["_time"] = [p[2] + p[3] / 60 + p[4] / 3600 for p in parsed]  # fractional hour-of-day, keeps sub-hourly data distinct

    for group, day_rows in df.groupby(group_cols + ["_month", "_day"], dropna=False):
        day = dict(zip(group_cols + ["month", "day"], group if isinstance(group, tuple) else (group,)))
        times = day_rows["_time"].to_numpy(dtype=float)
        if (times < 0).any() or (times >= 24).any():
            warnings.warn(f"day {day!r} has hour values outside [0, 24) -- skipping.")
            continue
        if day_rows["_time"].duplicated().any() or len(day_rows) < min_points:
            continue  # not a genuine single day
        yield day, times, day_rows[value_col].to_numpy(dtype=float), day_rows


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
