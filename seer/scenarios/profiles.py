"""Sample profiles from profile libraries: `sample` builds one profile from target features
(e.g. a renewable capacity factor) for creating a new instance; `resample` perturbs every day of an
existing instance's profiles by randomizing each day's features and resampling a library profile."""
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.utils.time_utils import parse_time_string


class ProfileSampler:
    """Shared sample/resample pipeline over a library of real daily profiles."""

    to_perturb: str  # SeerInstance attribute resample perturbs
    FILTER_COL: str | None = None  # library column restricting candidates, via filter_value

    @cached_property
    def library(self):
        return load_profile_library(self.library_dir)

    @cached_property
    def library_features(self):
        return pd.DataFrame(
            [self.extract_features(t, v) for t, v in zip(self.library["times"], self.library["values"])], index=self.library.index
        )

    def sample(self, features, times, filter_value=None, rng=None, scale=1.0):
        """Return the library profile matching target features, evaluated at times (fractional
        hours of day). features is a dict of feature values, a named scenario (see SCENARIOS), or
        "random" (a randomly chosen named scenario, randomized by scale). Returns None if no library
        profile matches filter_value. Not rescaled: a load profile comes back peak-normalized, a
        renewable profile as a capacity factor."""
        if features == "random":
            features = self.randomize_features(self.named_scenario(rng.choice(self.scenario_names(filter_value)), filter_value), rng, scale)
        elif isinstance(features, str):
            features = self.named_scenario(features, filter_value)
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

    def resample(self, seer_instance, rng, scale=1.0):
        """Return a copy of seer_instance with its profiles perturbed. scale: 0 = unperturbed, 1 =
        the largest shift."""
        out = seer_instance.copy()
        if scale == 0:
            return out
        df = getattr(out, self.to_perturb)  # the seer instance attribute to perturb

        for pid, times, values, day_rows in self.prepare_profiles(df):
            # Step 1: compute this day's own features
            features = self.extract_features(times, values)

            # Step 2: randomly perturb the features to get a target
            target_features = self.randomize_features(features, rng, scale)

            # Step 3: sample the target's profile from the library, restricted to rows matching this
            # day (e.g. renewable's solar/wind category; none for demand)
            matched = self.sample(target_features, times, self.filter_value(day_rows))
            if matched is None:
                continue  # e.g. renewable tech with no matching category library (geothermal etc.) -- leave unperturbed

            # Step 4: rescale (identity by default) and write back
            df.loc[day_rows.index, "value"] = self.rescale(matched, times, values)

        setattr(out, self.to_perturb, df)
        return out

    def prepare_profiles(self, df):
        """Yield (key, times, values, day_rows) per represented day in df -- see split_into_days below."""
        raise NotImplementedError

    def extract_features(self, times, values) -> dict:
        """Compute features for a profile."""
        raise NotImplementedError

    def scenario_names(self, filter_value=None):
        return list(self.SCENARIOS)

    def named_scenario(self, name, filter_value=None) -> dict:
        """Hardcoded feature values for a named scenario."""
        return self.SCENARIOS[name]

    def randomize_features(self, features, rng, scale) -> dict:
        """Shift each feature by about scale times its largest shift, in a random direction."""
        raise NotImplementedError

    def feature_distance(self, target_features, library_features) -> pd.Series:
        """Distance from target_features to each row of library_features"""
        raise NotImplementedError

    def filter_value(self, day_rows):
        return None

    def rescale(self, profile, times, values):
        return profile


# --- demand: load-curve shape, parametrized by (peakiness, peak_hour) ---


@dataclass
class LoadProfileSampler(ProfileSampler):
    """Sample demand_profile.csv's daily shape (peakiness + peak_hour); resample rescales to each day's own peak."""

    to_perturb = "demand_profile"

    library_dir: Path = Path(__file__).parent / "reference_data" / "demand_profiles"
    k: int = 5  # >1 blends the k nearest real profiles; 1 uses the single nearest as-is

    SCENARIOS = {
        "flat": {"peakiness": 0.9, "peak_hour": 19.0},
        "peaky": {"peakiness": 0.6, "peak_hour": 19.0},
        "morning-peak": {"peakiness": 0.75, "peak_hour": 9.0},
    }

    def prepare_profiles(self, df):
        """demand_profile.csv: zone, time, value[, year] -- one day per (zone[, year])."""
        group_cols = [c for c in ["zone", "year"] if c in df.columns]
        return split_into_days(df, group_cols)

    def extract_features(self, times, values):
        return {"peakiness": float(values.mean() / values.max()), "peak_hour": float(times[np.argmax(values)])}

    def randomize_features(self, features, rng, scale):
        # largest shifts: 0.2 peakiness, 4 hours
        shift = rng.choice([-1, 1]) * abs(rng.normal(scale * 0.2, 0.02))
        peakiness = features["peakiness"] + shift
        if not 0.3 <= peakiness <= 1.0:
            peakiness = features["peakiness"] - shift  # shift the other way instead
        peak_hour = features["peak_hour"] + rng.choice([-1, 1]) * abs(rng.normal(scale * 4.0, 0.5))
        return {"peakiness": float(np.clip(peakiness, 0.3, 1.0)), "peak_hour": float(peak_hour % 24)}

    @staticmethod
    def _hour_distance(h1, h2):
        """Circular distance between two hours-of-day, wrapping at 24 (so 23 and 1 are 2 apart)."""
        d = abs(h1 - h2) % 24
        return min(d, 24 - d)

    def feature_distance(self, target_features, library_features):
        peakiness_diff = library_features["peakiness"] - target_features["peakiness"]
        peak_hour_diff = library_features["peak_hour"].apply(lambda h: self._hour_distance(h, target_features["peak_hour"]))
        peak_hour_diff = peak_hour_diff / 12.0  # scale by its max possible value so it's comparable to peakiness_diff
        return (peakiness_diff ** 2 + peak_hour_diff ** 2) ** 0.5

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
    """Sample renewable_profiles.csv's daily capacity-factor shape; filter_value is the solar/wind category."""

    to_perturb = "renewable_profiles"
    FILTER_COL = "category"

    library_dir: Path = Path(__file__).parent / "reference_data" / "renewable_profiles"
    k: int = 1  # blending real profiles could push an hourly value above 1 (physically invalid)

    # capacity factor per category
    SCENARIOS = {
        "solar": {"low": 0.15, "mid": 0.2, "high": 0.25},
        "wind": {"low": 0.2, "mid": 0.3, "high": 0.4},
    }

    def scenario_names(self, filter_value=None):
        return list(self.SCENARIOS[filter_value])

    def named_scenario(self, name, filter_value=None):
        return {"capacity_factor": self.SCENARIOS[filter_value][name]}

    def prepare_profiles(self, df):
        """renewable_profiles.csv: profile_id, tech, zone, time, value -- one day per profile_id."""
        return split_into_days(df, ["profile_id"])

    def extract_features(self, times, values):
        return {"capacity_factor": float(np.mean(values))}

    def randomize_features(self, features, rng, scale):
        # largest shift: 50% of the day's own capacity factor (relative, since solar and wind differ)
        cf = features["capacity_factor"]
        shift = rng.choice([-1, 1]) * abs(rng.normal(scale * 0.5, 0.05)) * cf
        capacity_factor = cf + shift
        if not 0.001 <= capacity_factor <= 0.999:
            capacity_factor = cf - shift  # shift the other way instead
        return {"capacity_factor": float(np.clip(capacity_factor, 0.001, 0.999))}

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


def load_profile_library(library_dir):
    """Reshape a library's (metadata.csv, curves.csv) into one DataFrame with times/values array columns."""
    library_dir = Path(library_dir)
    metadata = pd.read_csv(library_dir / "daily_profile_metadata.csv").set_index("profile_id")
    profiles = pd.read_csv(library_dir / "daily_profile_curves.csv").groupby("profile_id")
    metadata["times"] = profiles["hour"].apply(lambda s: s.to_numpy(dtype=float))
    metadata["values"] = profiles["value"].apply(lambda s: s.to_numpy(dtype=float))
    return metadata.reset_index()
