"""Perturb profiles by resampling stochastically from profile libraries."""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Perturbation
from ..models.utils.time_utils import parse_time_string


class ProfilePerturbation(Perturbation):
    """Shared perturb pipeline."""

    to_perturb: str  # SeerInstance attribute to perturb
    FILTER_COL: str | None = None  # library column restricting candidates, via filter_value

    def apply(self, seer_instance, rng):
        """Perturb the profiles of a given attribute in a SeerInstance."""
        out = seer_instance.copy()
        df = getattr(out, self.to_perturb)  # the seer instance attribute to perturb
        library = load_profile_library(self.library_dir)  # the library of profiles we sample from

        # Step 1: compute features on each profile in the library
        library_features = pd.DataFrame(
            [self.extract_features(t, v) for t, v in zip(library["times"], library["values"])], index=library.index
        )
        shared_perturbation = self.get_perturbation(rng)  # only used if perturbation_mode == "shared"

        sampled_days = []
        for pid, times, values, day_rows in self.prepare_profiles(df):
            # Step 2: compute this day's own features
            features = self.extract_features(times, values)

            # Step 3: perturb the features to get a target
            perturbation = shared_perturbation if self.perturbation_mode == "shared" else self.get_perturbation(rng)
            target_features = self.randomize_features(features, perturbation)

            # Step 4: restrict to library rows matching this day (e.g. renewable's solar/wind category; none for demand)
            if self.FILTER_COL is None:
                library_profiles = library
            else:
                library_profiles = library[library[self.FILTER_COL] == self.filter_value(day_rows)]
            if library_profiles.empty:
                continue  # e.g. renewable tech with no matching category library (geothermal etc.) -- leave unperturbed

            # Step 5: sample perturbed profile from the library.
            matched = self.sample_perturbed_profile(times, target_features, library_profiles, library_features.loc[library_profiles.index], rng)

            # Step 6: rescale (identity by default) and write back
            df.loc[day_rows.index, "value"] = self.rescale(matched, times, values)

            sampled_days.append({**pid, **features, **{f"target_{n}": v for n, v in target_features.items()}})

        setattr(out, self.to_perturb, df)
        self.sampled_ = {"perturbation_mode": self.perturbation_mode, "days": sampled_days}
        return out

    def sample_perturbed_profile(self, target_times, target_features, library_profiles, library_features, rng):
        """Sample a perturbed profile(s) from the library."""
        # compute distance between target and library features
        dist = self.feature_distance(target_features, library_features)
        nearest = dist.nsmallest(self.k).index
        # sample perturbed profile(s) from the library
        perturbed_profiles = [time_align(library_profiles.loc[i, "times"], library_profiles.loc[i, "values"], target_times) for i in nearest]
        if self.k == 1:
            perturbed_profile = perturbed_profiles[0]  # deterministic: the single nearest real profile
        else:
            weights = 1.0 / (dist.loc[nearest].to_numpy() + 1e-6)
            perturbed_profile = np.average(perturbed_profiles, axis=0, weights=weights / weights.sum())
        return perturbed_profile

    def prepare_profiles(self, df):
        """Yield (key, times, values, day_rows) per represented day in df -- see split_into_days below."""
        raise NotImplementedError

    def extract_features(self, times, values) -> dict:
        """Compute features for a profile."""
        raise NotImplementedError

    def randomize_features(self, features, perturbation) -> dict:
        """Perturb features by amount perturbation"""
        raise NotImplementedError

    def get_perturbation(self, rng) -> dict:
        """Sample a random offset per feature (written out explicitly per subclass below)."""
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
class LoadProfilePerturbation(ProfilePerturbation):
    """Perturb demand_profile.csv's daily shape (peakiness + peak_hour), rescaled to the day's own peak."""

    to_perturb = "demand_profile"

    library_dir: Path = Path(__file__).parent / "reference_data" / "demand_profiles"
    scale: float = 1.0  # multiplies both features' jitter std -- 0 = unperturbed, 2 = twice the default spread
    k: int = 5  # >1 blends the k nearest real profiles; 1 uses the single nearest as-is
    perturbation_mode: str = "shared"  # "shared": one draw for every day; "independent": a fresh draw per day
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def prepare_profiles(self, df):
        """demand_profile.csv: zone, time, value[, year] -- one day per (zone[, year])."""
        group_cols = [c for c in ["zone", "year"] if c in df.columns]
        return split_into_days(df, group_cols)

    def extract_features(self, times, values):
        return {"peakiness": float(values.mean() / values.max()), "peak_hour": float(times[np.argmax(values)])}

    def get_perturbation(self, rng):
        # typical perturbation moves peakiness by a few hundredths, peak_hour by an hour or two
        return {"peakiness": rng.normal(0, 0.05 * self.scale), "peak_hour": rng.normal(0, 1.5 * self.scale)}

    def randomize_features(self, features, perturbation):
        peakiness = float(np.clip(features["peakiness"] + perturbation["peakiness"], 0.01, 1.0))
        peak_hour = (features["peak_hour"] + perturbation["peak_hour"]) % 24  # wraps across midnight
        return {"peakiness": peakiness, "peak_hour": peak_hour}

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
class ReProfilePerturbation(ProfilePerturbation):
    """Perturb renewable_profiles.csv's daily capacity-factor shape."""

    to_perturb = "renewable_profiles"
    FILTER_COL = "category"

    library_dir: Path = Path(__file__).parent / "reference_data" / "renewable_profiles"
    scale: float = 1.0  # 0 = unperturbed; larger = wider spread around each day's own capacity_factor
    k: int = 1  # blending real profiles could push an hourly value above 1 (physically invalid)
    perturbation_mode: str = "independent"
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def prepare_profiles(self, df):
        """renewable_profiles.csv: profile_id, tech, zone, time, value -- one day per profile_id."""
        return split_into_days(df, ["profile_id"])

    def extract_features(self, times, values):
        return {"capacity_factor": float(np.mean(values))}

    def get_perturbation(self, rng):
        return {"capacity_factor": rng.normal(0, 0.05 * self.scale)}  # typical perturbation moves it by a few hundredths

    def randomize_features(self, features, perturbation):
        capacity_factor = float(np.clip(features["capacity_factor"] + perturbation["capacity_factor"], 0.001, 0.999))
        return {"capacity_factor": capacity_factor}

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
