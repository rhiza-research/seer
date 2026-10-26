"""Shared time-handling utilities for both model backends.

seer schema time strings ("MM-DD HH:MM:SS", see times.csv in seer/schema/README.md) deliberately
carry no year -- they're a template dispatch grid, projected onto whichever real years are
modeled. Parsing them must never invent a year (pandas/dateutil defaults to today's year;
datetime.strptime defaults to 1900, which isn't even a leap year and would reject Feb 29) -- so
parsing here is done by hand, and a real (year-bearing) datetime is only ever constructed once a
specific modeled year is known.
"""

import re

import pandas as pd

TIME_FORMAT = "%m-%d %H:%M:%S"
_TIME_RE = re.compile(r"^(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$")


def parse_time_string(s):
    """Parse a "MM-DD HH:MM:SS" string into (month, day, hour, minute, second) ints."""
    m = _TIME_RE.match(s)
    if not m:
        raise ValueError(f"Invalid time string {s!r}; expected format 'MM-DD HH:MM:SS'")
    return tuple(int(g) for g in m.groups())


def to_profile_times(datetimes):
    """times.csv labels ("MM-DD HH:MM:SS", no year) for datetimes."""
    return pd.DatetimeIndex(datetimes).strftime(TIME_FORMAT).tolist()


def format_time_string(month, day, hour, minute=0, second=0):
    """Inverse of parse_time_string."""
    return f"{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def project_to_year(month, day, hour, minute, second, year):
    """A real datetime for (month, day, ...) in `year`, or None if that date doesn't exist in
    `year` (e.g. Feb 29 in a non-leap year) -- callers should skip such rows."""
    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    except ValueError:
        return None


def load_times(data_path):
    """Read times.csv: the canonical (year-less) dispatch time grid, in chronological order."""
    return pd.read_csv(f"{data_path}/times.csv")["time"].tolist()


def load_time_weights(data_path):
    """times.csv's optional `weight` column as {time: weight}, or None if absent (uniform)."""
    times = pd.read_csv(f"{data_path}/times.csv")
    if "weight" not in times.columns or times["weight"].isna().all():
        return None
    return times.set_index("time")["weight"].astype(float).to_dict()


def build_snapshot_calendar(years, times, weights=None):
    """One row per (year, time, hour, datetime, weight) snapshot actually modeled across every
    year: `hour` is a 1-indexed position within that year's own snapshot list, `datetime` is the
    real projected timestamp, and `weight` is times.csv's relative weight for that time (1.0
    everywhere if `weights` is None). A time that doesn't exist in a given year (e.g. Feb 29 in a
    non-leap year) is skipped for that year -- so different years can have different lengths.
    """
    rows = []
    for year in years:
        hour = 0
        for t in times:
            dt = project_to_year(*parse_time_string(t), year)
            if dt is None:
                continue
            hour += 1
            weight = 1.0 if weights is None else weights[t]
            rows.append({"year": year, "time": t, "hour": hour, "datetime": dt, "weight": weight})
    return pd.DataFrame(rows)


def compute_time_weights(calendar):
    """Per-(year, hour) weight: how many real hours each modeled snapshot stands for, so a
    representative-days dispatch grid still sums to a full year's worth of operating cost in the
    objective. Each snapshot's relative weight (times.csv `weight`, see seer/schema/README.md --
    uniform if absent) is scaled so a year's weights sum to that real calendar year's hours:
    weight = year_hours x w / sum(w over that year). A full 8760/8784-snapshot year with no
    weights gets 1.0 everywhere (a no-op)."""
    if "weight" not in calendar.columns:
        calendar = calendar.assign(weight=1.0)
    weights = {}
    for year, group in calendar.groupby("year"):
        year_hours = 8784 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 8760
        scale = year_hours / group["weight"].sum()
        for row in group.itertuples():
            weights[(row.year, row.hour)] = row.weight * scale
    return weights


def resolve_profile(profile_df, calendar, key_cols, value_col="value"):
    """Expand a demand_profile/renewable_profiles-shaped dataframe to one row per
    (*key_cols, year, hour, value_col), matching `calendar` (see build_snapshot_calendar).

    profile_df must have a `time` column (values matching times.csv) plus `key_cols` and
    `value_col`; `year` is optional. If `year` is absent, entirely null, or has a single unique
    value, profile_df is a template: broadcast across every modeled year. Otherwise every
    (key, year, time) implied by `calendar` must be present, or a ValueError is raised.
    """
    key_cols = list(key_cols)
    is_template = "year" not in profile_df.columns or profile_df["year"].nunique(dropna=True) <= 1

    if is_template:
        resolved = calendar.merge(profile_df.drop(columns=["year"], errors="ignore"), on="time", how="left")
    else:
        resolved = calendar.merge(profile_df, on=["year", "time"], how="left")

    # two distinct gaps to catch: a modeled time with no data at all for any key (shows up as
    # NaN keys, since a left-merge only nulls when nothing on the right matched), and a key
    # present but missing specific times (its row count falls short of the full calendar)
    if resolved[key_cols].isna().any().any():
        raise ValueError(f"Some modeled time has no {value_col!r} at all -- check times.csv coverage")
    counts = resolved.groupby(key_cols).size()
    incomplete = counts[counts != len(calendar)]
    if len(incomplete) > 0:
        raise ValueError(
            f"Incomplete {value_col!r} coverage for {incomplete.index.tolist()} -- "
            "check against years.csv x times.csv"
        )
    return resolved.drop(columns=["time", "datetime"])
