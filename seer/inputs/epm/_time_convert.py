"""Convert EPM's season/day/hour encoding ("m1".."m12"/"d1".."dN"/"t1".."t24") into the seer
schema's single `time` column (see seer/inputs/README.md's times.csv entry). "day" here is a
representative-day label, not necessarily a real calendar date (EPM data is often a handful of
representative days per month, not a full year) -- it's mapped onto a day-of-month number the
same way the prior season/day/hour schema did, so behavior relative to those old columns is
unchanged; only the representation (one `time` string vs. three columns) is new.
"""
from seer.models.time_utils import format_time_string


def convert_season_day_hour(df, season_col="season", day_col="day", hour_col="hour"):
    """Replace season/day/hour columns with a single `time` column."""
    df = df.copy()
    month = df[season_col].str.extract(r"m(\d+)")[0].astype(int)
    day = df[day_col].str.extract(r"d(\d+)")[0].astype(int)
    hour = df[hour_col].str.extract(r"t(\d+)")[0].astype(int)  # 1-indexed -> 0-indexed below
    df["time"] = [format_time_string(m, d, h - 1) for m, d, h in zip(month, day, hour)]
    return df.drop(columns=[season_col, day_col, hour_col])
