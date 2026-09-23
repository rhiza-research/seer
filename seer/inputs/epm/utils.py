from seer.models.time_utils import format_time_string


def convert_season_day_hour(df, season_col="season", day_col="day", hour_col="hour"):
    """Parse season/day/hour columns into a single `time` column."""
    df = df.copy()
    month = df[season_col].str.extract(r"m(\d+)")[0].astype(int)
    day = df[day_col].str.extract(r"d(\d+)")[0].astype(int)
    hour = df[hour_col].str.extract(r"t(\d+)")[0].astype(int)  # 1-indexed -> 0-indexed below
    df["time"] = [format_time_string(m, d, h - 1) for m, d, h in zip(month, day, hour)]
    return df.drop(columns=[season_col, day_col, hour_col])
