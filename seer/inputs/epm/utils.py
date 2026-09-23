from pathlib import Path

import pandas as pd

from seer.models.time_utils import format_time_string


def init_df(path, columns):
    """Read a csv, or if nonexisting create empty dataframe with given columns"""
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def pivot_wide(df, index, param_list):
    """Pivot a long-format (index cols..., parameters, value) dataframe to one row per
    `index` combination, one column per `param_list` entry."""
    return (
        df[df["parameters"].isin(param_list)]
        .pivot_table(index=index, columns="parameters", values="value", aggfunc="first")
        .reindex(columns=param_list)
    )


def pivot_years(df, index, years):
    """Pivot a long-format (index cols..., year, value) dataframe to one row per `index`
    combination, one column per year in `years`."""
    return df.pivot_table(index=index, columns="year", values="value", aggfunc="first").reindex(columns=years)


def align(df, df_target, on):
    """Reorder `df`'s rows to follow `df_target[on]`'s order, matched by the `on` column."""
    order = pd.Categorical(df[on], categories=df_target[on], ordered=True)
    return df.assign(**{on: order}).sort_values(on, kind="stable").astype({on: df[on].dtype}).reset_index(drop=True)


def convert_season_day_hour(df, season_col="season", day_col="day", hour_col="hour"):
    """Parse season/day/hour columns into a single `time` column."""
    df = df.copy()
    month = df[season_col].str.extract(r"m(\d+)")[0].astype(int)
    day = df[day_col].str.extract(r"d(\d+)")[0].astype(int)
    hour = df[hour_col].str.extract(r"t(\d+)")[0].astype(int)  # 1-indexed -> 0-indexed below
    df["time"] = [format_time_string(m, d, h - 1) for m, d, h in zip(month, day, hour)]
    return df.drop(columns=[season_col, day_col, hour_col])
