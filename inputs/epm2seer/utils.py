import re
from pathlib import Path

import pandas as pd

from seer.models.utils.time_utils import format_time_string


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
    """Parse season/day/hour columns into a single `time` column.

    `t<N>` is 1-indexed. Usually it spans a single 24-hour day in 24 steps (hourly), but not
    always -- `n_steps` is the highest t-index appearing in `df`, and when it's a clean multiple
    of 24 greater than 24, that's multiple representative days packed end-to-end under one
    nominal day label, not finer sub-hourly resolution: confirmed for Senegal's Demand/REProfile
    sheets (t1..t72) via its own Duration sheet, whose per-step weights repeat in three distinct
    24-column blocks rather than describing 72 uniform 20-minute steps. Each 24-step block gets
    its own synthetic day number (<day> * 10 + block index) instead of being chopped into
    sub-hourly timestamps."""
    df = df.copy()
    month = df[season_col].str.extract(r"m(\d+)")[0].astype(int)
    day = df[day_col].str.extract(r"d(\d+)")[0].astype(int)
    t_index = df[hour_col].str.extract(r"t(\d+)", flags=re.IGNORECASE)[0].astype(int)  # 1-indexed
    n_steps = t_index.max()

    if n_steps > 24 and n_steps % 24 == 0:
        block = (t_index - 1) // 24
        day = day * 10 + block + 1
        hour = (t_index - 1) % 24
        minute = pd.Series(0, index=df.index)
    else:
        total_minutes = ((t_index - 1) * (24 / n_steps) * 60).round().astype(int)
        hour, minute = total_minutes // 60, total_minutes % 60

    df["time"] = [format_time_string(m, d, h, mi) for m, d, h, mi in zip(month, day, hour, minute)]
    return df.drop(columns=[season_col, day_col, hour_col])
