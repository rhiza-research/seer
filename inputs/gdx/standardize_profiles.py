"""Standardize season/day/hour formatting in VRE and demand profile dataframes
for ingestion into epm_to_seer schema conversion.

Different regions' GDX exports encode the same information inconsistently:
season as quarters (Q1-Q4), half-years (Q1/Q2, wet/dry), bare month integers,
or already-correct "m1".."m12" strings; day as "d1".."dN" or bare integers;
and VRE (esp. PV) profiles that only cover daylight hours instead of the
full t01-t24 day. This module rewrites all of the above into the format
epm2seer expects: season -> "m1".."m12", day ->
"d1".."dN", hour -> zero-padded "t01".."t24" with zero-filled gaps.
"""

# Import relevant packages
import re
import pandas as pd

# Explicit, adjustable convention for 2-valued ("half-year") season columns:
# index 0 -> months 1-6, index 1 -> months 7-12. Not calendar-verified —
# wet/dry assignment in particular is an arbitrary but consistent choice.

# Define dictionaries for mapping seasons to months
SEASON_HALF_YEAR_MAP = {
    "Q1": 0,
    "Q2": 1,
    "dry": 0,
    "wet": 1,
}

QUARTER_MONTHS = {
    "Q1": [1, 2, 3],
    "Q2": [4, 5, 6],
    "Q3": [7, 8, 9],
    "Q4": [10, 11, 12],
}

HALF_YEAR_MONTHS = {
    0: [1, 2, 3, 4, 5, 6], 
    1: [7, 8, 9, 10, 11, 12]}

# Define the GAMS symbols for adjustment
PROFILE_SYMBOLS = ("pVREProfile", "pVREgenProfile", "pDemandProfile", "pDemandData")

# Adjust season columns to follow convention of m1, m2, ... m12
def standardize_season_column(df, season_col="season"):
    """Rewrite df[season_col] to "m1".."m12" strings, expanding rows for
    quarter/half-year seasons so every represented month gets a copy of the
    season's row (same value).
    """

    # If column not in dataframe, return dataframe
    if season_col not in df.columns:
        return df

    # Extract the values in season column
    values = df[season_col].astype(str)
    unique_values = set(values.unique())

    # Already matches desired formart "m<N>"
    if all(re.fullmatch(r"m\d+", v) for v in unique_values):
        return df

    # Bare month integers, e.g. "1".."12" (no letter prefix) -> just prefix "m"
    if all(re.fullmatch(r"\d+", v) for v in unique_values):
        out = df.copy()
        out[season_col] = "m" + values
        return out

    # Handle seasons represented as quarters: "Q1".."Q4"
    if unique_values <= set(QUARTER_MONTHS):
        if len(unique_values) == 4:
            return _expand_season(df, season_col, values, QUARTER_MONTHS)
        if len(unique_values) == 2:
            half_year_lookup = {q: HALF_YEAR_MONTHS[SEASON_HALF_YEAR_MAP[q]] for q in unique_values}
            return _expand_season(df, season_col, values, half_year_lookup)
        raise ValueError(
            f"Unexpected quarter set {sorted(unique_values)} in column '{season_col}' "
            "(expected 2 or 4 distinct quarters)"
        )

    # Hand seasons represented as wet/dry half-year
    if unique_values <= set(SEASON_HALF_YEAR_MAP) and len(unique_values) == 2:
        half_year_lookup = {v: HALF_YEAR_MONTHS[SEASON_HALF_YEAR_MAP[v]] for v in unique_values}
        return _expand_season(df, season_col, values, half_year_lookup)

    raise ValueError(f"Unrecognized season format in column '{season_col}': {sorted(unique_values)}")

# Create month rows for quarterly or binary season defintions
def _expand_season(df, season_col, values, months_lookup):
    """
    Note: Several regions present demand and renewable generation profile based on 
    representative days instead of calendar days.

    Duplicate each row once per month in months_lookup[value], setting
    season_col to "m<month>" on each copy.

    months_lookup -> dictionary
    """

    # Create a copy of the dataframe
    out = df.copy()

    # Write the season labels back in
    out[season_col] = values.values

    # Create scratch column to map season values to mapping dictionary
    # _months holds a list for the mapped values (e.g. Q1, hour=t1, value=0.5 has _months=[1,2,3])
    out["_months"] = out[season_col].map(months_lookup)

    # Turn 1 row into three rows based on the list in the row, all values are replicated across rows
    # For every (day, hour) pair within Q1, the row's _months list ([1, 2, 3]) turns that one row
    # into three rows, one per month with each row containing the same value
    out = out.explode("_months", ignore_index=True)

    # Reformat month label to "m<N>"
    out[season_col] = "m" + out["_months"].astype(int).astype(str)

    return out.drop(columns="_months")

# Adjust day columns to follow convention of d1, d2, ... d31
def standardize_day_column(df, day_col="day"):
    """Rewrite bare-integer day values (e.g. "1", "2") to "d1", "d2", ...
    No-op if already "d<N>"-formatted.
    """

    # If column not in dataframe, return dataframe
    if day_col not in df.columns:
        return df

    # Already matches desired format "d<N>"
    values = df[day_col].astype(str)
    if all(re.fullmatch(r"d\d+", v) for v in values.unique()):
        return df

    # Ensure that all day values start with d prefix
    if all(re.fullmatch(r"\d+", v) for v in values.unique()):
        out = df.copy()
        out[day_col] = "d" + values
        return out

    raise ValueError(f"Unrecognized day format in column '{day_col}': {sorted(values.unique())}")


def fill_missing_hours(df, hour_col="hour", value_col="value"):
    """Reindex every (all-columns-but-hour/value) group to the full t01..t24
    day, zero-filling any hour missing from the source data (e.g. PV rows
    that only cover daylight hours). Also zero-pads the hour label.
    """

    # If column not in dataframe, return dataframe
    if hour_col not in df.columns or value_col not in df.columns:
        return df

    # Group all other columns which are not the hour or value column (e.g. zone, tech, season, day)
    group_cols = [c for c in df.columns if c not in (hour_col, value_col)]

    # Create a copy of the dataframe with a scratch column for reindexing the hour column
    out = df.copy()
    out["_hour_num"] = out[hour_col].str.extract(r"t(\d+)").astype(int)

    # For every combination of group_cols, left-join with full 24 hours, fill missing hours with 0.0
    full_grid = out[group_cols].drop_duplicates().merge(
        pd.DataFrame({"_hour_num": range(1,25)}), how="cross"
    )

    out = full_grid.merge(out[group_cols + ["_hour_num", value_col]], on=group_cols + ["_hour_num"], how="left")
    out[value_col] = out[value_col].fillna(0.0)

    # Convert numeric hour to format "t01"..."t024"
    out[hour_col] = "t" + out["_hour_num"].astype(str).str.zfill(2)
    out = out.drop(columns="_hour_num")

    return out[group_cols + [hour_col, value_col]]

# 
def standardize_profile_dataframe(df):
    """Apply day -> season -> hour standardization, in that order (season
    expansion must happen before hour-filling so gaps are filled per
    already-expanded month/day group).
    """

    df = standardize_day_column(df)
    df = standardize_season_column(df)
    df = fill_missing_hours(df)

    # Sort the dataframe
    df = df.copy()
    df["_month_num"] = df["season"].str.extract(r"m(\d+)").astype(int)
    df["_day_num"] = df["day"].str.extract(r"d(\d+)").astype(int)
    df["_hour_num"] = df["hour"].str.extract(r"t(\d+)").astype(int)

    id_cols = [c for c in df.columns if c not in ("season", "day", "hour", "value") and not c.startswith("_")]
    for col in id_cols:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    df = df.sort_values(id_cols + ["_month_num", "_day_num", "_hour_num"])

    return df.drop(columns=["_month_num", "_day_num", "_hour_num"]).reset_index(drop=True)

def standardize_all_profiles(region, country_data):
    """Apply standardize_profile_dataframe to every present/non-empty profile
    symbol (pVREProfile, pVREgenProfile, pDemandProfile, pDemandData) for a
    region. Returns {symbol_name: normalized_df} for the symbols processed.
    """

    updates = {}
    for symbol in PROFILE_SYMBOLS:
        df = country_data.get(symbol)
        if df is None or df.empty:
            continue
        try:
            updates[symbol] = standardize_profile_dataframe(df)
        except ValueError as exc:
            print(f"  [{region}] [{symbol}] warn — skipping normalization: {exc}")

    return updates
