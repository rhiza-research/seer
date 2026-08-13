"""Shared helpers for consolidating PyPSA's per-vintage build-option components (one component
per (build option, investment-period vintage), e.g. `name_2025`, `name_2030`, ...) back into a
single logical row per base name, matching the fixed per-option naming used on the pyomo side.
Used by both output.py and storage_soc.py.
"""

import pandas as pd


def _base_name(name, build_year):
    """Strip the "_<build_year>" suffix used for per-vintage new-build components (see
    build_generators.py/build_storage.py: `gen_year_name = name + "_" + str(year)`), so multiple
    vintages of the same build option collapse back into one logical name, matching the fixed
    per-option names used on the pyomo side."""
    suffix = f"_{int(build_year)}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def capacity_by_year(model, static_df):
    """static_df: model.generators or model.storage_units (must have build_year, lifetime,
    p_nom_opt). Returns a DataFrame indexed by investment year, one column per logical (base)
    name, summing p_nom_opt across every vintage active in that year."""
    base_names = pd.Series(
        [_base_name(name, row.build_year) for name, row in static_df.iterrows()],
        index=static_df.index,
    )

    rows = {}
    for year in model.investment_periods:
        active = (static_df["build_year"] <= year) & (year < static_df["build_year"] + static_df["lifetime"])
        rows[year] = static_df.loc[active, "p_nom_opt"].groupby(base_names[active]).sum()

    df = pd.DataFrame(rows).T.fillna(0.0)
    df.index.name = None
    df.columns.name = None
    return df


def timeseries_by_base_name(static_df, t_df):
    """t_df: e.g. model.generators_t.p, indexed by model.snapshots (year, timestamp), columns =
    component names. Collapses vintage-specific columns to their logical base name (summing any
    that overlap) and reindexes to a plain (year, hour) index, hour = 1-indexed position within
    each year - matching the pyomo side's convention."""
    base_name_map = {name: _base_name(name, static_df.loc[name, "build_year"]) for name in t_df.columns}
    renamed = t_df.rename(columns=base_name_map)
    renamed = renamed.T.groupby(level=0).sum().T

    years = renamed.index.get_level_values(0)
    hours = pd.Series(range(len(renamed)), index=renamed.index).groupby(years).cumcount() + 1
    renamed.index = pd.MultiIndex.from_arrays([years, hours.values], names=["year", "hour"])
    renamed.columns.name = None
    return renamed


def genmap(static_df):
    """static_df: model.generators or model.storage_units. Returns a DataFrame indexed by
    logical (base) name, columns=[tech, zone], one row per distinct build option (collapsing
    vintages, same as capacity_by_year/timeseries_by_base_name)."""
    base_names = [_base_name(name, row.build_year) for name, row in static_df.iterrows()]
    df = pd.DataFrame({"tech": static_df["carrier"].values, "zone": static_df["bus"].values}, index=base_names)
    return df[~df.index.duplicated()]
