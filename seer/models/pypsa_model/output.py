"""Standardized long-format output tables for a solved PyPSA seer model (see
seer/models/README.md for the output schema): generator_capacity, storage_capacity, dispatch.

PyPSA needs one component per (build option, investment-period vintage) for anything
expandable; the _consolidate_* helpers below collapse these back into a single logical row
per base name, matching the fixed per-option naming used on the pyomo side.
"""

import pandas as pd


def _base_name(name, build_year):
    """Strip a build option's "_<build_year>" vintage suffix (see build.py:
    `gen_year_name = f"{name}_{year}"`) to recover its logical (base) name."""
    suffix = f"_{int(build_year)}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _consolidate_capacity_by_year(model, static_df):
    """static_df: model.generators or model.storage_units. Returns capacity (p_nom_opt)
    indexed by year, one column per base name, summed across every vintage active that year."""
    base_names = pd.Series(
        [_base_name(name, row.build_year) for name, row in static_df.iterrows()], index=static_df.index
    )
    rows = {}
    for year in model.investment_periods:
        active = (static_df["build_year"] <= year) & (year < static_df["build_year"] + static_df["lifetime"])
        rows[year] = static_df.loc[active, "p_nom_opt"].groupby(base_names[active]).sum()

    df = pd.DataFrame(rows).T.fillna(0.0)
    df.index.name = None
    df.columns.name = None
    return df


def _consolidate_timeseries_by_base_name(static_df, t_df):
    """t_df: e.g. model.generators_t.p, indexed by model.snapshots. Collapses vintage columns
    to their base name (summing overlaps) and reindexes to (year, hour), hour 1-indexed within
    each year -- matching the pyomo side's convention."""
    base_name_map = {name: _base_name(name, static_df.loc[name, "build_year"]) for name in t_df.columns}
    renamed = t_df.rename(columns=base_name_map).T.groupby(level=0).sum().T

    years = renamed.index.get_level_values(0)
    hours = pd.Series(range(len(renamed)), index=renamed.index).groupby(years).cumcount() + 1
    renamed.index = pd.MultiIndex.from_arrays([years, hours.values], names=["year", "hour"])
    renamed.columns.name = None
    return renamed


def _genmap(static_df):
    """static_df: model.generators or model.storage_units. Returns one row per base name
    (collapsing vintages), columns=[tech, zone]."""
    base_names = [_base_name(name, row.build_year) for name, row in static_df.iterrows()]
    df = pd.DataFrame({"tech": static_df["carrier"].values, "zone": static_df["bus"].values}, index=base_names)
    return df[~df.index.duplicated()]


def _melt_capacity(capacity_df, genmap_df, name_col):
    """capacity_df: from _consolidate_capacity_by_year (index=year, columns=base names)."""
    long_df = capacity_df.copy()
    long_df.index.name = "year"
    long_df = long_df.reset_index().melt(id_vars="year", var_name=name_col, value_name="capacity")
    long_df["zone"] = long_df[name_col].map(genmap_df["zone"])
    long_df["tech"] = long_df[name_col].map(genmap_df["tech"])
    return long_df[["zone", "tech", name_col, "year", "capacity"]]


def _melt_dispatch(dispatch_df, genmap_df):
    """dispatch_df: from _consolidate_timeseries_by_base_name (index=(year, hour), columns=base names)."""
    long_df = dispatch_df.reset_index().melt(id_vars=["year", "hour"], var_name="name", value_name="dispatch")
    long_df["zone"] = long_df["name"].map(genmap_df["zone"])
    long_df["tech"] = long_df["name"].map(genmap_df["tech"])
    return long_df[["zone", "tech", "name", "year", "hour", "dispatch"]]


def build_output_tables(model):
    """Build the three seer output tables from a solved PyPSA Network.

    Returns {"generator_capacity": df, "storage_capacity": df, "dispatch": df}.
    """
    gen_genmap = _genmap(model.generators)
    sto_genmap = _genmap(model.storage_units)

    generator_capacity = _melt_capacity(_consolidate_capacity_by_year(model, model.generators), gen_genmap, "generator")
    storage_capacity = _melt_capacity(_consolidate_capacity_by_year(model, model.storage_units), sto_genmap, "storage")

    generator_dispatch = _melt_dispatch(
        _consolidate_timeseries_by_base_name(model.generators, model.generators_t.p), gen_genmap
    )
    # PyPSA's native StorageUnit.p convention is already positive = discharging (net supply to
    # the grid), matching our dispatch sign convention directly - no negation needed here
    storage_dispatch = _melt_dispatch(
        _consolidate_timeseries_by_base_name(model.storage_units, model.storage_units_t.p), sto_genmap
    )

    dispatch = pd.concat([generator_dispatch, storage_dispatch], ignore_index=True)

    return {
        "generator_capacity": generator_capacity,
        "storage_capacity": storage_capacity,
        "dispatch": dispatch,
    }
