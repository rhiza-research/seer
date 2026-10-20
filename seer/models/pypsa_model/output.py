"""Standardized long-format output tables for a solved PyPSA seer model (see
seer/models/README.md for the output schema): generator_capacity, storage_capacity, dispatch,
line_capacity, lost_load.

PyPSA needs one component per (build option, investment-period vintage) for anything
expandable; the _consolidate_* helpers below collapse these back into a single logical row
per base name, matching the fixed per-option naming used on the pyomo side.
"""

import pandas as pd

from ..utils.output_utils import add_input_metadata
from .build import LOAD_SHEDDING


def _base_name(name, row):
    """Strip a build option's "_<build_year>" vintage suffix (see build.py:
    `gen_year_name = f"{name}_{year}"`) to recover its logical (base) name. Only build options
    carry the suffix -- an existing unit's own name can end in its start year (e.g. Botswana's
    Solar_Bot_2022) and is left alone."""
    if not row["p_nom_extendable"]:
        return name
    suffix = f"_{int(row['build_year'])}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _consolidate_capacity_by_year(model, static_df):
    """static_df: model.generators or model.storage_units. Returns capacity (p_nom_opt)
    indexed by year, one column per base name, summed across every vintage active that year."""
    base_names = pd.Series(
        [_base_name(name, row) for name, row in static_df.iterrows()], index=static_df.index
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
    to their base name (summing overlaps) and reindexes to (year, hour, datetime): hour is
    1-indexed within each year (matching the pyomo side's convention), datetime is the real
    timestamp already carried by model.snapshots."""
    base_name_map = {name: _base_name(name, static_df.loc[name]) for name in t_df.columns}
    renamed = t_df.rename(columns=base_name_map).T.groupby(level=0).sum().T

    years = renamed.index.get_level_values(0)
    datetimes = renamed.index.get_level_values(1)
    hours = pd.Series(range(len(renamed)), index=renamed.index).groupby(years).cumcount() + 1
    renamed.index = pd.MultiIndex.from_arrays([years, hours.values, datetimes], names=["year", "hour", "datetime"])
    renamed.columns.name = None
    return renamed


def _genmap(static_df):
    """static_df: model.generators or model.storage_units. Returns one row per base name
    (collapsing vintages), columns=[tech, zone]."""
    base_names = [_base_name(name, row) for name, row in static_df.iterrows()]
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
    """dispatch_df: from _consolidate_timeseries_by_base_name (index=(year, hour, datetime), columns=base names)."""
    long_df = dispatch_df.reset_index().melt(
        id_vars=["year", "hour", "datetime"], var_name="name", value_name="dispatch"
    )
    long_df["zone"] = long_df["name"].map(genmap_df["zone"])
    long_df["tech"] = long_df["name"].map(genmap_df["tech"])
    return long_df[["zone", "tech", "name", "year", "hour", "datetime", "dispatch"]]


def build_output_tables(model, data_path):
    """Build the three seer output tables from a solved PyPSA Network. zone/tech/fuel come
    from the seer-schema inputs at `data_path` (see output_utils.add_input_metadata).

    Returns {"line_capacity", "generator_capacity", "storage_capacity", "dispatch", "lost_load"}: df.
    """
    # load shedding isn't a real generator -- reported as lost_load, not in the generator tables
    is_shedding = model.generators["carrier"] == LOAD_SHEDDING
    generators = model.generators[~is_shedding]
    generators_p = model.generators_t.p[generators.index]
    gen_genmap = _genmap(generators)
    sto_genmap = _genmap(model.storage_units)

    generator_capacity = _melt_capacity(_consolidate_capacity_by_year(model, generators), gen_genmap, "generator")
    # storage capacity is reported in MWh of energy (output schema) -- a StorageUnit's p_nom is MW
    # of power, so scale by its fixed duration (max_hours)
    storage_units = model.storage_units.assign(p_nom_opt=model.storage_units["p_nom_opt"] * model.storage_units["max_hours"])
    storage_capacity = _melt_capacity(_consolidate_capacity_by_year(model, storage_units), sto_genmap, "storage")

    generator_dispatch = _melt_dispatch(
        _consolidate_timeseries_by_base_name(generators, generators_p), gen_genmap
    )
    # PyPSA's native StorageUnit.p convention is already positive = discharging (net supply to
    # the grid), matching our dispatch sign convention directly - no negation needed here
    storage_dispatch = _melt_dispatch(
        _consolidate_timeseries_by_base_name(model.storage_units, model.storage_units_t.p), sto_genmap
    )

    dispatch = pd.concat([generator_dispatch, storage_dispatch], ignore_index=True)

    # unserved demand: the load-shedding generators' output (pyomo's lost_load)
    shedding = model.generators[is_shedding]
    lost_load = _consolidate_timeseries_by_base_name(shedding, model.generators_t.p[shedding.index])
    lost_load = lost_load.rename(columns=shedding["bus"]).reset_index().melt(
        id_vars=["year", "hour", "datetime"], var_name="zone", value_name="lost_load"
    )[["zone", "year", "hour", "datetime", "lost_load"]]

    # lines: existing Link + expansion vintages, consolidated to one row per (zone pair, year)
    if len(model.links):
        links = model.links.copy()
        links["p_nom_opt"] = links["p_nom_opt"].fillna(links["p_nom"])
        line_wide = _consolidate_capacity_by_year(model, links)
        line_ends = links[~links.index.duplicated()].assign(
            base=[_base_name(name, row) for name, row in links.iterrows()]
        ).drop_duplicates("base").set_index("base")[["bus0", "bus1"]]
        line_capacity = line_wide.rename_axis("year").reset_index().melt(
            id_vars="year", var_name="line", value_name="capacity"
        )
        line_capacity["from"] = line_capacity["line"].map(line_ends["bus0"])
        line_capacity["to"] = line_capacity["line"].map(line_ends["bus1"])
        # one row per (zone pair, year): sum a pair's lines with different start years
        line_capacity = line_capacity.groupby(["from", "to", "year"], as_index=False, sort=False)["capacity"].sum()
    else:
        line_capacity = pd.DataFrame(columns=["from", "to", "year", "capacity"])

    return add_input_metadata({
        "line_capacity": line_capacity,
        "generator_capacity": generator_capacity,
        "storage_capacity": storage_capacity,
        "dispatch": dispatch,
        "lost_load": lost_load,
    }, data_path)
