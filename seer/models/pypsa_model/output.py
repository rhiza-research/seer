"""Standardized long-format output tables for a solved PyPSA seer model (see
seer/models/README.md for the output schema): generator_capacity, storage_capacity, dispatch.

PyPSA needs one component per (build option, investment-period vintage) for anything
expandable; these are consolidated back into a single logical row per base name by
_consolidate.py, shared with storage_soc.py.
"""

import pandas as pd

from ._consolidate import capacity_by_year, genmap, timeseries_by_base_name


def _melt_capacity(capacity_df, genmap_df, name_col):
    """capacity_df: from _consolidate.capacity_by_year (index=year, columns=base names)."""
    long_df = capacity_df.copy()
    long_df.index.name = "year"
    long_df = long_df.reset_index().melt(id_vars="year", var_name=name_col, value_name="capacity")
    long_df["zone"] = long_df[name_col].map(genmap_df["zone"])
    long_df["tech"] = long_df[name_col].map(genmap_df["tech"])
    return long_df[["zone", "tech", name_col, "year", "capacity"]]


def _melt_dispatch(dispatch_df, genmap_df):
    """dispatch_df: from _consolidate.timeseries_by_base_name (index=(year, hour), columns=base names)."""
    long_df = dispatch_df.reset_index().melt(id_vars=["year", "hour"], var_name="name", value_name="dispatch")
    long_df["zone"] = long_df["name"].map(genmap_df["zone"])
    long_df["tech"] = long_df["name"].map(genmap_df["tech"])
    return long_df[["zone", "tech", "name", "year", "hour", "dispatch"]]


def build_output_tables(model):
    """Build the three seer output tables from a solved PyPSA Network.

    Returns {"generator_capacity": df, "storage_capacity": df, "dispatch": df}.
    """
    gen_genmap = genmap(model.generators)
    sto_genmap = genmap(model.storage_units)

    generator_capacity = _melt_capacity(capacity_by_year(model, model.generators), gen_genmap, "generator")
    storage_capacity = _melt_capacity(capacity_by_year(model, model.storage_units), sto_genmap, "storage")

    generator_dispatch = _melt_dispatch(
        timeseries_by_base_name(model.generators, model.generators_t.p), gen_genmap
    )
    # PyPSA's native StorageUnit.p convention is already positive = discharging (net supply to
    # the grid), matching our dispatch sign convention directly - no negation needed here
    storage_dispatch = _melt_dispatch(
        timeseries_by_base_name(model.storage_units, model.storage_units_t.p), sto_genmap
    )

    dispatch = pd.concat([generator_dispatch, storage_dispatch], ignore_index=True)

    return {
        "generator_capacity": generator_capacity,
        "storage_capacity": storage_capacity,
        "dispatch": dispatch,
    }
