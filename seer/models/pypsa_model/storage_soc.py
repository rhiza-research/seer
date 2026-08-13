"""Standardized long-format output table for storage state of charge on a solved PyPSA seer
model (see seer/models/README.md for the output schema): storage_soc.
"""

from ._consolidate import genmap, timeseries_by_base_name


def build_storage_soc(model):
    """Build storage_soc: hourly state of charge (MWh) for each storage unit, consolidated
    across build-year vintages (see _consolidate.py).

    Returns a DataFrame with columns zone, tech, storage, year, hour, soc.
    """
    storage_genmap = genmap(model.storage_units)
    soc_df = timeseries_by_base_name(model.storage_units, model.storage_units_t.state_of_charge)

    long_df = soc_df.reset_index().melt(id_vars=["year", "hour"], var_name="storage", value_name="soc")
    long_df["zone"] = long_df["storage"].map(storage_genmap["zone"])
    long_df["tech"] = long_df["storage"].map(storage_genmap["tech"])
    return long_df[["zone", "tech", "storage", "year", "hour", "soc"]]
