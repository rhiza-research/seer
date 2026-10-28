"""Shared post-processing for both model backends' output tables."""

from pathlib import Path

import pandas as pd

KEY_COLS = ["zone", "tech", "fuel"]


def add_input_metadata(tables, data_path):
    """Attach zone/tech/fuel to each output table by joining on the generator/storage name
    against the seer-schema inputs at `data_path` -- the same source for both backends, rather
    than whatever each solver happens to carry (PyPSA only has a single `carrier`). Storage has
    no fuel in the schema, so its `fuel` is empty. Returns new tables, key columns first."""
    data_path = Path(data_path)
    gens = pd.read_csv(data_path / "generator_constants.csv")[["generator", *KEY_COLS]]
    sto = pd.read_csv(data_path / "storage_constants.csv")[["storage", "zone", "tech"]].assign(fuel=pd.NA)
    meta = pd.concat([gens.rename(columns={"generator": "name"}), sto.rename(columns={"storage": "name"})])
    meta = meta.drop_duplicates("name").set_index("name")

    out = {}
    for table, df in tables.items():
        if table not in ("generator_capacity", "storage_capacity", "storage_soc", "dispatch"):
            out[table] = df  # e.g. line_capacity -- keyed by zone pair, nothing to join
            continue
        name_col = {"generator_capacity": "generator", "storage_capacity": "storage", "storage_soc": "storage"}.get(table, "name")
        df = df.drop(columns=[c for c in KEY_COLS if c in df.columns])
        joined = df.join(meta, on=name_col)
        out[table] = joined[KEY_COLS + [c for c in df.columns]]
    return out


SUMMARY_UNITS = {"objective": "$ (discounted to the first year)", "generator_capex": "$ (discounted to the first year)",
                 "storage_capex": "$ (discounted to the first year)", "line_capex": "$ (discounted to the first year)",
                 "variable_cost": "$ (discounted to the first year)", "lost_load_cost": "$ (discounted to the first year)",
                 "demand": "MWh (all modeled years)", "lost_load": "MWh (all modeled years)"}


def summary_table(objective, costs, demand, lost_load):
    """The summary output table: the objective, its cost terms (costs: {term: $}), and demand and lost load
    energy summed over every modeled year -- one row per metric, as (metric, value, unit)."""
    values = {"objective": objective, **costs, "demand": demand, "lost_load": lost_load}
    return pd.DataFrame({"metric": list(values), "value": [float(v) for v in values.values()],
                         "unit": [SUMMARY_UNITS[m] for m in values]})
