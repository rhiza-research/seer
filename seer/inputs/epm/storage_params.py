"""Build storage_constants.csv and storage_costs.csv from EPM input data."""
import warnings

import pandas as pd

from .generator_params import NO_RETIREMENT_YEAR


def build_storage_params(data_path="data/Gambia/", years=None):
    """
    Processes storage input data into two seer schema files:
    - storage_constants.csv
    - storage_costs.csv

    Files that are read containing storage data:
    - pStorageDataInput.csv: storage data (by plant)
    - pGenDataInputCustom.csv: generator parameters (by generator)
    Returns two dataframes: storage_constants and storage_costs.

    `years` should be the same modeled years as years.csv (the model's investment periods) --
    storage_costs.csv must have a column for every one of them, since build_storage's
    per-vintage capex lookup indexes by those exact year values. Defaults to 2025-2050 for
    standalone use, but epm_to_seer always passes the region's actual years explicitly.
    """
    if years is None:
        years = list(range(2025, 2051))

    storage_data = pd.read_csv(data_path + "pStorageDataInput.csv")
    custom_gen_params = pd.read_csv(data_path + "pGenDataInputCustom.csv")

    """Setup storage constants.
    Use storage data to fill in a single dataframe of storage constants.
    """
    # pivot parameters into columns
    storage_constants = (
        storage_data
        .pivot(index="plant", columns="parameters", values="value")
        .reset_index()
    )

    # select and rename parameters (reindex, not [], since some regions have no "Life"
    # parameter at all for any storage plant -- that column just comes back all-NaN)
    param_list = ["plant", "CapacityMWh", "CapexMWh", "Efficiency", "Life"]
    param_names = ["storage", "capacity_mwh", "capex_mwh", "efficiency", "life"]
    storage_constants = storage_constants.reindex(columns=param_list).rename(columns=dict(zip(param_list, param_names)))

    """Pull the remaining attributes from the generator custom-parameters file - storage plants
    are just generators with tech="Storage" there - matched here by name, same as generator_constants.
    """
    storage_gen_params = custom_gen_params[custom_gen_params["generator"].isin(storage_constants["storage"])]

    # zone and tech (fuel) are plain columns rather than "parameters" rows. tech naming is
    # aligned to fixed values ("Battery", "PumpedHydro") upstream in gdx.enrich_data's
    # normalize_storage_tech, regardless of each region's raw naming quirks.
    zone_tech = storage_gen_params[["generator", "zone", "fuel"]].drop_duplicates().rename(
        columns={"generator": "storage", "fuel": "tech"}
    )
    storage_constants = storage_constants.merge(zone_tech, on="storage", how="left")

    # a storage plant with no matching pGenDataInputCustom row at all has no zone/tech
    # this should be dropped with a warning
    missing_zone = storage_constants["zone"].isna()
    if missing_zone.any():
        warnings.warn(
            f"No pGenDataInputCustom.csv entry at all for storage plant(s) in {data_path!r} "
            f"(no zone/tech to attach them with): {sorted(storage_constants.loc[missing_zone, 'storage'])}"
        )

    # fill missing lifetimes (some regions have no "Life" value at all for any storage
    # plant) with a default by tech.
    default_lifetime = {
        "Battery": 15,
        "PumpedHydro": 50,
    }
    missing_life = storage_constants["life"].isna()
    storage_constants.loc[missing_life, "life"] = storage_constants.loc[missing_life, "tech"].map(default_lifetime)

    # fill in the remaining GAMS parameters (same names generator_constants pulls)
    gams_param_list = ["Capacity", "RetrYr", "StYr", "UnitSize", "BuildLimitperYear"]
    for param in gams_param_list:
        param_values = storage_gen_params[storage_gen_params["parameters"] == param].set_index("generator")["value"]
        storage_constants[param] = storage_constants["storage"].map(param_values)

    # build option if BuildLimitperYear is not nan - same signal generator_constants uses
    storage_constants["expandable"] = storage_constants["BuildLimitperYear"].notna()
    # rename to seer-schema column names, matching generator_constants conventions
    storage_constants.rename(columns={
        "Capacity": "capacity_mw",
        "RetrYr": "retirement_year",
        "StYr": "start_year",
        "UnitSize": "unit_size_mwh",
        "BuildLimitperYear": "annual_build_limit_mwh",
    }, inplace=True)
    # no RetrYr value found means "no known retirement" -- default to a sentinel year well
    # beyond any realistic modeled horizon, not NaN (see generator_params.py for why: a NaN
    # retirement_year makes every `year <= retirement_year` check False, silently dropping
    # the unit from every modeled year instead of leaving it active for the full horizon).
    storage_constants["retirement_year"] = (
        storage_constants["retirement_year"].infer_objects(copy=False).fillna(NO_RETIREMENT_YEAR)
    )

    storage_constants["duration_hours"] = storage_constants["capacity_mwh"] / storage_constants["capacity_mw"]

    """Setup storage costs.
    Use storage constants to fill in a single dataframe of storage costs.
    """
    rows = []

    for _, storage in storage_constants.iterrows():

        row_capex = {"storage": storage.storage, "cost_type": "capex"}
        for y in years:
            row_capex[y] = storage.capex_mwh
        rows.append(row_capex)

    storage_costs = pd.DataFrame(rows, columns=["storage", "cost_type"] + years)

    # reorder to match the seer schema. capacity_mwh/capex_mwh are dropped here -- they were
    # only needed to compute duration_hours and the capex trajectory above; energy capacity is
    # derived as capacity_mw * duration_hours wherever it's needed downstream.
    storage_constants = storage_constants[[
        "storage", "zone", "tech", "duration_hours", "capacity_mw",
        "efficiency", "life", "start_year", "retirement_year",
        "unit_size_mwh", "annual_build_limit_mwh", "expandable",
    ]]

    return storage_constants, storage_costs
