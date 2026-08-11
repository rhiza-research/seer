"""Build storage_constants.csv and storage_costs.csv from EPM input data."""
import pandas as pd


def build_storage_params(data_path="data/Gambia/", start_year=2025, end_year=2050):
    """
    Processes storage input data into two seer schema files:
    - storage_constants.csv
    - storage_costs.csv

    Files that are read containing storage data:
    - pStorageDataInput.csv: storage data (by plant)
    - pGenDataInputCustom.csv: generator parameters (by generator)
    Returns two dataframes: storage_constants and storage_costs.
    """

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

    # select and rename parameters
    param_list = ["plant", "CapacityMWh", "CapexMWh", "Efficiency", "Life"]
    param_names = ["storage", "capacity_mwh", "capex_mwh", "efficiency", "life"]
    storage_constants = storage_constants[param_list].rename(columns=dict(zip(param_list, param_names)))

    """Pull the remaining attributes from the generator custom-parameters file - storage plants
    are just generators with tech="Storage" there - matched here by name, same as generator_constants.
    """
    storage_gen_params = custom_gen_params[custom_gen_params["generator"].isin(storage_constants["storage"])]

    # zone and carrier (fuel) are plain columns rather than "parameters" rows
    zone_carrier = storage_gen_params[["generator", "zone", "fuel"]].drop_duplicates().rename(
        columns={"generator": "storage", "fuel": "carrier"}
    )
    storage_constants = storage_constants.merge(zone_carrier, on="storage", how="left")

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

    storage_constants["duration_hours"] = storage_constants["capacity_mwh"] / storage_constants["capacity_mw"]

    # reorder to match the seer schema
    storage_constants = storage_constants[[
        "storage", "zone", "carrier", "capacity_mwh", "duration_hours", "capacity_mw",
        "capex_mwh", "efficiency", "life", "start_year", "retirement_year",
        "unit_size_mwh", "annual_build_limit_mwh", "expandable",
    ]]

    """Setup storage costs.
    Use storage constants to fill in a single dataframe of storage costs.
    """
    years = list(range(start_year, end_year + 1))

    rows = []

    for _, storage in storage_constants.iterrows():

        row_capex = {"storage": storage.storage, "cost_type": "capex"}
        for y in years:
            row_capex[y] = storage.capex_mwh
        rows.append(row_capex)

    storage_costs = pd.DataFrame(rows, columns=["storage", "cost_type"] + years)

    return storage_constants, storage_costs
