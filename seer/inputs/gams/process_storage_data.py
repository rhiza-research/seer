import pandas as pd


def process_storage_data(data_path="data/Gambia/", start_year=2025, end_year=2050):
    """
    Processes storage input data into:
    - storage_constants.csv
    - storage_costs.csv
    """

    storage_data = pd.read_csv(data_path + "pStorageDataInput.csv")

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


if __name__ == "__main__":

    data_path = "gams2pypsa/data/processed/Gambia/"

    storage_constants, storage_costs = process_storage_data(data_path)

    storage_constants.to_csv(
        data_path + "storage_constants.csv",
        index=False
    )

    storage_costs.to_csv(
        data_path + "storage_costs.csv",
        index=False
    )