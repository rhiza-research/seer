"""Convert csvs specifying EPM (GAMS) input data into the seer schema."""

from pathlib import Path

import pandas as pd

from .process_generator_data import process_generator_data
from .process_re_data import process_re_data
from .process_storage_data import process_storage_data


def process_passthrough_data(data_path):
    """Files that need only trivial reshaping to match the seer schema:
    y.csv -> years, zcmap.csv -> zones, pDemandForecast/pDemandProfile -> demandforecast/demandprofile
    (already in seer-schema column form)."""
    years = pd.read_csv(f"{data_path}y.csv")[["year"]]

    zones = pd.read_csv(f"{data_path}zcmap.csv")[["zone", "country"]]
    zones["name"] = zones["zone"]

    demandforecast = pd.read_csv(f"{data_path}pDemandForecast.csv")
    demandprofile = pd.read_csv(f"{data_path}pDemandProfile.csv")

    return years, zones, demandforecast, demandprofile


def gams_to_seer(input_path, output_path, start_year=2025, end_year=2050):
    """Run all conversion steps to turn EPM (GAMS) model specification to seer schema
    seer schema files are written to the output path. 
    returns a dict of all the dataframes, also written to csvs.
    """

    # process_*_data build paths via `data_path + "file.csv"`, so it must end in "/"
    input_dir = f"{Path(input_path)}/"
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_constants, generator_costs = process_generator_data(input_dir)
    renewable_profiles = process_re_data(input_dir)
    storage_constants, storage_costs = process_storage_data(input_dir, start_year=start_year, end_year=end_year)
    years, zones, demandforecast, demandprofile = process_passthrough_data(input_dir)

    outputs = {
        "generator_constants": generator_constants,
        "generator_costs": generator_costs,
        "renewable_profiles": renewable_profiles,
        "storage_constants": storage_constants,
        "storage_costs": storage_costs,
        "years": years,
        "zones": zones,
        "demandforecast": demandforecast,
        "demandprofile": demandprofile,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    return outputs
