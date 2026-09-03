"""Convert csvs specifying EPM input data into the seer schema."""

from pathlib import Path
import pandas as pd

from .generator_params import build_generator_params
from .re_potentials import build_re_potentials
from .storage_params import build_storage_params


def epm_to_seer(input_path, output_path, start_year=2025, end_year=2050):
    """Run all conversion steps to turn EPM model specification to seer schema
    seer schema files are written to the output path. 
    returns a dict of all the dataframes, also written to csvs.
    """

    # process_*_data build paths via `data_path + "file.csv"`, so it must end in "/"
    input_dir = f"{Path(input_path)}/"
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_constants, generator_costs = build_generator_params(input_dir)
    renewable_profiles = build_re_potentials(input_dir)
    storage_constants, storage_costs = build_storage_params(input_dir, start_year=start_year, end_year=end_year)
    years, zones, demand_forecast, demand_profile = passthrough(input_dir)

    outputs = {
        "generator_constants": generator_constants,
        "generator_costs": generator_costs,
        "renewable_profiles": renewable_profiles,
        "storage_constants": storage_constants,
        "storage_costs": storage_costs,
        "years": years,
        "zones": zones,
        "demand_forecast": demand_forecast,
        "demand_profile": demand_profile,
    }
    # save seer schema csv files to output directory
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    return outputs


def passthrough(data_path):
    """Files that need only trivial reshaping to match the seer schema:
    y.csv -> years, zcmap.csv -> zones, pDemandForecast/pDemandProfile -> demand_forecast/demand_profile
    (already in seer-schema column form)."""
    years = pd.read_csv(f"{data_path}y.csv")[["year"]]

    zones = pd.read_csv(f"{data_path}zcmap.csv")[["zone", "country"]]
    zones["name"] = zones["zone"]

    demand_forecast = pd.read_csv(f"{data_path}pDemandForecast.csv")
    demand_profile = pd.read_csv(f"{data_path}pDemandProfile.csv")

    return years, zones, demand_forecast, demand_profile
