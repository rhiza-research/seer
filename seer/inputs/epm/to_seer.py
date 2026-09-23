"""Convert csvs specifying EPM input data into the seer schema."""

from pathlib import Path
import pandas as pd

from .utils import convert_season_day_hour
from .generator_params import build_generator_params
from .storage_params import build_storage_params
from .data_checks import *


def epm_to_seer(input_path, output_path):
    """Run all conversion steps to turn EPM model specification to seer schema
    seer schema files are written to the output path.
    returns a dict of all the dataframes, also written to csvs.
    """

    # set input and output directories
    input_dir = f"{Path(input_path)}/"
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    """modelled years"""
    years = pd.read_csv(f"{input_dir}y.csv")[["year"]]

    """demand forecast & profile"""
    demand_forecast = check_demand_forecast(pd.read_csv(f"{input_dir}pDemandForecast.csv"))
    demand_profile = pd.read_csv(f"{input_dir}pDemandProfile.csv")
    demand_profile = convert_season_day_hour(demand_profile)[["zone", "time", "value"]]
    # times.csv: the canonical dispatch time grid, taken directly from demand_profile's own
    # `time` values (EPM data is often already a handful of representative days per month, not
    # a full year -- see seer/inputs/README.md's times.csv entry)
    times = pd.DataFrame({"time": sorted(demand_profile["time"].unique())})

    """zones"""
    zones = check_zones(pd.read_csv(f"{input_dir}zcmap.csv"))
    # default name for a zone is the zone itself
    zones["name"] = zones["zone"]

    """network"""
    network_path = Path(f"{input_dir}pTransferLimit.csv")
    if network_path.exists():
        network = check_network(pd.read_csv(network_path), zones["zone"])
        # original data has time varying capacities, but seer has static -> use average
        network = network.groupby(["from", "to"])["value"].mean().reset_index().rename(columns={"value": "capacity"})
    else:
        # if there is no network, create a placeholder empty network
        network = pd.DataFrame(columns=["from", "to", "capacity"])
    
    """settings"""
    settings = check_settings(pd.read_csv(f"{input_dir}pSettings.csv"))

    """build generator params"""
    generator_constants, generator_costs, renewable_profiles = build_generator_params(
        input_dir, zones["zone"], years["year"].tolist(), times["time"].tolist()
    )

    # storage_costs.csv must carry a column for every modeled year (years.csv) -- build.py's
    # per-vintage capex lookup indexes storage_costs by those exact values (see
    # storage_params.py's build_storage_params docstring)
    storage_constants, storage_costs = build_storage_params(
        input_dir, years=years["year"].tolist(), valid_zones=zones["zone"]
    )

    outputs = {
        "generator_constants": generator_constants,
        "generator_costs": generator_costs,
        "renewable_profiles": renewable_profiles,
        "storage_constants": storage_constants,
        "storage_costs": storage_costs,
        "years": years,
        "times": times,
        "zones": zones,
        "network": network,
        "settings": settings,
        "demand_forecast": demand_forecast,
        "demand_profile": demand_profile,
    }
    # save seer schema csv files to output directory
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    return outputs
