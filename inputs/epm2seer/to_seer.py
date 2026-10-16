"""Convert csvs specifying EPM input data into the seer schema."""

from pathlib import Path
import warnings

import pandas as pd

from .utils import convert_season_day_hour
from .generator_params import build_generator_params
from .storage_params import build_storage_params
from .network_params import build_network
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

    """zones"""
    zones = check_zones(pd.read_csv(f"{input_dir}zcmap.csv"))
    # default name for a zone is the zone itself
    zones["name"] = zones["zone"]

    """demand forecast & profile"""
    demand_forecast = check_demand_forecast(pd.read_csv(f"{input_dir}pDemandForecast.csv"), zones["zone"])
    demand_profile = pd.read_csv(f"{input_dir}pDemandProfile.csv")
    demand_profile = convert_season_day_hour(demand_profile)
    # times.csv: the canonical dispatch time grid, taken directly from demand_profile's own
    # `time` values (EPM data is often already a handful of representative days per month, not
    # a full year -- see seer/schema/README.md's times.csv entry). "MM-DD HH:MM:SS" sorts
    # chronologically: months in order, each day's hours contiguous.
    times = pd.DataFrame({"time": sorted(demand_profile["time"].unique())})
    # each representative day's weight (EPM's Duration: e.g. 1 for a peak day, ~30 for a
    # typical weekday standing for the rest of the month) -- relative, scaled to a full year by
    # the models. No weight in the input means uniform (column left out).
    if "weight" in demand_profile.columns and demand_profile["weight"].notna().any():
        per_time = demand_profile.groupby("time")["weight"]
        if (per_time.nunique() > 1).any():
            warnings.warn("pDemandProfile.csv: weight differs between zones for the same time -- using the first")
        times["weight"] = times["time"].map(per_time.first())
    demand_profile = demand_profile[["zone", "time", "value"]]
    demand_profile = check_demand_profile(demand_profile, zones["zone"])
    demand_profile = demand_profile.sort_values(["zone", "time"]).reset_index(drop=True)

    """network"""
    network = build_network(input_dir, zones["zone"])

    """settings"""
    settings = check_settings(pd.read_csv(f"{input_dir}pSettings.csv"))

    """build generator params"""
    generator_constants, generator_costs, renewable_profiles = build_generator_params(
        input_dir, zones["zone"], years["year"].tolist(), times["time"].tolist()
    )

    """build storage params"""
    storage_constants, storage_costs = build_storage_params(
        input_dir, years=years["year"].tolist(), valid_zones=zones["zone"]
    )

    # profiles in chronological order within each profile (see times above)
    renewable_profiles = renewable_profiles.sort_values(["profile_id", "time"]).reset_index(drop=True)

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
        # 3 decimal places is plenty for every quantity in the schema (MW, $/MW, $/MWh, capacity
        # factors, fractions) -- keeps the csvs readable
        float_cols = df.select_dtypes("float").columns
        df = df.copy()
        df[float_cols] = df[float_cols].round(3)
        df.to_csv(output_dir / f"{name}.csv", index=False)

    return outputs
