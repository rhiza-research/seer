"""Convert csvs specifying EPM input data into the seer schema."""

from pathlib import Path
import pandas as pd

from ._time_convert import convert_season_day_hour
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
    # times.csv: the canonical dispatch time grid, taken directly from demand_profile's own
    # `time` values (EPM data is often already a handful of representative days per month, not
    # a full year -- see seer/inputs/README.md's times.csv entry)
    times = pd.DataFrame({"time": sorted(demand_profile["time"].unique())})

    outputs = {
        "generator_constants": generator_constants,
        "generator_costs": generator_costs,
        "renewable_profiles": renewable_profiles,
        "storage_constants": storage_constants,
        "storage_costs": storage_costs,
        "years": years,
        "times": times,
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
    y.csv -> years, zcmap.csv -> zones, pDemandForecast -> demand_forecast (already in
    seer-schema column form); pDemandProfile's season/day/hour columns are converted to a
    single `time` column -> demand_profile."""
    years = pd.read_csv(f"{data_path}y.csv")[["year"]]

    demand_forecast = pd.read_csv(f"{data_path}pDemandForecast.csv")
    demand_profile = pd.read_csv(f"{data_path}pDemandProfile.csv")
    demand_profile = convert_season_day_hour(demand_profile)[["zone", "time", "value"]]

    zcmap_path = Path(f"{data_path}zcmap.csv")
    if zcmap_path.exists():
        zones = pd.read_csv(zcmap_path)[["zone", "country"]]
    else:
        # Some regions' raw gdx has no zcmap symbol at all. Only "zone" drives spatial
        # resolution (buses) in either model backend -- "country" isn't consumed anywhere
        # -- so fall back to the zone list implied by demand data and stub in a placeholder
        # country rather than failing.
        zones = pd.DataFrame({"zone": sorted(demand_forecast["zone"].unique())})
        zones["country"] = "default"
    zones["name"] = zones["zone"]

    return years, zones, demand_forecast, demand_profile
