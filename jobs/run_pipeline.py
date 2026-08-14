"""
Run full pipeline to solve the model: 
gdx -> csv -> seer schema -> pypsa solve

Usage:
    python -m jobs.run_experiment Gambia
    python -m jobs.run_experiment Gambia --solver highs --time-resolution 1h
"""

import argparse

from jobs.epm_to_csv import epm_to_csv
from jobs.epm_to_seer import epm_to_seer
from jobs.solve_pypsa_model import solve_pypsa_model

def run_pipeline(region, strict=True, time_resolution="1h", solver_name="highs"):
    # Convert GAMS file to csvs
    epm_to_csv(region=region, strict=strict)

    # Convert csvs to seer schema
    epm_to_seer(region)

    # Solve the model
    return solve_pypsa_model(
        region,
        time_resolution=time_resolution,
        solver_name=solver_name
    )

def main():
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("region", help="Region/country name, e.g. Gambia — must match a folder under gdx/data/epm_gams")
    # parser.add_argument("--solver", default="highs", help="PyPSA solver name (default: highs)")
    # parser.add_argument("--time-resolution", default="1h", help="Snapshot time resolution (default: 1h)")
    # parser.add_argument("--no-strict", action="store_true", help="Disable strict symbol-mapping validation in the gdx step")
    # args = parser.parse_args()

    run_pipeline(
        region="Rwanda",
        strict=True,
        time_resolution="1h",
        solver_name="highs",
    )

if __name__ == "__main__":
    main()


