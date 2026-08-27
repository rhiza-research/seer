"""End-to-end experiment: go from a raw GAMS .gdx input all the way to solved PyPSA
output tables.

    gdx (.gdx) -> csv (per-symbol EPM csvs) -> seer schema (csv) -> pypsa model -> solve -> output tables

To point this at a different country/region, just change REGION below to any folder
name that exists under GDX_DATA_DIR (i.e. a folder containing an `input.gdx`).

Usage:
    python -m experiments.seer_from_gdx
"""

from pathlib import Path

import gdx
from seer.inputs.epm import epm_to_seer
from seer.models.pypsa_model import build, solve, print_summary, build_output_tables, build_storage_soc

# ---------------------------------------------------------------------------
# Change this to run a different region (must match a folder under GDX_DATA_DIR
# containing an input.gdx)
REGION = "Gambia"
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# all intermediate/output data for this experiment is kept local to experiments/,
# separate from the repo's tracked gdx/data and data/seer directories
TMP_DATA_DIR = REPO_ROOT / "experiments" / "tmp_data" / "seer_from_gdx"

GDX_DATA_DIR = REPO_ROOT / "gdx" / "data" / "epm_gams"          # <GDX_DATA_DIR>/<REGION>/input.gdx (read-only source)
SYMBOL_MAPPING_PATH = REPO_ROOT / "gdx" / "data" / "symbol_mapping.csv"
EPM_CSV_DIR = TMP_DATA_DIR / "epm_gams_csv"                     # step 1 output (per-symbol EPM csvs)
SEER_SCHEMA_DIR = TMP_DATA_DIR / "seer" / REGION                # step 2 output (seer-schema csvs)
OUTPUT_DIR = TMP_DATA_DIR / "seer" / REGION / "outputs"         # step 4 output (solved model tables)

TIME_RESOLUTION = "1h"
SOLVER_NAME = "highs"


def main():

    # --- Step 1: gdx -> csv -----------------------------------------------
    # Load the raw GAMS .gdx input, align symbol names/columns to the canonical
    # schema, enrich fuel/tech/gen/VRE data, and write one csv per symbol.
    print(f"[1/4] Converting {REGION} .gdx -> per-symbol EPM csvs ...")
    gdx.gams_to_csv(
        data_dir=GDX_DATA_DIR,
        mapping_path=SYMBOL_MAPPING_PATH,
        output_dir=EPM_CSV_DIR,
        strict=True,
        region=REGION,
    )
    print(f"       wrote EPM csvs to {EPM_CSV_DIR / REGION}")

    # --- Step 2: csv -> seer schema -----------------------------------------
    # Convert the EPM-schema csvs into the seer schema (generator/storage
    # constants & costs, renewable profiles, demand, years, zones).
    print(f"[2/4] Converting EPM csvs -> seer schema ...")
    epm_to_seer(EPM_CSV_DIR / REGION, SEER_SCHEMA_DIR)
    print(f"       wrote seer-schema csvs to {SEER_SCHEMA_DIR}")

    # --- Step 3: seer schema -> pypsa model -> solve ------------------------
    # Instantiate a PyPSA network from the seer-schema data and solve it.
    print(f"[3/4] Building and solving PyPSA model ...")
    model = build(f"{SEER_SCHEMA_DIR}/", time_resolution=TIME_RESOLUTION)
    print(f"       Buses (zones): {list(model.buses.index)}")
    print(f"       Investment periods: {list(model.investment_periods)}")
    print(f"       Snapshots: {len(model.snapshots)}")
    print(f"       Generators: {len(model.generators)}")
    print(f"       Storage units: {len(model.storage_units)}")

    solve(model, solver_name=SOLVER_NAME)
    print_summary(model)

    # --- Step 4: solved model -> seer output files --------------------------
    # Extract output tables (generator/storage capacity, dispatch, storage SOC)
    # from the solved model and write them to csv.
    print(f"[4/4] Writing seer output files ...")
    output_tables = build_output_tables(model)
    storage_soc = build_storage_soc(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in output_tables.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    storage_soc.to_csv(OUTPUT_DIR / "storage_soc.csv", index=False)
    print(f"       wrote output csvs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
