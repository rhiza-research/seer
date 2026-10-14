"""Convert hand-cleaned EPM csvs into seer-schema data, region by region.

This is stage 2 of the gdx -> seer-schema pipeline: data/processed_gams (a hand-cleaned copy
of data/extracted_gams -- see data/processed_gams/README.md for what was changed and why,
region by region) -> seer-schema csvs. Stage 1 (jobs/extract_gams.py) does the raw .gdx ->
extracted_gams extraction; this script never touches extracted_gams or raw_gams, and never
writes to processed_gams -- that folder is maintained by hand.

Usage:
    python -m jobs.epm_to_seer                       # process all regions under data/processed_gams
    python -m jobs.epm_to_seer Gambia Rwanda          # process just these regions
    python -m jobs.epm_to_seer --processed-gams-dir data/other --seer-output-dir data/seer/other

!! DANGER -- regions with a hand-patched generator_costs.csv !!
A few regions have marginal_cost values that were computed and written by hand directly into
their seer-schema output (not reproducible by this script), because their own source data has
no real price for some fuel and the fix was to borrow a price from a comparable sibling region
or (Serbia) re-derive it from a differently-shaped Fuel Prices sheet. Re-running this script
for one of these regions SILENTLY OVERWRITES that patch back to the broken (NaN) values, with
no error -- it already happened once by accident (a no-args full sweep regenerated Mauritania's
output while fixing an unrelated region, wiping out its patch without anyone noticing until the
next validation pass). As of this writing the affected regions are:
    - Serbia (gams)                 -- see data/processed_gams/README.md
    - Rwanda, CAPP, Tanzania (gams)  -- see data/processed_gams/README.md
    - Mauritania (epm_open)         -- see data/processed_epm_open/README.md
Before running this script for ALL regions (no region arguments) or for any of the above by
name, re-check those README sections for the current list and re-apply the patch afterward if
needed. There is currently no automated guard against this -- the warning below is it.
"""

import argparse
import sys
from pathlib import Path

from epm2seer import epm_to_seer as convert_epm_to_seer

# keep in sync with the module docstring above and with the "re-running ... will NOT reproduce
# this" notes in data/processed_gams/README.md and data/processed_epm_open/README.md
MANUALLY_PATCHED_REGIONS = {"Serbia", "Rwanda", "CAPP", "Tanzania", "Mauritania"}


def epm_to_seer(region, processed_gams_dir, seer_output_dir):
    """Convert one region's hand-cleaned EPM csvs (data/processed_gams/<region>) into
    seer-schema csvs."""
    input_path = Path(processed_gams_dir) / region
    output_path = Path(seer_output_dir) / region
    convert_epm_to_seer(input_path, output_path)
    print(f"[{region}] wrote seer-schema data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("regions", nargs="*", help="Region names to process (default: all folders under --processed-gams-dir)")
    parser.add_argument("--processed-gams-dir", type=Path, default=Path("data/processed_gams"))
    parser.add_argument("--seer-output-dir", type=Path, default=Path("data/seer/from_gams"))
    args = parser.parse_args()

    regions = args.regions or sorted(
        d.name for d in args.processed_gams_dir.iterdir() if d.is_dir()
    )

    at_risk = MANUALLY_PATCHED_REGIONS & set(regions)
    if at_risk:
        print(
            f"!! WARNING: {sorted(at_risk)} have a hand-patched generator_costs.csv (see the "
            "module docstring above). Running this will silently overwrite that patch with "
            "broken (NaN) marginal_cost values. Re-apply it afterward if you proceed.",
            file=sys.stderr,
        )

    results = {}
    for region in regions:
        try:
            epm_to_seer(region, processed_gams_dir=args.processed_gams_dir, seer_output_dir=args.seer_output_dir)
            results[region] = "ok"
        except Exception as e:
            results[region] = f"FAILED: {e}"
            print(f"[{region}] FAILED: {e}")

    print("\n--- summary ---")
    for region, status in results.items():
        print(f"{region}: {status}")

    n_failed = sum(1 for status in results.values() if status != "ok")
    if n_failed:
        print(f"\n{n_failed}/{len(results)} region(s) failed.")
    else:
        print(f"\nAll {len(results)} region(s) converted successfully.")


if __name__ == "__main__":
    main()
