"""Extract EPM GAMS .gdx inputs into per-symbol CSVs, region by region.

This is stage 1 of the gdx -> seer-schema pipeline: raw .gdx -> data/extracted_gams csvs.
Always safe to (re)run -- output here is fully regenerated from the raw .gdx and never
hand-edited. Stage 2 (jobs/epm_to_seer.py) reads from data/processed_gams instead, which starts
as a copy of this stage's output but is then hand-cleaned per region (see
data/processed_gams/README.md for what was changed and why) -- so re-running this script
never touches processed_gams.

Usage:
    python -m jobs.extract_gams                          # process all regions under data/raw_gams
    python -m jobs.extract_gams Gambia Rwanda             # process just these regions
    python -m jobs.extract_gams --raw-gams-dir data/other_gams --extracted-gams-dir data/other
    python -m jobs.extract_gams --no-strict               # disable strict symbol-mapping validation
"""

import argparse
from pathlib import Path

from gdx import gams_to_csv

MAPPING_PATH = Path("gdx/data/symbol_mapping.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("regions", nargs="*", help="Region names to process (default: all folders under --raw-gams-dir)")
    parser.add_argument("--raw-gams-dir", type=Path, default=Path("data/raw_gams"))
    parser.add_argument("--mapping-path", type=Path, default=MAPPING_PATH)
    parser.add_argument("--extracted-gams-dir", type=Path, default=Path("data/extracted_gams"))
    parser.add_argument("--no-strict", action="store_true", help="Disable strict symbol-mapping validation")
    args = parser.parse_args()

    regions = args.regions or sorted(
        d.name for d in args.raw_gams_dir.iterdir() if d.is_dir()
    )

    results = {}
    for region in regions:
        try:
            gams_to_csv(
                data_dir=args.raw_gams_dir,
                mapping_path=args.mapping_path,
                output_dir=args.extracted_gams_dir,
                strict=not args.no_strict,
                region=region,
            )
            print(f"[{region}] wrote extracted csvs to {args.extracted_gams_dir / region}")
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
        print(f"\nAll {len(results)} region(s) extracted successfully.")


if __name__ == "__main__":
    main()
