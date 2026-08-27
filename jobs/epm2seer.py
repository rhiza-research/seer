"""Convert EPM GAMS .gdx inputs directly into seer-schema data, region by region.

Processes every folder under raw-gams-dir by default (each must contain an input.gdx),
writing matching-named output folders under seer-output-dir.

Usage:
    python -m jobs.epm2seer                       # process all regions under data/raw_gams
    python -m jobs.epm2seer Gambia Rwanda          # process just these regions
    python -m jobs.epm2seer --raw-gams-dir data/other_gams --seer-output-dir data/seer/other
    python -m jobs.epm2seer --save-epm-csv         # also keep the intermediate EPM csvs
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from gdx import gams_to_csv
from seer.inputs.epm import epm_to_seer as convert_epm_to_seer

MAPPING_PATH = Path("gdx/data/symbol_mapping.csv")


def epm2seer(region, raw_gams_dir, seer_output_dir, mapping_path=MAPPING_PATH,
             epm_csv_dir=None, save_epm_csv=False, strict=True):
    """Convert one region's raw GAMS .gdx input all the way to seer-schema csvs.

    The intermediate per-symbol EPM csvs (gdx -> csv step) are only needed transiently:
    if epm_csv_dir isn't given, they're written to a temporary directory that's deleted
    afterward, unless save_epm_csv=True, in which case that temp directory is kept.
    If epm_csv_dir is given explicitly, it's always kept (never auto-deleted).
    """
    tmp_dir = None
    if epm_csv_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="epm_csv_")
        epm_csv_dir = Path(tmp_dir)

    try:
        # gdx -> csv (per-symbol EPM csvs, intermediate)
        gams_to_csv(
            data_dir=raw_gams_dir,
            mapping_path=mapping_path,
            output_dir=epm_csv_dir,
            strict=strict,
            region=region,
        )

        # csv -> seer schema
        input_path = epm_csv_dir / region
        output_path = Path(seer_output_dir) / region
        convert_epm_to_seer(input_path, output_path)

        print(f"[{region}] wrote seer-schema data to {output_path}")
    finally:
        if tmp_dir is not None and not save_epm_csv:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("regions", nargs="*", help="Region names to process (default: all folders under --raw-gams-dir)")
    parser.add_argument("--raw-gams-dir", type=Path, default=Path("data/raw_gams"))
    parser.add_argument("--mapping-path", type=Path, default=MAPPING_PATH)
    parser.add_argument("--seer-output-dir", type=Path, default=Path("data/seer/from_gams"))
    parser.add_argument("--epm-csv-dir", type=Path, default=Path("data/processed_gams"),
                         help="Where to write intermediate EPM csvs (default: data/processed_gams, overwritten each run)")
    parser.add_argument("--save-epm-csv", action="store_true", help="Keep the intermediate EPM csvs instead of deleting them")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict symbol-mapping validation")
    args = parser.parse_args()

    regions = args.regions or sorted(
        d.name for d in args.raw_gams_dir.iterdir() if d.is_dir()
    )

    results = {}
    for region in regions:
        try:
            epm2seer(
                region,
                raw_gams_dir=args.raw_gams_dir,
                seer_output_dir=args.seer_output_dir,
                mapping_path=args.mapping_path,
                epm_csv_dir=args.epm_csv_dir,
                save_epm_csv=args.save_epm_csv,
                strict=not args.no_strict,
            )
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
