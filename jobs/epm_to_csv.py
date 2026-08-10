"""Convert EPM GAMS .gdx inputs into per-region, per-symbol CSVs"""

from pathlib import Path

from gdx import gams_to_csv

def epm_to_csv(strict=True):

    data_path = Path("gdx/data/epm_gams")
    mapping_path = Path("gdx/data/symbol_mapping.csv")
    output_path = Path("gdx/data/epm_gams_csv")

    gams_to_csv(
        data_dir=data_path,
        mapping_path=mapping_path,
        output_dir=output_path,
        strict=strict
        )

    print(f"Wrote CSVs for all regions in {data_path} to {output_path}")

if __name__ == "__main__":
    epm_to_csv()