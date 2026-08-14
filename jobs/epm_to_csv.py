"""Convert EPM GAMS .gdx inputs into per-region, per-symbol CSVs"""

from pathlib import Path

from gdx import gams_to_csv

def epm_to_csv(region=None, strict=True):

    data_path = Path("gdx/data/epm_gams")
    mapping_path = Path("gdx/data/symbol_mapping.csv")
    output_path = Path("gdx/data/epm_gams_csv")

    gams_to_csv(
        data_dir=data_path,
        mapping_path=mapping_path,
        output_dir=output_path,
        strict=strict,
        region=region
        )

    target = output_path / region if region else output_path
    if region and not target.exists():
        raise FileNotFoundError(
            f"No CSVs were written for region '{region}' — check that "
            f"{data_path / region / 'input.gdx'} exists."
        )

    print(f"Wrote CSVs for {region or 'all regions'} in {data_path} to {output_path}")

if __name__ == "__main__":
    epm_to_csv()