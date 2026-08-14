"""Convert models from EPM schema into SEER schema"""

from pathlib import Path

from seer.inputs.epm import epm_to_seer as convert_epm_to_seer


def epm_to_seer(model_name):
    input_path = Path("gdx/data/epm_gams_csv") / model_name
    output_path = Path("data/seer") / model_name

    convert_epm_to_seer(input_path, output_path)
    print(f"Wrote seer-schema data for {model_name} to {output_path}")


if __name__ == "__main__":
    model_name = "Gambia"
    epm_to_seer(model_name)
