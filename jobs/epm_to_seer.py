"""Convert models from EPM schema into SEER schema"""

from pathlib import Path

from seer.inputs.gams import gams_to_seer


def epm_to_seer(model_name):
    input_path = Path("data/processed_gams") / model_name
    output_path = Path("data/seer") / model_name

    gams_to_seer(input_path, output_path)
    print(f"Wrote seer-schema data for {model_name} to {output_path}")


if __name__ == "__main__":
    model_name = "Gambia"
    epm_to_seer(model_name)
