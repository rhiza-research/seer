"""Convert EPM schema into SEER"""

from pathlib import Path

from seer.inputs.gams import gams_to_seer

MODEL_NAME = "Gambia"  # change this to convert a different model's data


def main():
    input_path = Path("data/processed_gams") / MODEL_NAME
    output_path = Path("data/seer") / MODEL_NAME

    gams_to_seer(input_path, output_path)
    print(f"Wrote seer-schema data for {MODEL_NAME} to {output_path}")


if __name__ == "__main__":
    main()
