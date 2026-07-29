"""Build a PyPSA model from seer-schema data."""

from seer.build.pypsa_model import build_model

DATA_PATH = "data/seer/Gambia/"


def main():
    model = build_model(time_resolution="1h", data_path=DATA_PATH)

    print(f"Buses (zones): {list(model.buses.index)}")
    print(f"Investment periods: {list(model.investment_periods)}")
    print(f"Snapshots: {len(model.snapshots)}")
    print(f"Generators: {len(model.generators)}")
    print(f"Storage units: {len(model.storage_units)}")
    print(f"Loads: {list(model.loads.index)}")
    print(f"Carriers: {list(model.carriers.index)}")


if __name__ == "__main__":
    main()
