"""Build a PyPSA model from seer-schema data."""

from seer.models.pypsa_model import build, solve, print_summary


def main():
    solve_pypsa_model("Gambia")


def solve_pypsa_model(region, time_resolution="1h", solver_name="highs"):

    DATA_PATH = f"data/seer/{region}/"

    model = build(DATA_PATH, time_resolution=time_resolution)

    print(f"Buses (zones): {list(model.buses.index)}")
    print(f"Investment periods: {list(model.investment_periods)}")
    print(f"Snapshots: {len(model.snapshots)}")
    print(f"Generators: {len(model.generators)}")
    print(f"Storage units: {len(model.storage_units)}")
    print(f"Loads: {list(model.loads.index)}")
    print(f"Carriers: {list(model.carriers.index)}")

    solve(model, solver_name=solver_name)
    print_summary(model)

if __name__ == "__main__":
    
    main()
