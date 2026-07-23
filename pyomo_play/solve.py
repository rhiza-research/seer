import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

from model import seer_base_model
from plot import plot_demand, plot_generation

INPUT_DIR = Path(__file__).parent / "input_data"
SOLVER_NAME = "appsi_highs"  # swap for "cbc" / "glpk" / "gurobi" if not installed


def load_data():
    """Read the seer-schema input CSVs and build a Pyomo AbstractModel data dict."""

    years_df = pd.read_csv(INPUT_DIR / "years.csv")
    zones_df = pd.read_csv(INPUT_DIR / "zones.csv")
    generator_constants_df = pd.read_csv(INPUT_DIR / "generator_constants.csv")
    generator_costs_df = pd.read_csv(INPUT_DIR / "generator_costs.csv")
    demandforecast_df = pd.read_csv(INPUT_DIR / "demandforecast.csv")
    demandprofile_df = pd.read_csv(INPUT_DIR / "demandprofile.csv")
    renewable_profiles_df = pd.read_csv(INPUT_DIR / "renewable_profiles.csv")

    years = years_df["year"].tolist()
    zones = zones_df["zone"].tolist()
    generators = generator_constants_df["generator"].tolist()  # local convenience list, not passed to Pyomo directly

    # hourly index: derived from the demand profile's per-zone row count (e.g. 365 days * 24 hours)
    hours_per_zone = len(demandprofile_df) // len(zones)
    hours = list(range(1, hours_per_zone + 1))

    # ---- demand: annual peak (demandforecast) x normalized hourly shape (demandprofile) ----
    demand_shape = {
        (zone, hour): value
        for zone, group in demandprofile_df.groupby("zone", sort=False)
        for hour, value in zip(hours, group["value"])
    }
    peak_demand = {
        (row.zone, int(row.year)): row.value
        for row in demandforecast_df.itertuples()
    }
    demand = {
        (zone, year, hour): peak_demand[(zone, year)] * demand_shape[(zone, hour)]
        for zone in zones
        for year in years
        for hour in hours
    }

    # ---- renewable availability, keyed by (zone, tech) ----
    renewable_availability = {
        (zone, tech, hour): value
        for (zone, tech), group in renewable_profiles_df.groupby(["zone", "tech"], sort=False)
        for hour, value in zip(hours, group["value"])
    }

    # ---- generator attributes ----
    constants = generator_constants_df.set_index("generator")
    existing_generators = constants[~constants["expandable"]].index.tolist()
    new_generators = constants[constants["expandable"]].index.tolist()

    generator_zone = constants["zone"].to_dict()
    generator_tech = constants["tech"].to_dict()
    generator_retech = constants["RETech"].astype(bool).to_dict()

    # only meaningful (and only loaded) for existing generators
    generator_capacity = constants.loc[existing_generators, "capacity"].to_dict()
    generator_start_year = constants.loc[existing_generators, "start_year"].to_dict()
    generator_retirement_year = constants.loc[existing_generators, "retirement_year"].to_dict()
    # only meaningful (and only loaded) for new-build generators
    generator_annual_build_limit = constants.loc[new_generators, "annual_build_limit"].to_dict()
    generator_unit_size = constants.loc[new_generators, "unit_size"].to_dict()

    # dispatchable generators have a capacity factor of 1.0 every hour; renewables are derated by their zone/tech profile
    generator_capacity_factor = {
        (gen, hour): (
            renewable_availability[(generator_zone[gen], generator_tech[gen], hour)]
            if generator_retech[gen] else 1.0
        )
        for gen in generators
        for hour in hours
    }

    # ---- costs ----
    costs = generator_costs_df.set_index(["generator", "cost_type"])
    generator_capex_cost = {
        (gen, year): costs.loc[(gen, "capex"), str(year)]
        for gen in new_generators
        for year in years
    }
    generator_marginal_cost = {
        (gen, year): costs.loc[(gen, "marginal_cost"), str(year)]
        for gen in generators
        for year in years
    }

    return {
        None: {
            "zones": {None: zones},
            "years": {None: years},
            "hours": {None: hours},
            "existing_generators": {None: existing_generators},
            "new_generators": {None: new_generators},
            "generator_zone": generator_zone,
            "generator_tech": generator_tech,
            "generator_capacity": generator_capacity,
            "generator_start_year": generator_start_year,
            "generator_retirement_year": generator_retirement_year,
            "generator_annual_build_limit": generator_annual_build_limit,
            "generator_unit_size": generator_unit_size,
            "generator_capex_cost": generator_capex_cost,
            "generator_marginal_cost": generator_marginal_cost,
            "generator_capacity_factor": generator_capacity_factor,
            "demand": demand,
        }
    }


def solve(discrete_additions=False):
    model = seer_base_model(discrete_additions=discrete_additions)
    instance = model.create_instance(load_data())

    solver = pyo.SolverFactory(SOLVER_NAME)

    start_time = time.perf_counter()
    results = solver.solve(instance, tee=False)
    solve_time = time.perf_counter() - start_time

    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solve did not reach optimality: {results.solver.termination_condition}")

    return instance, results, solve_time


def print_summary(instance):
    print(f"Total system cost: ${pyo.value(instance.objective):,.0f}\n")

    total_lost_load = sum(pyo.value(instance.lost_load[zone, year, hour]) for zone, year, hour in instance.lost_load)
    print(f"Total lost load: {total_lost_load:,.1f} MWh\n")

    print("Capacity added (MW):")
    for gen in instance.new_generators:
        for year in instance.years:
            added = pyo.value(instance.generation_added_capacity[gen, year])
            if added > 1e-3:
                print(f"  {gen:<24} {year}: {added:8.1f} MW")

    print("\nTotal capacity by year (MW):")
    for gen in instance.generators:
        for year in instance.years:
            total = pyo.value(instance.total_capacity[gen, year])
            print(f"  {gen:<24} {year}: {total:8.1f} MW")


if __name__ == "__main__":
    # non integer additions
    instance, results, solve_time = solve(discrete_additions=False)
    print(f"Solve time: {solve_time:.2f}s\n")
    print_summary(instance)

    breakpoint()
    
    # non integer additions
    instance, results, solve_time = solve(discrete_additions=True)
    print(f"Solve time: {solve_time:.2f}s\n")
    print_summary(instance)

    breakpoint()

    fig, (demand_ax, generation_ax) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plot_demand(demand_ax, instance)
    plot_generation(generation_ax, instance)
    fig.tight_layout()

    output_path = INPUT_DIR.parent / "plot.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
