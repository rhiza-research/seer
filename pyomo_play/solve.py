import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

from model import seer_base_model
from plot import plot_demand, plot_generation, plot_storage

INPUT_DIR = Path(__file__).parent / "input_data"
SOLVER_NAME = "appsi_highs"  # swap for "cbc" / "glpk" / "gurobi" if not installed


def load_data():
    """Read the seer-schema input CSVs and build a Pyomo AbstractModel data dict."""

    years_df = pd.read_csv(INPUT_DIR / "years.csv")
    settings_df = pd.read_csv(INPUT_DIR / "settings.csv")
    zones_df = pd.read_csv(INPUT_DIR / "zones.csv")
    generator_constants_df = pd.read_csv(INPUT_DIR / "generator_constants.csv")
    generator_costs_df = pd.read_csv(INPUT_DIR / "generator_costs.csv")
    demandforecast_df = pd.read_csv(INPUT_DIR / "demandforecast.csv")
    demandprofile_df = pd.read_csv(INPUT_DIR / "demandprofile.csv")
    renewable_profiles_df = pd.read_csv(INPUT_DIR / "renewable_profiles.csv")
    storage_constants_df = pd.read_csv(INPUT_DIR / "storage_constants.csv")
    storage_costs_df = pd.read_csv(INPUT_DIR / "storage_costs.csv")

    years = years_df["year"].tolist()
    zones = zones_df["zone"].tolist()
    generators = generator_constants_df["generator"].tolist()  # local convenience list, not passed to Pyomo directly

    # discount rate is a single scalar setting; the per-year factor is computed here
    discount_rate = settings_df.set_index("parameters")["value"]["dr"]
    base_year = years[0]
    year_discount_factor = {year: 1 / (1 + discount_rate) ** (year - base_year) for year in years}

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

    # ---- storage attributes ----
    storage_constants = storage_constants_df.set_index("storage")
    storage = storage_constants.index.tolist()  # local convenience list, not passed to Pyomo directly
    existing_storage = storage_constants[~storage_constants["expandable"]].index.tolist()
    new_storage = storage_constants[storage_constants["expandable"]].index.tolist()

    storage_zone = storage_constants["zone"].to_dict()
    storage_duration_hours = storage_constants["duration_hours"].to_dict()

    # only meaningful (and only loaded) for existing storage
    storage_capacity_mwh = storage_constants.loc[existing_storage, "capacity_mwh"].to_dict()
    storage_start_year = storage_constants.loc[existing_storage, "start_year"].to_dict()
    storage_retirement_year = storage_constants.loc[existing_storage, "retirement_year"].to_dict()
    # only meaningful (and only loaded) for new-build storage
    storage_annual_build_limit_mwh = storage_constants.loc[new_storage, "annual_build_limit_mwh"].to_dict()

    # ---- storage costs ----
    storage_costs = storage_costs_df.set_index(["storage", "cost_type"])
    storage_capex_cost = {
        (s, year): storage_costs.loc[(s, "capex"), str(year)]
        for s in new_storage
        for year in years
    }
    storage_marginal_cost = {
        (s, year): storage_costs.loc[(s, "marginal_cost"), str(year)]
        for s in storage
        for year in years
    }

    return {
        None: {
            "zones": {None: zones},
            "years": {None: years},
            "hours": {None: hours},
            "year_discount_factor": year_discount_factor,
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
            "existing_storage": {None: existing_storage},
            "new_storage": {None: new_storage},
            "storage_zone": storage_zone,
            "storage_duration_hours": storage_duration_hours,
            "storage_capacity_mwh": storage_capacity_mwh,
            "storage_start_year": storage_start_year,
            "storage_retirement_year": storage_retirement_year,
            "storage_annual_build_limit_mwh": storage_annual_build_limit_mwh,
            "storage_capex_cost": storage_capex_cost,
            "storage_marginal_cost": storage_marginal_cost,
            "demand": demand,
        }
    }


def build_instance(discrete_additions=False, storage_daily=False, storage_reset_hour=0, storage_reset_soc=0.5):
    """Build a concrete model instance from the CSVs currently in input_data/."""
    model = seer_base_model(
        discrete_additions=discrete_additions,
        storage_daily=storage_daily,
        storage_reset_hour=storage_reset_hour,
        storage_reset_soc=storage_reset_soc,
    )
    return model.create_instance(load_data())


def solve(instance):
    """Solve a model instance in place (Pyomo fills in variable values on the instance itself).
    Returns (results, solve_time)."""
    solver = pyo.SolverFactory(SOLVER_NAME)

    start_time = time.perf_counter()
    results = solver.solve(instance, tee=False)
    solve_time = time.perf_counter() - start_time

    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solve did not reach optimality: {results.solver.termination_condition}")

    return results, solve_time


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

    print("\nStorage capacity added (MWh):")
    for s in instance.new_storage:
        for year in instance.years:
            added = pyo.value(instance.storage_added_capacity[s, year])
            if added > 1e-3:
                print(f"  {s:<24} {year}: {added:8.1f} MWh")

    print("\nTotal storage energy capacity by year (MWh):")
    for s in instance.storage:
        for year in instance.years:
            total = pyo.value(instance.storage_total_energy_capacity[s, year])
            print(f"  {s:<24} {year}: {total:8.1f} MWh")


def _pivot_by_year(values, entity_col, years):
    """values: {(entity, year): value} -> DataFrame indexed by year, one column per entity."""
    df = pd.DataFrame(
        [(entity, year, value) for (entity, year), value in values.items()],
        columns=[entity_col, "year", "value"],
    ).pivot(index="year", columns=entity_col, values="value").reindex(years)
    df.columns.name = None
    return df


def _pivot_by_year_hour(values, entity_col, years, hours):
    """values: {(entity, year, hour): value} -> DataFrame indexed by (year, hour), one column per entity."""
    df = pd.DataFrame(
        [(entity, year, hour, value) for (entity, year, hour), value in values.items()],
        columns=[entity_col, "year", "hour", "value"],
    ).pivot(index=["year", "hour"], columns=entity_col, values="value").reindex(
        pd.MultiIndex.from_product([years, hours], names=["year", "hour"])
    )
    df.columns.name = None
    return df


def build_results_dataframes(instance):
    """Return solved-instance outputs as a dict of DataFrames:
    - capacity_df: index=years, columns=generators, values=total capacity (MW)
    - generation_df: index=(year, hour), columns=generators, values=output (MW)
    - storage_capacity_df: index=years, columns=storage, values=total energy capacity (MWh)
    - storage_change_df: index=(year, hour), columns=storage, values=charge(+)/discharge(-) (MW)
    - soc_df: index=(year, hour), columns=storage, values=state of charge (MWh)
    """
    years = list(instance.years)
    hours = list(instance.hours)

    capacity_df = _pivot_by_year(instance.total_capacity.extract_values(), "generator", years)
    generation_df = _pivot_by_year_hour(instance.generation_output.extract_values(), "generator", years, hours)

    storage_capacity_df = _pivot_by_year(instance.storage_total_energy_capacity.extract_values(), "storage", years)
    storage_change_df = _pivot_by_year_hour(instance.storage_change.extract_values(), "storage", years, hours)
    soc_df = _pivot_by_year_hour(instance.soc.extract_values(), "storage", years, hours)

    return {
        "capacity_df": capacity_df,
        "generation_df": generation_df,
        "storage_capacity_df": storage_capacity_df,
        "storage_change_df": storage_change_df,
        "soc_df": soc_df,
    }


def aggregate_generators(df, instance, by=("tech",)):
    """Aggregate a capacity_df/generation_df (columns = individual generators, as returned by
    build_results_dataframes) by generator zone and/or tech. `by` may be ("zone",), ("tech",),
    or ("zone", "tech")."""
    attrs = {
        "zone": instance.generator_zone.extract_values(),
        "tech": instance.generator_tech.extract_values(),
    }
    group_keys = [pd.Series({gen: attrs[attr][gen] for gen in df.columns}, name=attr) for attr in by]

    return df.T.groupby(group_keys).sum().T


if __name__ == "__main__":
    # non integer additions
    instance = build_instance(discrete_additions=False)
    results, solve_time = solve(instance)
    print(f"Solve time: {solve_time:.2f}s\n")
    print_summary(instance)
    rez = build_results_dataframes(instance)

    # integer additions
    instance = build_instance(discrete_additions=True)
    results, solve_time = solve(instance)
    print(f"Solve time: {solve_time:.2f}s\n")
    print_summary(instance)

    fig, (demand_ax, generation_ax, storage_ax) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    plot_demand(demand_ax, instance)
    plot_generation(generation_ax, instance)
    plot_storage(storage_ax, instance)
    fig.tight_layout()
    breakpoint()