import time
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

SOLVER_NAME = "appsi_highs"  # swap for "cbc" / "glpk" / "gurobi" if not installed


def solve(instance, solver_name=SOLVER_NAME, verbose=True):
    """Solve a model instance in place (Pyomo fills in variable values on the instance itself)
    and print a summary of the results. Returns (results, solve_time)."""
    solver = pyo.SolverFactory(solver_name)

    start_time = time.perf_counter()
    results = solver.solve(instance, tee=False)
    solve_time = time.perf_counter() - start_time

    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solve did not reach optimality: {results.solver.termination_condition}")

    if verbose:
        print(f"Solve time: {solve_time:.2f}s\n")
        print_summary(instance)

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
    from .build import build

    data_path = Path(__file__).parent / "input_data"

    instance = build(data_path, discrete_additions=False)
    solve(instance)
    rez = build_results_dataframes(instance)

    instance = build(data_path, discrete_additions=True)
    solve(instance)

    from seer.viz.plot import plot_demand, plot_generation, plot_storage
    import matplotlib.pyplot as plt

    fig, (demand_ax, generation_ax, storage_ax) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    plot_demand(demand_ax, instance)
    plot_generation(generation_ax, instance)
    plot_storage(storage_ax, instance)
    fig.tight_layout()
    breakpoint()
