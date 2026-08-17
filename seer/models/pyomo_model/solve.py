import time

import pyomo.environ as pyo


def solve(instance, solver_name="appsi_highs", verbose=True, solver_options=None):
    # solver - swap for "cbc" / "glpk" / "gurobi" if not installed
    """Solve a model instance in place (Pyomo fills in variable values on the instance itself)
    and print a summary of the results. Returns (results, solve_time).

    solver_options: optional dict of solver-native options (e.g. {"presolve": "off"} for
    appsi_highs), passed straight through to the solver.
    """
    solver = pyo.SolverFactory(solver_name)
    if solver_options:
        solver.options.update(solver_options)

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
