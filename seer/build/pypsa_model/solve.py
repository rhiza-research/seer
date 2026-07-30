"""Solve a built PyPSA seer model and print a summary of the results."""

import time

SOLVER_NAME = "highs"  # matches the HiGHS solver used on the pyomo side


def solve(model, solver_name=SOLVER_NAME, verbose=True, **optimize_kwargs):
    """Solve a PyPSA Network in place and print a summary of the results.
    Returns (status, condition, solve_time)."""
    start_time = time.perf_counter()
    status, condition = model.optimize(solver_name=solver_name, **optimize_kwargs)
    solve_time = time.perf_counter() - start_time

    if condition != "optimal":
        raise RuntimeError(f"Solve did not reach optimality: status={status}, condition={condition}")

    if verbose:
        print(f"Solve time: {solve_time:.2f}s\n")
        print_summary(model)

    return status, condition, solve_time


def print_summary(model):
    print(f"Objective: {model.objective:,.0f}\n")

    print("Optimized generator capacity (MW):")
    for name, p_nom_opt in model.generators["p_nom_opt"].items():
        print(f"  {name:<24} {p_nom_opt:8.1f} MW")

    print("\nOptimized storage capacity (MW):")
    for name, p_nom_opt in model.storage_units["p_nom_opt"].items():
        print(f"  {name:<24} {p_nom_opt:8.1f} MW")
