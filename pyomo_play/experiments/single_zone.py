"""Experiment: for a single zone, measure solver performance as the number of years increases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyomo.environ as pyo

from generate_data import generate_all_data
from solve import build_instance, solve

BASE_YEAR = 2025
N_ZONES = 1
YEAR_COUNTS = [1, 2, 3, 5, 7, 10]

results = []
for n_years in YEAR_COUNTS:
    years = list(range(BASE_YEAR, BASE_YEAR + n_years))
    generate_all_data(n_zones=N_ZONES, years=years)

    instance = build_instance()
    _, solve_time = solve(instance)

    results.append({
        "n_years": n_years,
        "n_variables": instance.nvariables(),
        "n_constraints": instance.nconstraints(),
        "solve_time": solve_time,
        "total_cost": pyo.value(instance.objective),
    })
    print(
        f"n_years={n_years:>3}  "
        f"vars={results[-1]['n_variables']:>8}  "
        f"constraints={results[-1]['n_constraints']:>8}  "
        f"solve_time={solve_time:6.2f}s"
    )
