"""Experiment: for a single zone, generate synthetic seer-schema data, write it to disk, then
build and solve a pyomo model from it, measuring solver performance as years increase."""

from pathlib import Path

import pyomo.environ as pyo

from seer.inputs.synthetic import generate_all_data
from seer.models.pyomo_model import build, solve

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "seer" / "synthetic" / "single_zone"

BASE_YEAR = 2025
N_ZONES = 1
YEAR_COUNTS = [1, 2, 3, 5, 7, 10]

results = []
for n_years in YEAR_COUNTS:
    years = list(range(BASE_YEAR, BASE_YEAR + n_years))
    data_path = generate_all_data(n_zones=N_ZONES, years=years, output_path=OUTPUT_PATH)

    instance = build(data_path)
    _, solve_time = solve(instance, verbose=False)

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
