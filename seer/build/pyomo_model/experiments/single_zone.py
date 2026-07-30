"""Experiment: for a single zone, measure solver performance as the number of years increases,
comparing continuous vs. integer (discrete-additions) capacity expansion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyomo.environ as pyo

from seer.inputs.synthetic import generate_all_data
from build import build
from solve import solve
from plotting_utils import plot_solve_time_scaling

BASE_YEAR = 2025
N_ZONES = 1
YEAR_COUNTS = [1, 2, 3, 5, 7, 10]

CONFIGURATIONS = {
    "Continuous": False,
    "Discrete (integer)": True,
}

results = {label: [] for label in CONFIGURATIONS}
for n_years in YEAR_COUNTS:
    years = list(range(BASE_YEAR, BASE_YEAR + n_years))
    data_path = generate_all_data(n_zones=N_ZONES, years=years)

    for label, discrete_additions in CONFIGURATIONS.items():
        instance = build(data_path, discrete_additions=discrete_additions)
        _, solve_time = solve(instance, verbose=False)

        results[label].append({
            "n_years": n_years,
            "n_variables": instance.nvariables(),
            "n_constraints": instance.nconstraints(),
            "solve_time": solve_time,
            "total_cost": pyo.value(instance.objective),
        })
        print(
            f"{label:<20} n_years={n_years:>3}  "
            f"vars={results[label][-1]['n_variables']:>8}  "
            f"constraints={results[label][-1]['n_constraints']:>8}  "
            f"solve_time={solve_time:6.2f}s"
        )

plot_solve_time_scaling(
    results=results,
    x="n_years",
    y="solve_time",
    title="Solve time vs. number of years (1 zone)",
    save_path=Path(__file__).parent / "single_zone_scaling.png",
)

breakpoint()