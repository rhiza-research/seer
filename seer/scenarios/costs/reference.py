"""Reference cost tables (seer/scenarios/reference_data/cost_assumptions) and the trajectory-curve
helper shared by every cost perturbation (and, later, cost creation). Built by
scripts/build_cost_ranges.py and scripts/build_fuel_prices.py.
"""
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).parents[1] / "reference_data" / "cost_assumptions"

TRAJECTORY_SCENARIOS = ["low", "mid", "high"]


def load_capex_tables(reference_dir=REFERENCE_DIR):
    return (
        pd.read_csv(reference_dir / "costs.csv"),
        pd.read_csv(reference_dir / "decline_curves.csv"),
    )


def load_fuel_tables(reference_dir=REFERENCE_DIR):
    return (
        pd.read_csv(reference_dir / "costs.csv"),
        pd.read_csv(reference_dir / "fuel_prices.csv"),
        pd.read_csv(reference_dir / "decline_curves_fuel.csv"),
    )


def trajectory_curve(decline_df, key_col, key, value_col, scenario, years, start_year):
    """One scenario's decline curve for `key` (a tech or fuel), indexed by year and filled onto
    every year in `years` plus start_year -- interpolated between the table's own years, held
    flat beyond either end."""
    curve = decline_df[(decline_df[key_col] == key) & (decline_df["scenario"] == scenario)].set_index("year")[value_col].sort_index()
    all_years = sorted(set(curve.index) | set(years) | {start_year})
    return curve.reindex(all_years).interpolate(method="index").ffill().bfill()
