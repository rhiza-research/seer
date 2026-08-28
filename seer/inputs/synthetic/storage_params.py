"""Generates the seer schema files `storage_constants.csv` and `storage_costs.csv`.
seed: parameters for a single storage type, including unit size, duration, costs. Assumes
exactly one storage type exists - unlike generators, there's no mix to split across types.
"""

import numpy as np
import pandas as pd

from .noise import noise_multiplier


def build_storage(zones, initial_capacity, years, noise_pct, aggregate_units=True,
                   seed_path="seed_data/storage_seed.csv"):
    """Build storage_constants and storage_costs.

    Existing storage capacity:
    initial_capacity is each zone's total starting storage capacity (MWh).
    If aggregate_units, existing capacity is assigned to a single storage unit, with capacity =
    unit_size_mwh * unit_count. Otherwise, individual units of unit_size_mwh are created.
    aggregate_units allows us to have a simpler model, but manifests as less realistic constraints.

    New-build storage options:
    This is always specified as a single modular unit of unit capacity unit_size_mwh. Each zone
    recieves a new-build option.

    noise_pct: one multiplicative noise draw per storage unit's capex, held constant across
    years. marginal_cost is fixed at 0 for storage, so no noise applies to it.
    """
    seed = pd.read_csv(seed_path).set_index("type")
    assert len(seed) == 1, "build_storage assumes exactly one storage type in the seed"
    storage_type = seed.index[0]

    # determine the number of units of the storage type per zone based on total zone capacity
    unit_counts = set_existing_capacity(zones, initial_capacity, seed)

    base_year = years[0]
    constants_rows, cost_rows = [], []

    # helper function to add a storage unit to the constants and cost rows
    def add_storage(storage, zone, capacity_mwh, unit_size_mwh, start_year, retirement_year,
                     annual_build_limit_mwh, expandable):
        params = seed.loc[storage_type]
        duration_hours = params["duration_hours"]
        constants_rows.append({
            "storage": storage, "zone": zone, "tech": params["carrier"],
            "duration_hours": duration_hours,
            "capacity_mw": round(capacity_mwh / duration_hours, 3),
            "efficiency": params["efficiency"], "life": params["life"],
            "start_year": start_year, "retirement_year": retirement_year,
            "unit_size_mwh": unit_size_mwh, "annual_build_limit_mwh": annual_build_limit_mwh,
            "expandable": expandable,
        })

        multiplier = noise_multiplier(noise_pct)
        capex_row = {"storage": storage, "cost_type": "capex"}
        marginal_cost_row = {"storage": storage, "cost_type": "marginal_cost"}
        for year in years:
            capex_row[year] = round(
                params["capex_per_mwh"] * multiplier * (1 + params["capex_escalation"]) ** (year - base_year), 4
            )
            marginal_cost_row[year] = 0.0
        cost_rows.extend([capex_row, marginal_cost_row])

    # add existing and new-build storage units
    for zone, unit_count in unit_counts.items():
        if unit_count <= 0:
            continue
        unit_size_mwh = seed.loc[storage_type, "unit_size_mwh"]
        # if aggregate_units, create a single storage unit; otherwise multiple units
        count, multiplier = (1, unit_count) if aggregate_units else (unit_count, 1)

        for i in range(count):
            suffix = "existing" if count == 1 else f"existing_{i}"
            capacity_mwh = round(unit_size_mwh * multiplier, 3)
            add_storage(
                f"{zone}_{storage_type}_{suffix}", zone,
                capacity_mwh=capacity_mwh, unit_size_mwh=capacity_mwh,
                start_year=years[0], retirement_year=years[-1] + 1,
                annual_build_limit_mwh=None, expandable=False,
            )

        # new-build option
        add_storage(
            f"{zone}_{storage_type}_new", zone,
            capacity_mwh=unit_size_mwh, unit_size_mwh=unit_size_mwh,
            start_year=None, retirement_year=None,
            annual_build_limit_mwh=seed.loc[storage_type, "annual_build_limit_mwh"], expandable=True,
        )

    return pd.DataFrame(constants_rows), pd.DataFrame(cost_rows)


def set_existing_capacity(zones, initial_capacity, seed):
    """Based on total zone capacity, calculate the number of storage units per zone.

    Returns {zone: unit_count}.
    """
    if np.isscalar(initial_capacity):
        initial_capacity = [initial_capacity] * len(zones)

    unit_size = seed.loc[seed.index[0], "unit_size_mwh"]
    return {
        zone: round(total_capacity / unit_size)
        for zone, total_capacity in zip(zones, initial_capacity)
    }
