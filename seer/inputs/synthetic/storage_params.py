"""Generates the seer schema files `storage_constants.csv` and `storage_costs.csv`.

The "seed" is `seed_data/storage_seed.csv`: one row per storage type, one column per fixed
parameter, so it can be edited without touching code. It's read in here and expanded into
per-storage-unit, per-year rows.

Only seer-schema columns actually read by a model builder (seer/models/pypsa_model or
seer/models/pyomo_model) are set; the rest of the schema is left unset for now.
"""

import numpy as np
import pandas as pd


def build_storage(zones, mix, initial_capacity, years, cost_variance_pct, aggregate_units=True,
                   seed_path="seed_data/storage_seed.csv"):
    """Build storage_constants and storage_costs together (one row of each per storage unit).

    If aggregate_units, all existing units of a type in a zone become a single storage unit,
    with capacity = unit_size_mwh * unit_count. Otherwise, each physical unit gets its own row,
    each at capacity = unit_size_mwh. Either way there's also one new-build option per
    (zone, type), always sized at a single unit.

    cost_variance_pct: {"capex": ...} +/- fraction (e.g. 0.10 = +/- 10%) for a one-time draw
    per storage unit, held constant across years. marginal_cost is fixed at 0 for storage, so
    no variance is needed for it.
    """
    seed = pd.read_csv(seed_path).set_index("type")
    unit_counts = set_capacities(zones, mix, initial_capacity, seed)
    base_year = years[0]

    constants_rows, cost_rows = [], []

    def add_storage(storage, storage_type, zone, capacity_mwh, unit_size_mwh, start_year, retirement_year,
                     annual_build_limit_mwh, expandable):
        params = seed.loc[storage_type]
        duration_hours = params["duration_hours"]
        constants_rows.append({
            "storage": storage, "zone": zone, "carrier": params["carrier"],
            "capacity_mwh": capacity_mwh, "duration_hours": duration_hours,
            "capacity_mw": round(capacity_mwh / duration_hours, 3),
            "efficiency": params["efficiency"], "life": params["life"],
            "start_year": start_year, "retirement_year": retirement_year,
            "unit_size_mwh": unit_size_mwh, "annual_build_limit_mwh": annual_build_limit_mwh,
            "expandable": expandable,
        })

        variance = cost_variance_pct["capex"]
        multiplier = 1 + np.random.uniform(-variance, variance)
        capex_row = {"storage": storage, "cost_type": "capex"}
        marginal_cost_row = {"storage": storage, "cost_type": "marginal_cost"}
        for year in years:
            capex_row[year] = round(
                params["capex_per_mwh"] * multiplier * (1 + params["capex_escalation"]) ** (year - base_year), 4
            )
            marginal_cost_row[year] = 0.0
        cost_rows.extend([capex_row, marginal_cost_row])

    for (zone, storage_type), unit_count in unit_counts.items():
        if unit_count <= 0:
            continue
        unit_size_mwh = seed.loc[storage_type, "unit_size_mwh"]
        count, multiplier = (1, unit_count) if aggregate_units else (unit_count, 1)

        for i in range(count):
            suffix = "existing" if count == 1 else f"existing_{i}"
            capacity_mwh = round(unit_size_mwh * multiplier, 3)
            add_storage(
                f"{zone}_{storage_type}_{suffix}", storage_type, zone,
                capacity_mwh=capacity_mwh, unit_size_mwh=capacity_mwh,
                start_year=years[0], retirement_year=years[-1] + 1,
                annual_build_limit_mwh=None, expandable=False,
            )

        # new-build option: always a single modular unit, independent of aggregate_units
        add_storage(
            f"{zone}_{storage_type}_new", storage_type, zone,
            capacity_mwh=unit_size_mwh, unit_size_mwh=unit_size_mwh,
            start_year=None, retirement_year=None,
            annual_build_limit_mwh=seed.loc[storage_type, "annual_build_limit_mwh"], expandable=True,
        )

    return pd.DataFrame(constants_rows), pd.DataFrame(cost_rows)


def set_capacities(zones, mix, initial_capacity, seed):
    """Convert each zone's starting capacity (MWh) and storage mix (share of that capacity by
    type) into a whole number of per-unit-sized storage units.

    Returns {(zone, storage_type): unit_count}.
    """
    if isinstance(mix, dict):
        mix_by_zone = {zone: mix for zone in zones}
    elif len(mix) == len(zones):
        mix_by_zone = dict(zip(zones, mix))
    else:
        raise ValueError("mix must be a single dict, or one dict per zone")

    if np.isscalar(initial_capacity):
        initial_capacity = [initial_capacity] * len(zones)

    unit_counts = {}
    for zone, total_capacity in zip(zones, initial_capacity):
        zone_mix = mix_by_zone[zone]
        for storage_type, share in zone_mix.items():
            unit_size = seed.loc[storage_type, "unit_size_mwh"]
            unit_counts[(zone, storage_type)] = round(total_capacity * share / unit_size)

    return unit_counts
