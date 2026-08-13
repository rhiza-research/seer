"""Generates the seer schema files `generator_constants.csv` and `generator_costs.csv`.
seed: parameters for each generator type, including unit size, ramp rates, costs.
"""

import numpy as np
import pandas as pd

from .noise import noise_multiplier


def build_generators(zones, mix, initial_capacity, years, noise_pct, aggregate_units=True,
                      seed_path="seed_data/generator_seed.csv"):
    """Build generator_constants and generator_costs.

    Existing generation capacity:
    If aggregate_units, existing capacity of each type is assigned to a single generator,
    with capacity = unit_size_mw * unit_count. Otherwise, individual units of unit_size_mw are created.
    aggregate_units allows us to have a simpler model, but manifests as less realistic constraints. 

    New-build generation options: 
    This is always specified as a single modular unit of unit capacity unit_size_mw. Each zone recieves
    a new-build option for each generator type.
    """
    seed = pd.read_csv(seed_path).set_index("type")
    # determine the number of units of each type per zone based on total zone capacity and mix
    unit_counts = set_existing_capacity(zones, mix, initial_capacity, seed)

    base_year = years[0]
    constants_rows, cost_rows = [], []

    # helper function to add a generator to the constants and cost rows
    def add_generator(generator, gen_type, zone, capacity, unit_size, start_year, retirement_year,
                       annual_build_limit, expandable):
        params = seed.loc[gen_type]
        constants_rows.append({
            "generator": generator, "zone": zone, "tech": gen_type, "fuel": params["fuel"],
            "capacity": capacity, "unit_size": unit_size,
            "ramp_down_rate": params["ramp_down_rate"], "ramp_up_rate": params["ramp_up_rate"],
            "life": params["life"], "RETech": params["retech"],
            "start_year": start_year, "retirement_year": retirement_year,
            "annual_build_limit": annual_build_limit, "expandable": expandable,
        })
        for cost_type, base_value, escalation in (
            ("capex", params["capex_per_mw"], params["capex_escalation"]),
            ("marginal_cost", params["marginal_cost_per_mwh"], params["marginal_cost_escalation"]),
        ):
            multiplier = noise_multiplier(noise_pct)
            row = {"generator": generator, "cost_type": cost_type}
            for year in years:
                row[year] = round(base_value * multiplier * (1 + escalation) ** (year - base_year), 4)
            cost_rows.append(row)

    # add existing and new-build generators
    for (zone, gen_type), unit_count in unit_counts.items():
        if unit_count <= 0:
            continue
        unit_size_mw = seed.loc[gen_type, "unit_size_mw"]
        # if aggregate_units, create a single generator; otherwise multiple units
        count, multiplier = (1, unit_count) if aggregate_units else (unit_count, 1)

        for i in range(count):
            suffix = "existing" if count == 1 else f"existing_{i}"
            capacity = round(unit_size_mw * multiplier, 3)
            add_generator(
                f"{zone}_{gen_type}_{suffix}", gen_type, zone,
                capacity=capacity, unit_size=capacity,
                start_year=years[0], retirement_year=years[-1] + 1,
                annual_build_limit=None, expandable=False,
            )

        # new-build option
        add_generator(
            f"{zone}_{gen_type}_new", gen_type, zone,
            capacity=unit_size_mw, unit_size=unit_size_mw,
            start_year=None, retirement_year=None,
            annual_build_limit=seed.loc[gen_type, "annual_build_limit_mw"], expandable=True,
        )

    return pd.DataFrame(constants_rows), pd.DataFrame(cost_rows)


def set_existing_capacity(zones, mix, initial_capacity, seed):
    """Based on total zone capacity and mix, calculate number of units of each type per zone.

    Returns {(zone, generator_type): unit_count}.
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
        for gen_type, share in zone_mix.items():
            unit_size = seed.loc[gen_type, "unit_size_mw"]
            unit_counts[(zone, gen_type)] = round(total_capacity * share / unit_size)

    return unit_counts
