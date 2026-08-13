"""Standardized long-format output tables for a solved pyomo seer model (see
seer/models/README.md for the output schema): generator_capacity, storage_capacity, dispatch.
"""

import pandas as pd
import pyomo.environ as pyo


def build_output_tables(instance):
    """Build the three seer output tables from a solved pyomo model instance.

    Returns {"generator_capacity": df, "storage_capacity": df, "dispatch": df}.
    """
    generator_zone = instance.generator_zone.extract_values()
    generator_tech = instance.generator_tech.extract_values()
    storage_zone = instance.storage_zone.extract_values()
    storage_carrier = instance.storage_carrier.extract_values()

    generator_capacity = pd.DataFrame([
        {
            "zone": generator_zone[gen], "tech": generator_tech[gen], "generator": gen,
            "year": year, "capacity": pyo.value(instance.total_capacity[gen, year]),
        }
        for gen in instance.generators for year in instance.years
    ])

    storage_capacity = pd.DataFrame([
        {
            "zone": storage_zone[s], "tech": storage_carrier[s], "storage": s,
            "year": year, "capacity": pyo.value(instance.storage_total_energy_capacity[s, year]),
        }
        for s in instance.storage for year in instance.years
    ])

    generator_dispatch_rows = [
        {
            "zone": generator_zone[gen], "tech": generator_tech[gen], "name": gen,
            "year": year, "hour": hour, "dispatch": pyo.value(instance.generation_output[gen, year, hour]),
        }
        for gen in instance.generators for year, hour in instance.times
    ]
    # storage dispatch: net supply to the grid (positive = discharging), i.e. the opposite sign
    # of storage_change (positive = charging/drawing from the grid - see abstract_model.py)
    storage_dispatch_rows = [
        {
            "zone": storage_zone[s], "tech": storage_carrier[s], "name": s,
            "year": year, "hour": hour, "dispatch": -pyo.value(instance.storage_change[s, year, hour]),
        }
        for s in instance.storage for year, hour in instance.times
    ]
    dispatch = pd.DataFrame(generator_dispatch_rows + storage_dispatch_rows)

    return {
        "generator_capacity": generator_capacity,
        "storage_capacity": storage_capacity,
        "dispatch": dispatch,
    }
