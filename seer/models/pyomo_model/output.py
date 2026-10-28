"""Standardized long-format output tables for a solved pyomo seer model (see
seer/models/README.md for the output schema): generator_capacity, storage_capacity, dispatch,
line_capacity, lost_load, line_flow, storage_soc, prices, summary.
"""

import pandas as pd

from ..utils.output_utils import add_input_metadata, summary_table
import pyomo.environ as pyo


def build_output_tables(instance, data_path):
    """Build the three seer output tables from a solved pyomo model instance. zone/tech/fuel
    come from the seer-schema inputs at `data_path` (see output_utils.add_input_metadata).

    Returns {"line_capacity", "generator_capacity", "storage_capacity", "dispatch", "lost_load", "line_flow",
    "storage_soc", "prices", "summary"}: df.
    """
    generator_zone = instance.generator_zone.extract_values()
    generator_tech = instance.generator_tech.extract_values()
    storage_zone = instance.storage_zone.extract_values()
    storage_tech = instance.storage_tech.extract_values()
    snapshot_datetime = instance.snapshot_datetime.extract_values()

    generator_capacity = pd.DataFrame([
        {
            "zone": generator_zone[gen], "tech": generator_tech[gen], "generator": gen,
            "year": year, "capacity": pyo.value(instance.total_capacity[gen, year]),
        }
        for gen in instance.generators for year in instance.years
    ])

    storage_capacity = pd.DataFrame([
        {
            "zone": storage_zone[s], "tech": storage_tech[s], "storage": s,
            "year": year, "capacity": pyo.value(instance.storage_total_energy_capacity[s, year]),
        }
        for s in instance.storage for year in instance.years
    ], columns=["zone", "tech", "storage", "year", "capacity"])  # columns even with no storage

    generator_dispatch_rows = [
        {
            "zone": generator_zone[gen], "tech": generator_tech[gen], "name": gen,
            "year": year, "hour": hour, "datetime": snapshot_datetime[year, hour],
            "dispatch": pyo.value(instance.generation_output[gen, year, hour]),
        }
        for gen in instance.generators for year, hour in instance.times
    ]
    # storage dispatch: net supply to the grid (positive = discharging), i.e. the opposite sign
    # of storage_change (positive = charging/drawing from the grid - see abstract_model.py)
    storage_dispatch_rows = [
        {
            "zone": storage_zone[s], "tech": storage_tech[s], "name": s,
            "year": year, "hour": hour, "datetime": snapshot_datetime[year, hour],
            "dispatch": -pyo.value(instance.storage_change[s, year, hour]),
        }
        for s in instance.storage for year, hour in instance.times
    ]
    dispatch = pd.DataFrame(generator_dispatch_rows + storage_dispatch_rows)

    # unserved demand, priced at settings.csv's `voll` in the objective
    lost_load = pd.DataFrame([
        {
            "zone": zone, "year": year, "hour": hour, "datetime": snapshot_datetime[year, hour],
            "lost_load": pyo.value(instance.lost_load[zone, year, hour]),
        }
        for zone in instance.zones for year, hour in instance.times
    ], columns=["zone", "year", "hour", "datetime", "lost_load"])

    line_from = instance.line_from.extract_values()
    line_to = instance.line_to.extract_values()
    line_capacity = pd.DataFrame([
        {
            "from": line_from[line], "to": line_to[line], "year": year,
            "capacity": pyo.value(instance.line_total_capacity[line, year]),
        }
        for line in instance.lines for year in instance.years
    ], columns=["from", "to", "year", "capacity"])
    # one row per (zone pair, year): sum a pair's lines with different start years
    line_capacity = line_capacity.groupby(["from", "to", "year"], as_index=False, sort=False)["capacity"].sum()

    # positive from -> to, summed per zone pair as in line_capacity
    line_flow = pd.DataFrame([
        {
            "from": line_from[line], "to": line_to[line], "year": year, "hour": hour,
            "datetime": snapshot_datetime[year, hour], "flow": pyo.value(instance.line_flow[line, year, hour]),
        }
        for line in instance.lines for year, hour in instance.times
    ], columns=["from", "to", "year", "hour", "datetime", "flow"])
    line_flow = line_flow.groupby(["from", "to", "year", "hour", "datetime"], as_index=False, sort=False)["flow"].sum()

    storage_soc = pd.DataFrame([
        {
            "zone": storage_zone[s], "tech": storage_tech[s], "storage": s, "year": year, "hour": hour,
            "datetime": snapshot_datetime[year, hour], "soc": pyo.value(instance.soc[s, year, hour]),
        }
        for s in instance.storage for year, hour in instance.times
    ], columns=["zone", "tech", "storage", "year", "hour", "datetime", "soc"])

    # supply_equals_demand's dual / (year_discount_factor x time_weight) = $/MWh; None for a MILP
    duals = getattr(instance, "dual", None)
    prices = pd.DataFrame([
        {
            "zone": zone, "year": year, "hour": hour, "datetime": snapshot_datetime[year, hour],
            "price": duals[instance.supply_equals_demand[zone, year, hour]]
            / (pyo.value(instance.year_discount_factor[year]) * pyo.value(instance.time_weight[year, hour]))
            if duals is not None and instance.supply_equals_demand[zone, year, hour] in duals else None,
        }
        for zone in instance.zones for year, hour in instance.times
    ], columns=["zone", "year", "hour", "datetime", "price"])

    return add_input_metadata({
        "line_capacity": line_capacity,
        "generator_capacity": generator_capacity,
        "storage_capacity": storage_capacity,
        "dispatch": dispatch,
        "lost_load": lost_load,
        "line_flow": line_flow,
        "storage_soc": storage_soc,
        "prices": prices,
        "summary": summary(instance),
    }, data_path)


def summary(instance):
    """The summary table, from the objective's named cost terms (see abstract_model.py)."""
    def weighted_energy(var):
        return sum(pyo.value(instance.time_weight[t]) * pyo.value(var[z, t]) for z in instance.zones for t in instance.times)

    return summary_table(
        objective=pyo.value(instance.objective),
        costs={name: pyo.value(getattr(instance, name))
               for name in ["generator_capex", "storage_capex", "line_capex", "variable_cost", "lost_load_cost"]},
        demand=weighted_energy(instance.demand), lost_load=weighted_energy(instance.lost_load),
    )
