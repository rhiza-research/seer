"""Sample and resample capital and marginal cost trajectories (start x pathway).

Parameter targets: capex_start, capex_pathway (key: tech), fuel_start, fuel_pathway (key: fuel)
Seer schema files: generator_costs.csv, storage_costs.csv
"""
import warnings

import numpy as np

from .scenario_parameters import parameter_levels, tech_parameter, vom_trajectory


def sample_capex(tech, years, scenario_options):
    """Capital cost trajectory, requiring sampling of start and trajectory."""
    start = scenario_options.sample("capex_start", key=tech, years=years)
    pathway = scenario_options.sample("capex_pathway", key=tech, years=years)
    return start * pathway / pathway[0]


def resample_capex(instance, scenario_options):
    """Given an instance, resample capex trajectories for every build option."""
    out = instance.copy()
    # resample only if user has specified capex as target
    if not scenario_options.active("capex_start", "capex_pathway"):
        return out

    # iterate over generator and storage costs
    for costs, constants, id_col in [(out.generator_costs, out.generator_constants, "generator"),
                                     (out.storage_costs, out.storage_constants, "storage")]:
        constants = constants.set_index(id_col)
        # get years
        year_cols, years = map(list, zip(*[(c, int(c)) for c in costs.columns if str(c).isdigit()]))
        for i, row in costs[costs["cost_type"] == "capital"].iterrows():
            unit = constants.loc[row[id_col]]
            if not unit["expandable"]:
                continue
            if parameter_levels("capex_start", unit["tech"], years) is None:
                warnings.warn(f"no capex levels for tech {unit['tech']!r} -- leaving {row[id_col]!r} unchanged")
                continue
            # get current capex
            curr_capex = row[year_cols].to_numpy(dtype=float)
            # sample start and pathway
            start = scenario_options.sample("capex_start", initial=curr_capex[0], key=unit["tech"], years=years)
            pathway = scenario_options.sample("capex_pathway", initial=curr_capex / curr_capex[0], key=unit["tech"], years=years)
            # set new capex using start and pathway
            costs.loc[i, year_cols] = start * pathway / pathway[0]
    return out


def sample_marginal(tech, fuel, years, scenario_options):
    """Marginal cost trajectory, requiring sampling of fuel cost start and pathway."""
    heat_rate, vom = tech_parameter(tech, "heat_rate"), vom_trajectory(tech, years)
    if heat_rate is None or parameter_levels("fuel_start", fuel, years) is None:
        return vom
    # sample start and pathway
    start = scenario_options.sample("fuel_start", key=fuel, years=years)
    pathway = scenario_options.sample("fuel_pathway", key=fuel, years=years)
    # return marginal cost
    return heat_rate * start * pathway / pathway[0] + vom


def resample_marginal(instance, scenario_options):
    """Given an instance, resample marginal cost trajectories for every generator by resampling fuel cost and pathway."""
    out = instance.copy()
    # resample only if user has specified fuel as target
    if not scenario_options.active("fuel_start", "fuel_pathway"):
        return out

    # iterate over generator costs
    costs, constants = out.generator_costs, out.generator_constants.set_index("generator")
    year_cols, years = map(list, zip(*[(c, int(c)) for c in costs.columns if str(c).isdigit()]))
    for i, row in costs[costs["cost_type"] == "marginal"].iterrows():
        # get unit
        unit = constants.loc[row["generator"]]
        # get heat rate and fuel price
        heat_rate = tech_parameter(unit["tech"], "heat_rate")
        if heat_rate is None or parameter_levels("fuel_start", unit["fuel"], years) is None:
            continue
        # get current fuel price and VOM
        vom = vom_trajectory(unit["tech"], years)
        curr_fuel_cost = np.clip((row[year_cols].to_numpy(dtype=float) - vom) / heat_rate, 1e-6, None)
        # sample start and pathway
        start = scenario_options.sample("fuel_start", initial=curr_fuel_cost[0], key=unit["fuel"], years=years)
        pathway = scenario_options.sample("fuel_pathway", initial=curr_fuel_cost / curr_fuel_cost[0], key=unit["fuel"], years=years)
        # set new marginal cost using start and pathway
        costs.loc[i, year_cols] = heat_rate * start * pathway / pathway[0] + vom
    return out
