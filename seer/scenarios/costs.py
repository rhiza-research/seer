"""Sample and resample capital and marginal cost trajectories (start x pathway).

Parameter targets: capex_start, capex_pathway (item: tech), fuel_start, fuel_pathway (item: fuel),
    line_capex (item: line)
Seer schema files: generator_costs.csv, storage_costs.csv, network.csv
"""
import warnings

import numpy as np

from .scenario_parameters import get_parameter_levels, get_tech_parameter, get_vom_trajectory


def sample_capex(tech, years, scenario_options):
    """Capital cost trajectory, requiring sampling of start and trajectory."""
    start = scenario_options.sample("capex_start", item=tech, years=years)
    pathway = scenario_options.sample("capex_pathway", item=tech, years=years)
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
            if get_parameter_levels("capex_start", unit["tech"], years) is None:
                warnings.warn(f"no capex levels for tech {unit['tech']!r} -- leaving {row[id_col]!r} unchanged")
                continue
            # get current capex
            curr_capex = row[year_cols].to_numpy(dtype=float)
            # sample start and pathway
            start = scenario_options.sample("capex_start", initial=curr_capex[0], item=unit["tech"], years=years)
            pathway = scenario_options.sample("capex_pathway", initial=curr_capex / curr_capex[0], item=unit["tech"], years=years)
            # set new capex using start and pathway
            costs.loc[i, year_cols] = start * pathway / pathway[0]
    return out


def sample_marginal(tech, fuel, years, scenario_options):
    """Marginal cost trajectory, requiring sampling of fuel cost start and pathway."""
    heat_rate, vom = get_tech_parameter(tech, "heat_rate"), get_vom_trajectory(tech, years)
    if heat_rate is None or get_parameter_levels("fuel_start", fuel, years) is None:
        return vom
    # sample start and pathway
    start = scenario_options.sample("fuel_start", item=fuel, years=years)
    pathway = scenario_options.sample("fuel_pathway", item=fuel, years=years)
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
        heat_rate = get_tech_parameter(unit["tech"], "heat_rate")
        if heat_rate is None or get_parameter_levels("fuel_start", unit["fuel"], years) is None:
            continue
        # get current fuel price and VOM
        vom = get_vom_trajectory(unit["tech"], years)
        curr_fuel_cost = np.clip((row[year_cols].to_numpy(dtype=float) - vom) / heat_rate, 1e-6, None)
        # sample start and pathway
        start = scenario_options.sample("fuel_start", initial=curr_fuel_cost[0], item=unit["fuel"], years=years)
        pathway = scenario_options.sample("fuel_pathway", initial=curr_fuel_cost / curr_fuel_cost[0], item=unit["fuel"], years=years)
        # set new marginal cost using start and pathway
        costs.loc[i, year_cols] = heat_rate * start * pathway / pathway[0] + vom
    return out


def sample_line_capex(length_km, years, scenario_options, line="HVAC overhead"):
    """A line's capital cost ($/MW): line_capex ($/MW/km, in years[0]) x its length."""
    return scenario_options.sample("line_capex", item=line, years=years) * length_km


def resample_line_capex(instance, scenario_options, line="HVAC overhead"):
    """Given an instance, rescale every expandable line's capital by line_capex relative to its mid level
    (one draw for every line)."""
    out = instance.copy()
    # resample only if user has specified line_capex as target, and the network has expandable lines
    if not scenario_options.active("line_capex") or not {"expandable", "capital"} <= set(out.network.columns):
        return out

    # scaling factor: sampled line_capex over its mid level
    years = list(out.years["year"])
    mid = get_parameter_levels("line_capex", line, years)["mid"]
    factor = scenario_options.sample("line_capex", initial=mid, item=line, years=years) / mid

    # rescale expandable lines' capital
    rows = out.network["expandable"].fillna(False).astype(bool)
    out.network.loc[rows, "capital"] = out.network.loc[rows, "capital"] * factor
    return out

