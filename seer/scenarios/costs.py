"""Sample and resample generator_costs.csv / storage_costs.csv. Parameters: capex_start,
capex_pathway (key: tech), fuel_start, fuel_pathway (key: fuel).

A cost trajectory is start x pathway: start the value in the first modeled year, pathway the ratio
to it, each with its own parameter. Marginal cost is heat_rate x fuel price + vom.
"""
import warnings

import numpy as np

from .scenario_data import parameter_levels, table_levels, tech_parameter
from .scenario_options import per_key_rng


def sample_capex(tech, years, options, rng=None):
    """Capital cost trajectory ($/MW, or $/MWh for storage) for a build option of tech, per options."""
    start = options.sample("capex_start", key=tech, years=years, rng=rng)
    pathway = options.sample("capex_pathway", key=tech, years=years, rng=rng)
    return start * pathway / pathway[0]


def resample_capex(instance, options, rng=None):
    """Copy of instance with every build option's capital cost trajectory set / perturbed per options,
    from its own start and pathway; one draw per tech."""
    out = instance.copy()
    if not options.active("capex_start", "capex_pathway"):
        return out
    rng_for = per_key_rng(rng)
    for costs, constants, id_col in [(out.generator_costs, out.generator_constants, "generator"),
                                     (out.storage_costs, out.storage_constants, "storage")]:
        constants = constants.set_index(id_col)
        year_cols = [c for c in costs.columns if str(c).isdigit()]
        years = [int(c) for c in year_cols]
        for i, row in costs[costs["cost_type"] == "capital"].iterrows():
            unit = constants.loc[row[id_col]]
            if not unit["expandable"]:
                continue
            if parameter_levels("capex_start", unit["tech"], years) is None:
                warnings.warn(f"no capex levels for tech {unit['tech']!r} -- leaving {row[id_col]!r} unchanged")
                continue
            own, unit_rng = row[year_cols].to_numpy(dtype=float), rng_for(unit["tech"])
            start = options.sample("capex_start", initial=own[0], key=unit["tech"], years=years, rng=unit_rng)
            pathway = options.sample("capex_pathway", initial=own / own[0], key=unit["tech"], years=years, rng=unit_rng)
            costs.loc[i, year_cols] = start * pathway / pathway[0]
    return out


def sample_marginal(tech, fuel, years, options, rng=None):
    """Marginal cost trajectory ($/MWh) for tech burning fuel, per options; 0 (or vom) without a
    heat rate and fuel price."""
    heat_rate, vom = tech_parameter(tech, "heat_rate"), vom_trajectory(tech, years)
    if heat_rate is None or parameter_levels("fuel_start", fuel, years) is None:
        return vom
    start = options.sample("fuel_start", key=fuel, years=years, rng=rng)
    pathway = options.sample("fuel_pathway", key=fuel, years=years, rng=rng)
    return heat_rate * start * pathway / pathway[0] + vom


def resample_marginal(instance, options, rng=None):
    """Copy of instance with every thermal generator's marginal cost set / perturbed per options: its
    implied fuel price ((marginal - vom) / heat_rate) is set / perturbed and recombined; one draw per fuel."""
    out = instance.copy()
    if not options.active("fuel_start", "fuel_pathway"):
        return out
    rng_for = per_key_rng(rng)
    costs, constants = out.generator_costs, out.generator_constants.set_index("generator")
    year_cols = [c for c in costs.columns if str(c).isdigit()]
    years = [int(c) for c in year_cols]
    for i, row in costs[costs["cost_type"] == "marginal"].iterrows():
        unit = constants.loc[row["generator"]]
        heat_rate = tech_parameter(unit["tech"], "heat_rate")
        if heat_rate is None or parameter_levels("fuel_start", unit["fuel"], years) is None:
            continue
        vom, unit_rng = vom_trajectory(unit["tech"], years), rng_for(unit["fuel"])
        own = np.clip((row[year_cols].to_numpy(dtype=float) - vom) / heat_rate, 1e-6, None)
        start = options.sample("fuel_start", initial=own[0], key=unit["fuel"], years=years, rng=unit_rng)
        pathway = options.sample("fuel_pathway", initial=own / own[0], key=unit["fuel"], years=years, rng=unit_rng)
        costs.loc[i, year_cols] = heat_rate * start * pathway / pathway[0] + vom
    return out


def vom_trajectory(tech, years):
    """tech's mid variable O&M trajectory ($/MWh); 0 if it has none."""
    levels = table_levels("tech", tech, "vom", years)
    return np.zeros(len(years)) if levels is None else levels[0]["mid"] * levels[1]["mid"]
