"""Perturb an existing seer-schema instance: set parameters to fixed scenario levels and / or perturb them.
Examples:
    perturb(instance, scenario={"fuel:gas": "high", "demand_growth": "low"}, seed=0)
    perturb(instance, perturbation={"capex_pathway": 0.3}, seed=0)
    perturb(instance, scenario={"hydro": "low"}, perturbation=0.2, seed=0)

See scenario_parameters for parameters, groups and items. Parameters in neither are left as they are.
"""
from . import costs, demand_trajectory, profiles, settings
from .scenario_parameters import parse


def perturb(seer_instance, scenario=None, perturbation=None, seed=None):
    """Copy of seer_instance with scenario's levels set, then perturbation's scales applied."""
    scenario_options = parse(scenario, perturbation, seed=seed)
    out = seer_instance.copy()

    # settings
    out = settings.resample(out, scenario_options)

    # demand
    out = demand_trajectory.resample(out, scenario_options)
    out = profiles.resample_demand_profile(out, scenario_options)

    # renewables
    out = profiles.resample_re_profile(out, scenario_options)
    out = profiles.resample_hydro(out, scenario_options)

    # costs
    out = costs.resample_capex(out, scenario_options)
    out = costs.resample_marginal(out, scenario_options)

    return out
