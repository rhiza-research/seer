"""Perturb an existing seer-schema instance: set parameters to scenario levels and / or perturb them.
Examples:
    perturb(instance, scenario={"fuel:gas": "high", "demand_growth": "low"}, seed=0)
    perturb(instance, perturbation={"capex_pathway": 0.3}, seed=0)
    perturb(instance, scenario={"hydro": "low"}, perturbation=0.2, seed=0)

See scenario_options for parameters, groups and keys. Parameters in neither are left as they are.
"""
import numpy as np

from . import costs, demand_trajectory, profiles, settings
from .scenario_options import parse


def perturb(seer_instance, scenario=None, perturbation=None, seed=None):
    """Copy of seer_instance with scenario's levels set, then perturbation's scales applied."""
    options = parse(scenario, perturbation)
    rng = np.random.default_rng(seed)
    out = seer_instance.copy()

    # settings
    out = settings.resample(out, options, rng)

    # demand
    out = demand_trajectory.resample(out, options, rng)
    out = profiles.resample_demand_profile(out, options, rng)

    # renewables
    out = profiles.resample_re_profile(out, options, rng)
    out = profiles.resample_hydro(out, options, rng)

    # costs
    out = costs.resample_capex(out, options, rng)
    out = costs.resample_marginal(out, options, rng)

    return out
