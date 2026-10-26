"""Perturb an existing seer-schema instance: set targets to scenario levels and / or perturb them.
Examples:
    perturb(instance, scenario=["capex-high", "demand-growth-low"], seed=0)
    perturb(instance, perturbation=["fuel-0.3", "demand-profile-1"], seed=0)
    perturb(instance, scenario="hydro-low", perturbation=0.2, seed=0)

See scenario_options for the spec format, targets and groups. Targets not in either spec are left as
they are.
"""
import numpy as np

from . import costs, demand_trajectory, profiles, settings
from .scenario_options import parse


def perturb(seer_instance, scenario=None, perturbation=None, seed=None):
    """Return a copy of seer_instance with scenario's levels set, then perturbation's scales applied."""
    options = parse(scenario, perturbation)
    rng = np.random.default_rng(seed)
    out = seer_instance.copy()

    # settings
    out = settings.resample(out, options, rng)

    # demand
    out = demand_trajectory.DemandTrajectorySampler().resample(out, options, rng)
    out = profiles.LoadProfileSampler().resample(out, options, rng)

    # renewables
    out = profiles.ReProfileSampler().resample(out, options, rng)
    out = profiles.HydroProfileSampler().resample(out, options, rng)

    # costs
    out = costs.CapexSampler().resample(out, options, rng)
    out = costs.MarginalSampler().resample(out, options, rng)

    return out
