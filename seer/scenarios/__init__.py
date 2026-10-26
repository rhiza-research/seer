"""Scenario generation on seer-schema instances:

    from seer.scenarios import create, perturb
    instance = create("TUR", scenario=["capex-start-high", "demand-growth-low"], seed=0)
    perturbed = perturb(instance, perturbation=["demand-profile-0.5"], seed=0)

scenario_options.parse turns scenario / perturbation specs into one ScenarioOptions, which create.py,
perturb.py and the topic modules' samplers (costs, demand_trajectory, profiles, settings) all read.
"""

from . import costs, demand_trajectory, profiles, settings
from .create import create
from .perturb import perturb

__all__ = ["costs", "demand_trajectory", "profiles", "settings", "create", "perturb"]
