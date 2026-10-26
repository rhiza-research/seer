"""Scenario generation on seer-schema instances:

    from seer.scenarios import create, perturb
    instance = create("TUR", scenario={"fuel:gas": "high", "demand_growth": "low"}, seed=0)
    perturbed = perturb(instance, perturbation={"capex_pathway": 0.3}, seed=0)

scenario_parameters.parse turns scenario / perturbation specs into one ScenarioOptions, which create.py,
perturb.py and the topic modules' sample / resample functions (costs, demand_trajectory, profiles,
settings) all read; scenario_parameters.parameter_levels gives every parameter's low / mid / high.
"""

from . import costs, demand_trajectory, profiles, settings
from .create import create
from .perturb import perturb

__all__ = ["costs", "demand_trajectory", "profiles", "settings", "create", "perturb"]
