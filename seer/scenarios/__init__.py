"""Scenario generation on seer-schema instances:

    from seer.scenarios import create, perturb
    instance = create("TUR", "high-capex-start, low-demand-growth", seed=0)
    perturbed = perturb(instance, "random-demand-profile, scale-0.5", seed=0)

create.py and perturb.py share one spec grammar (perturb.parse) and call the topic modules (costs,
demand_trajectory, profiles, settings), whose samplers hold the sampling logic.
"""

from . import costs, demand_trajectory, profiles, settings
from .create import create
from .perturb import perturb

__all__ = ["costs", "demand_trajectory", "profiles", "settings", "create", "perturb"]
