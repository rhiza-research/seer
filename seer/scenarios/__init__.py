"""Scenario generation on seer-schema instances:

    from seer.scenarios import create, perturb
    instance, info = create("Turkiye", base_peak_mw=57000, scenario={"high-cost", "mid-demand"})
    perturbed, log = perturb(instance, "random-load-profile, demand-scale-0.5", seed=0)

create.py and perturb.py each parse their own spec grammar and call the topic modules (costs,
demand_trajectory, profiles, settings), which hold the sampling logic.
"""

from .base import Perturbation, apply_all
from . import costs, demand_trajectory, profiles, settings
from .create import create
from .perturb import perturb

__all__ = ["Perturbation", "apply_all", "costs", "demand_trajectory", "profiles", "settings", "create", "perturb"]
