"""Scenario generation on seer-schema instances, organized by topic (one module per topic, holding
both its perturbation and -- later -- its from-scratch creation logic, since the two share almost
everything):

    from seer.scenarios import apply_all
    from seer.scenarios.costs import CapexPerturbation, MarginalPerturbation
    from seer.scenarios.profiles import LoadProfilePerturbation, ReProfilePerturbation
    from seer.scenarios.demand_trajectory import DemandGrowthPerturbation
    from seer.scenarios.settings import DiscountRatePerturbation
"""

from .base import Perturbation, apply_all
from . import costs, demand_trajectory, profiles, settings

__all__ = ["Perturbation", "apply_all", "costs", "demand_trajectory", "profiles", "settings"]
