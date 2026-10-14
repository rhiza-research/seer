"""Demand scenarios: growth trajectory (demand_forecast) and daily load-curve shape (demand_profile).

- library: real daily-profile library -- shape features and feature-matched sampling
- perturb: perturbations of an existing instance's demand
"""

from .perturb import DemandGrowthPerturbation, LoadCurvePerturbation

__all__ = ["DemandGrowthPerturbation", "LoadCurvePerturbation"]
