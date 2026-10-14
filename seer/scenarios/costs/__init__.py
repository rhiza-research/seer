"""Cost scenarios: capex and marginal-cost trajectories.

- reference: reference cost/fuel tables and decline-trajectory curves
- perturb: perturbations of an existing instance's costs
"""

from .perturb import CostPerturbation, MarginalCostPerturbation

__all__ = ["CostPerturbation", "MarginalCostPerturbation"]
