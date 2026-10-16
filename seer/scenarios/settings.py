"""Global settings perturbations (settings.csv) -- currently just the discount rate. This is a
single modeling-horizon-wide choice rather than something spread across rows per tech/fuel/day, so
there's no reference data or trajectory to draw from, just a plausible range to sample uniformly
from.
"""
from dataclasses import dataclass, field

from .base import Perturbation


@dataclass
class DiscountRatePerturbation(Perturbation):
    low: float = 0.03
    high: float = 0.12
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        dr = float(rng.uniform(self.low, self.high))
        out.settings = out.settings.copy()
        out.settings.loc[out.settings["parameters"] == "dr", "value"] = dr
        self.sampled_ = {"dr": dr}
        return out

    def params(self):
        return {"sampled": self.sampled_}
