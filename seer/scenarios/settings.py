"""Global settings perturbations (settings.csv) -- currently just the discount rate. This is a
single modeling-horizon-wide choice rather than something spread across rows per tech/fuel/day, so
there's no reference data or trajectory to draw from, just a plausible range to sample uniformly
from.
"""
from dataclasses import dataclass, field

import pandas as pd

from .base import Perturbation
from .spec import REFERENCE, sample_value


def resample(instance, options, rng):
    """Resample dr and voll at the levels in options."""
    out = instance.copy()
    for name in ["dr", "voll"]:
        if name not in options:
            continue
        ref = REFERENCE["settings"][name]
        rows = out.settings["parameters"] == name
        start = float(out.settings.loc[rows, "value"].iloc[0]) if rows.any() else ref["mid"]
        value = sample_value(options[name], start, ref, rng, options["scale"])
        if rows.any():
            out.settings.loc[rows, "value"] = value
        else:
            out.settings = pd.concat([out.settings, pd.DataFrame({"parameters": [name], "value": [value]})], ignore_index=True)
    return out


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
