"""Demand growth trajectory (demand_forecast.csv) perturbation. Separate from profiles.py's daily
load-curve shape logic: this works on the annual forecast, not a profile library, and is driven by
the dataclass's own sampled bounds rather than reference data, since there's no external source of
"how much might a country's demand growth rate vary".
"""
from dataclasses import dataclass, field

from .base import Perturbation


def scale_demand_growth(demand_forecast, exponent):
    """new_value[year] = start_value * (value[year] / start_value) ** exponent, per (zone, type)
    trajectory. exponent=1 leaves it unchanged; >1 grows (or shrinks) more sharply than the
    original trajectory implied, <1 more gently -- this scales whatever growth shape the
    original data already had, rather than refitting a constant annual rate and discarding it.
    """
    demand_forecast = demand_forecast.copy()
    for _, group in demand_forecast.groupby(["zone", "type"]):
        start_year = group["year"].min()
        start_value = group.loc[group["year"] == start_year, "value"].iloc[0]
        ratio = group["value"] / start_value
        demand_forecast.loc[group.index, "value"] = start_value * ratio**exponent
    return demand_forecast


@dataclass
class DemandGrowthPerturbation(Perturbation):
    low: float = 0.5
    high: float = 1.5
    sampled_: dict = field(default_factory=dict, repr=False, compare=False)

    def apply(self, instance, rng):
        out = instance.copy()
        exponent = float(rng.uniform(self.low, self.high))
        self.sampled_ = {"growth_exponent": exponent}
        out.demand_forecast = scale_demand_growth(out.demand_forecast, exponent)
        return out

    def params(self):
        return {"sampled": self.sampled_}
