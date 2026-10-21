"""Perturb an existing seer-schema instance.
Examples:
    perturb(instance, "high-dr, random-voll, scale-0.5", seed=0)
    perturb(instance, "random", seed=0)
"""
import numpy as np

from . import costs, demand_trajectory, profiles, settings
from .spec import tokenize

TARGETS = ["dr", "voll", "demand-growth", "demand-profile", "re-profile", "hydro", "capex-start", "capex-pathway", "fuel-start", "fuel-pathway"]
GROUPS = {"capex": ["capex-start", "capex-pathway"], "fuel": ["fuel-start", "fuel-pathway"],
          "cost": ["capex-start", "capex-pathway", "fuel-start", "fuel-pathway"]}


def perturb(seer_instance, spec, seed=None):
    """Return a perturbed copy of seer_instance."""
    options = parse(spec)
    rng = np.random.default_rng(seed)
    out = seer_instance.copy()

    # settings
    out = settings.resample(out, options, rng)

    # demand
    if "demand-growth" in options:
        out = demand_trajectory.DemandTrajectorySampler().resample(out, rng, options["demand-growth"], options["scale"])
    if "demand-profile" in options:
        out = profiles.LoadProfileSampler().resample(out, rng, options["scale"])
    # renewables
    if "re-profile" in options:
        out = profiles.ReProfileSampler().resample(out, rng, options["scale"])
    if "hydro" in options:
        out = profiles.HydroProfileSampler().resample(out, rng, options["hydro"], options["scale"])
    # costs
    if "capex-start" in options or "capex-pathway" in options:
        out = costs.CapexSampler().resample(out, rng, options.get("capex-start"), options.get("capex-pathway"), options["scale"])
    if "fuel-start" in options or "fuel-pathway" in options:
        out = costs.MarginalSampler().resample(out, rng, options.get("fuel-start"), options.get("fuel-pathway"), options["scale"])

    return out


def parse(spec):
    """'high-dr, random-re-profile, scale-0.5' -> {"dr": "high", "re-profile": "random", "scale": 0.5}."""
    options = {"scale": 1.0}
    for token in tokenize(spec):
        level, _, target = token.partition("-")
        if token == "random":
            options.update(dict.fromkeys(TARGETS, "random"))
        elif level == "scale":
            options["scale"] = float(target)
        elif target in GROUPS:
            options.update(dict.fromkeys(GROUPS[target], level))
        elif target in TARGETS:
            options[target] = level
        else:
            raise ValueError(f"can't parse {token!r}; expected <level>-<target> with target in {TARGETS + list(GROUPS)}, scale-<x> or random")
    return options
