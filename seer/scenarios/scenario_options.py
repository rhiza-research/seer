"""Scenario options: parse scenario and perturbation specs into one standardized ScenarioOptions,
which create, perturb and every sampler read.

    scenario = ["capex-high", "demand-growth-low"]       # target-level: low / mid / high / random
    perturbation = ["fuel-0.3", "demand-profile-1"]       # target-scale: 0 (none) to 1 (full range)
    perturbation = 0.2                                     # one scale for every target

A spec is a list of "target-value" strings, or one string of them separated by "," or "+". The value
follows the last "-", since target names contain hyphens. Targets can be groups (GROUPS), which
expand to their granular targets; a more specific target overrides a group that contains it,
whatever the order. demand-profile also accepts named shapes (scenario_data.yaml's shapes).

Each value is a scenario level applied first, then a perturbation scale applied around it -- see
sample_value, and reference_data/scenario_data.yaml for the low / mid / high levels.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SCENARIO_DATA = yaml.safe_load((Path(__file__).parent / "reference_data" / "scenario_data.yaml").read_text())

LEVELS = ["low", "mid", "high", "random"]
TARGETS = {  # granular target -> scenario levels it accepts
    "dr": LEVELS,
    "voll": LEVELS,
    "demand-growth": LEVELS,
    "demand-profile": LEVELS + list(SCENARIO_DATA["demand_profile"]["shapes"]),
    "re-profile": LEVELS,
    "hydro": LEVELS,
    "capex-start": LEVELS,
    "capex-pathway": LEVELS,
    "fuel-start": LEVELS,
    "fuel-pathway": LEVELS,
}
GROUPS = {
    "settings": ["dr", "voll"],
    "demand": ["demand-growth", "demand-profile"],
    "renewables": ["re-profile", "hydro"],
    "capex": ["capex-start", "capex-pathway"],
    "fuel": ["fuel-start", "fuel-pathway"],
    "cost": ["capex-start", "capex-pathway", "fuel-start", "fuel-pathway"],
    "all": list(TARGETS),
}


@dataclass(frozen=True)
class ScenarioOptions:
    """Per granular target: a scenario level and / or a perturbation scale (absent: leave as is)."""

    levels: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)

    def level(self, target):
        return self.levels.get(target)

    def scale(self, target):
        return self.scales.get(target)

    def active(self, *targets):
        """Whether any of targets has a level or a scale to apply."""
        return any(t in self.levels or self.scales.get(t) for t in targets)


def parse(scenario=None, perturbation=None, defaults=None):
    """ScenarioOptions from scenario (target-level specs) and perturbation (target-scale specs, or one
    number for every target). defaults: {target: level} for targets the scenario doesn't set."""
    levels = expand(scenario, "scenario", lambda target, value: check_level(target, value))
    if isinstance(perturbation, (int, float)):
        scales = dict.fromkeys(TARGETS, check_scale("all", perturbation))
    else:
        scales = expand(perturbation, "perturbation", lambda target, value: check_scale(target, float(value)))
    return ScenarioOptions(levels={**(defaults or {}), **levels}, scales=scales)


def expand(spec, kind, check):
    """{granular target: checked value}, broadest groups first so more specific targets override them."""
    entries = []
    for token in tokenize(spec):
        target, _, value = token.rpartition("-")
        if target not in TARGETS and target not in GROUPS:
            raise ValueError(f"unknown {kind} target in {token!r}; targets: {list(TARGETS)}, groups: {list(GROUPS)}")
        entries.append((GROUPS.get(target, [target]), value, token))
    out, set_by = {}, {}
    for granular, value, token in sorted(entries, key=lambda e: -len(e[0])):
        for target in granular:
            if set_by.get(target) is not None and len(set_by[target][0]) == len(granular) and out[target] != check(target, value):
                raise ValueError(f"conflicting {kind} values for {target!r}: {set_by[target][1]!r} and {token!r}")
            out[target] = check(target, value)
            set_by[target] = (granular, token)
    return out


def check_level(target, value):
    if value not in TARGETS[target]:
        raise ValueError(f"{target!r} scenario level must be one of {TARGETS[target]}, got {value!r}")
    return value


def check_scale(target, value):
    if not 0 <= value <= 1:
        raise ValueError(f"{target!r} perturbation scale must be between 0 and 1, got {value}")
    return value


def tokenize(spec):
    """Split a spec into normalized (lowercase, stripped) tokens."""
    if spec is None:
        return []
    parts = [spec] if isinstance(spec, str) else list(spec)
    return [t.strip().lower() for part in parts for t in str(part).replace("+", ",").split(",") if t.strip()]


def sample_value(start, levels, rng, level=None, scale=None):
    """start, set to a scenario level then perturbed by a scale. levels: {"low", "mid", "high"}.

    level low / mid / high: levels[level] x (1 + U(-noise, noise)); random: uniform between levels'
    extremes; None: start. scale (0-1): uniform around the value, up to scale x the distance to levels'
    extremes on each side; None / 0: no perturbation. Works elementwise on arrays (e.g. a cost
    trajectory), with one draw shared by every element so a trajectory stays smooth."""
    bounds_low = np.minimum(np.minimum(levels["low"], levels["high"]), levels["mid"])
    bounds_high = np.maximum(np.maximum(levels["low"], levels["high"]), levels["mid"])
    value = start
    if level == "random":
        value = bounds_low + rng.uniform() * (bounds_high - bounds_low)
    elif level is not None:
        value = levels[level] * (1 + rng.uniform(-SCENARIO_DATA["noise"], SCENARIO_DATA["noise"]))
    if scale:
        low = value - scale * (value - np.minimum(bounds_low, value))
        high = value + scale * (np.maximum(bounds_high, value) - value)
        value = low + rng.uniform() * (high - low)
    return value
