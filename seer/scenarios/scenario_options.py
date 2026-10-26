"""Scenario options: parse scenario and perturbation dicts into one ScenarioOptions, which create,
perturb and every sampler read.

    scenario = {"fuel:gas": "high", "capex:solar-utility|onwind": "high", "dr": "low"}
    perturbation = {"capex_pathway": 0.3}        # or one number for every parameter

Keys are a parameter or group, optionally qualified by ":key" (a tech, fuel or solar / wind; "|"
lists several). Values are a scenario level (low / mid / high / random, or a demand_profile shape)
or a perturbation scale (0 to 1). More specific entries override less specific ones: keyed over
unkeyed, then a parameter over a group, a smaller group over a larger one.
"""
from dataclasses import dataclass, field

import numpy as np

from .scenario_data import SCENARIO_DATA, load_table, parameter_levels

LEVELS = ["low", "mid", "high", "random"]
PARAMETERS = {  # parameter -> (scenario levels it accepts, what its key is)
    "dr": (LEVELS, None),
    "voll": (LEVELS, None),
    "demand_growth": (LEVELS, None),
    "demand_profile": (LEVELS + list(SCENARIO_DATA["demand_profile"]["shapes"]), None),
    "re_profile": (LEVELS, "category"),
    "hydro": (LEVELS, None),
    "capex_start": (LEVELS, "tech"),
    "capex_pathway": (LEVELS, "tech"),
    "fuel_start": (LEVELS, "fuel"),
    "fuel_pathway": (LEVELS, "fuel"),
}
GROUPS = {
    "settings": ["dr", "voll"],
    "demand": ["demand_growth", "demand_profile"],
    "renewables": ["re_profile", "hydro"],
    "capex": ["capex_start", "capex_pathway"],
    "fuel": ["fuel_start", "fuel_pathway"],
    "cost": ["capex_start", "capex_pathway", "fuel_start", "fuel_pathway"],
    "all": list(PARAMETERS),
}


def valid_keys(kind):
    return {"tech": set(load_table("technology/tech_cost_starts.csv")["tech"]),
            "fuel": set(load_table("fuels/fuel_cost_starts.csv")["fuel"]),
            "category": set(SCENARIO_DATA["re_profile"])}[kind]


@dataclass(frozen=True)
class ScenarioOptions:
    """Scenario levels and perturbation scales, by (parameter, key); key None applies to every key."""

    scenario: dict = field(default_factory=dict)
    perturbation: dict = field(default_factory=dict)

    def level(self, parameter, key=None):
        return self.scenario.get((parameter, key), self.scenario.get((parameter, None)))

    def scale(self, parameter, key=None):
        return self.perturbation.get((parameter, key), self.perturbation.get((parameter, None)))

    def active(self, *parameters):
        return any(p in parameters for p, _ in self.scenario) or any(p in parameters and s for (p, _), s in self.perturbation.items())

    def sample(self, parameter, initial=None, key=None, years=None, rng=None):
        """sample_value with this parameter's (and key's) scenario level and perturbation scale."""
        return sample_value(parameter, initial, self.level(parameter, key), self.scale(parameter, key), key, years, rng)


def parse(scenario=None, perturbation=None, defaults=None):
    """ScenarioOptions from scenario {parameter[:key]: level} and perturbation {parameter[:key]: scale}
    (or one scale for every parameter). defaults: {parameter: level} where the scenario sets none."""
    if isinstance(perturbation, (int, float)):
        perturbation = {"all": perturbation}
    for name, spec in [("scenario", scenario), ("perturbation", perturbation)]:
        if spec is not None and not isinstance(spec, dict):
            raise TypeError(f"{name} must be a dict (or, for perturbation, a number), got {type(spec).__name__}")

    # Step 1 - expand each entry to (parameter, key) pairs, ranked by how specific it is
    entries = []
    for kind, spec in [("scenario", scenario or {}), ("perturbation", perturbation or {})]:
        for name, value in spec.items():
            name, _, keys = name.partition(":")
            if name not in PARAMETERS and name not in GROUPS:
                raise ValueError(f"unknown {kind} parameter {name!r}; parameters: {list(PARAMETERS)}, groups: {list(GROUPS)}")
            parameters = GROUPS.get(name, [name])
            keys = keys.split("|") if keys else [None]
            if keys != [None]:
                key_kinds = {PARAMETERS[p][1] for p in parameters}
                if len(key_kinds) != 1 or None in key_kinds:
                    raise ValueError(f"{name!r} takes no single kind of key, so {name}:{'|'.join(keys)} is ambiguous")
                unknown = set(keys) - valid_keys(key_kinds.pop())
                if unknown:
                    raise ValueError(f"unknown key(s) {sorted(unknown)} for {name!r}")
            rank = (keys != [None], -len(parameters))  # keyed beats unkeyed; a smaller group beats a larger one
            entries += [(kind, rank, p, key, check(kind, p, value)) for key in keys for p in parameters]

    # Step 2 - apply least specific first so more specific entries override; same rank must agree
    out = {"scenario": {(p, None): level for p, level in (defaults or {}).items()}, "perturbation": {}}
    set_at = {}
    for kind, rank, parameter, key, value in sorted(entries, key=lambda e: e[1]):
        if set_at.get((kind, parameter, key)) == rank and out[kind][(parameter, key)] != value:
            raise ValueError(f"conflicting {kind} values for {parameter}{':' + key if key else ''}")
        out[kind][(parameter, key)] = value
        set_at[(kind, parameter, key)] = rank
    return ScenarioOptions(scenario=out["scenario"], perturbation=out["perturbation"])


def check(kind, parameter, value):
    if kind == "scenario" and value not in PARAMETERS[parameter][0]:
        raise ValueError(f"{parameter!r} scenario level must be one of {PARAMETERS[parameter][0]}, got {value!r}")
    if kind == "perturbation" and not (isinstance(value, (int, float)) and 0 <= value <= 1):
        raise ValueError(f"{parameter!r} perturbation must be a number between 0 and 1, got {value!r}")
    return value


def per_key_rng(rng=None):
    """key -> an rng seeded once per key from rng, so everything sharing a key (a tech, a fuel) gets
    the same draw."""
    rng, seeds = rng or np.random.default_rng(), {}

    def rng_for(key):
        seeds.setdefault(key, int(rng.integers(2**63)))
        return np.random.default_rng(seeds[key])
    return rng_for


def sample_value(parameter, initial=None, scenario=None, perturbation=None, key=None, years=None, rng=None):
    """Base value: the scenario level's value if given (random: uniform over the levels' range), else
    initial, else the parameter's mid. Then perturbed: uniform around the base, up to perturbation
    (0-1) x the distance to the levels' range on each side. Works elementwise on arrays (cost
    trajectories) with one draw shared by every element, so a trajectory stays smooth."""
    rng = rng or np.random.default_rng()
    levels = parameter_levels(parameter, key, years)
    low = np.minimum(np.minimum(levels["low"], levels["high"]), levels["mid"])
    high = np.maximum(np.maximum(levels["low"], levels["high"]), levels["mid"])

    if scenario == "random":
        value = low + rng.uniform() * (high - low)
    elif scenario is not None:
        value = levels[scenario] * (1 + rng.uniform(-SCENARIO_DATA["noise"], SCENARIO_DATA["noise"]))
    else:
        value = levels["mid"] if initial is None else initial

    if perturbation:
        below = value - perturbation * (value - np.minimum(low, value))
        above = value + perturbation * (np.maximum(high, value) - value)
        value = below + rng.uniform() * (above - below)
    return value
