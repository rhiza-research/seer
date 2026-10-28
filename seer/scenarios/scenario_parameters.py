"""Scenario parameters: their reference data, definitions and low / mid / high levels, and parsing
scenario and perturbation dicts into one ScenarioOptions, which create, perturb and every sampler read.

    scenario = {"fuel:gas": "high", "capex:solar-utility|onwind": "high", "dr": "low"}
    perturbation = {"capex_pathway": 0.3}        # or one number for every parameter

Dict keys are a parameter or group, optionally for specific items after ":" (a tech, fuel or solar /
wind; "|" lists several). Values are a scenario level (low / mid / high / random, or a demand_profile
shape) or a perturbation scale (0 to 1). More specific entries override less specific ones: for an
item over for every item, then a parameter over a group, a smaller group over a larger one.

Cost levels come from one reference dataset (cost_dataset, default wb_epm: World Bank EPM models), per
table and item, falling back to the others in COST_DATASETS order where it has no rows (e.g. wb_epm has
starting costs but no pathways, so pathways come from ATB / AEO; atb has no fuels, so fuels come from pink_sheet).
"""
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------- reference data

DATA_DIR = Path(__file__).parent / "reference_data"
SCENARIO_DATA = yaml.safe_load((DATA_DIR / "scenario_data.yaml").read_text())
# cost tables' dataset column: wb_epm (World Bank EPM models), atb (NREL ATB; PyPSA where ATB has none),
# pink_sheet (fuels: World Bank commodity prices, AEO pathways). The chosen one first, then this order.
COST_DATASETS = ["atb", "pink_sheet", "wb_epm"]
DEFAULT_COST_DATASET = "wb_epm"


@cache
def load_table(name):
    return pd.read_csv(DATA_DIR / name)


def get_tech_parameter(tech, column):
    params = load_table("technology/tech_parameters.csv").set_index("tech")
    return params.loc[tech, column] if tech in params.index and pd.notna(params.loc[tech, column]) else None


# ---------------------------------------------------------------- parameters

SCENARIO_LEVELS = ["low", "mid", "high", "random"]
PARAMETERS = {  # parameter -> (scenario levels it accepts, what its items are)
    "dr": (SCENARIO_LEVELS, None),
    "voll": (SCENARIO_LEVELS, None),
    "demand_growth": (SCENARIO_LEVELS, None),
    "demand_profile": (SCENARIO_LEVELS + list(SCENARIO_DATA["demand_profile"]["shapes"]), None),
    "re_profile": (SCENARIO_LEVELS, "category"),
    "hydro": (SCENARIO_LEVELS, None),
    "capex_start": (SCENARIO_LEVELS, "tech"),
    "capex_pathway": (SCENARIO_LEVELS, "tech"),
    "fuel_start": (SCENARIO_LEVELS, "fuel"),
    "fuel_pathway": (SCENARIO_LEVELS, "fuel"),
    "line_capex": (SCENARIO_LEVELS, "line"),
}
# parameters that must be resampled together so that prices are consistent across techs / fuels
SHARED_DRAW_PARAMETERS = ["capex_start", "capex_pathway", "fuel_start", "fuel_pathway", "line_capex"]
PARAM_GROUPS = {
    "settings": ["dr", "voll"],
    "demand": ["demand_growth", "demand_profile"],
    "renewables": ["re_profile", "hydro"],
    "capex": ["capex_start", "capex_pathway"],
    "fuel": ["fuel_start", "fuel_pathway"],
    "cost": ["capex_start", "capex_pathway", "fuel_start", "fuel_pathway", "line_capex"],
    "all": list(PARAMETERS),
}


def get_valid_items(item_type):
    return {"tech": set(load_table(COST_TABLES["tech"][0])["tech"]),
            "fuel": set(load_table(COST_TABLES["fuel"][0])["fuel"]),
            "line": set(load_table(COST_TABLES["line"][0])["line"]),
            "category": set(SCENARIO_DATA["re_profile"])}[item_type]


def get_parameter_levels(parameter, item=None, years=None, cost_dataset=DEFAULT_COST_DATASET):
    """{"low", "mid", "high"} for parameter (and item: a tech, fuel or solar / wind), in model units;
    None if item has no reference. Cost parameters are trajectories over years, from cost_dataset
    (falling back to the others)."""
    if parameter in ("dr", "voll", "demand_growth", "demand_profile"):
        return {level: SCENARIO_DATA[parameter][level] for level in ["low", "mid", "high"]}
    if parameter == "re_profile":
        return SCENARIO_DATA["re_profile"].get(item)
    if parameter == "hydro":
        return {level: get_tech_parameter("hydro", f"cf_{level}") for level in ["low", "mid", "high"]}
    if parameter in ("capex_start", "capex_pathway"):
        levels = get_cost_start_and_pathway("tech", item, "capex", years, cost_dataset)
        if levels is None:
            return None
        per_mw = 1000 / (get_tech_parameter(item, "duration_hours") or 1)  # $/kW -> $/MW, or $/MWh of energy for storage
        return {level: v * per_mw for level, v in levels[0].items()} if parameter == "capex_start" else levels[1]
    if parameter in ("fuel_start", "fuel_pathway"):
        levels = get_cost_start_and_pathway("fuel", item, "price", years, cost_dataset)
        return None if levels is None else levels[0] if parameter == "fuel_start" else levels[1]
    if parameter == "line_capex":  # $/MW/km in years[0]; a line's capital is this x its length
        levels = get_cost_start_and_pathway("line", item, "capex", years, cost_dataset)
        return None if levels is None else levels[0]
    raise ValueError(f"unknown parameter {parameter!r}")


# ---------------------------------------------------------------- scenario options: parsing and sampling

@dataclass(frozen=True)
class ScenarioOptions:
    """Scenario levels and perturbation scales, by (parameter, item); item None applies to every item.
    rng draws in sequence; SHARED_DRAW_PARAMETERS draw from a generator seeded once per (parameter,
    item), so every value for the same item (e.g. a tech's capex in every zone) gets the same draw."""

    scenario: dict = field(default_factory=dict)
    perturbation: dict = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    item_seeds: dict = field(default_factory=dict)
    cost_dataset: str = DEFAULT_COST_DATASET

    def level(self, parameter, item=None):
        return self.scenario.get((parameter, item), self.scenario.get((parameter, None)))

    def scale(self, parameter, item=None):
        return self.perturbation.get((parameter, item), self.perturbation.get((parameter, None)))

    def active(self, *parameters):
        return any(p in parameters for p, _ in self.scenario) or any(p in parameters and s for (p, _), s in self.perturbation.items())

    def sample(self, parameter, initial=None, item=None, years=None):
        """sample_value with this parameter's (and item's) scenario level and perturbation scale."""
        rng = self.rng
        if parameter in SHARED_DRAW_PARAMETERS and item is not None:
            rng = np.random.default_rng(self.item_seeds.setdefault((parameter, item), int(self.rng.integers(2**63))))
        return sample_value(parameter, initial, self.level(parameter, item), self.scale(parameter, item), item, years, rng,
                            self.cost_dataset)


def parse(scenario=None, perturbation=None, defaults=None, seed=None, cost_dataset=DEFAULT_COST_DATASET):
    """ScenarioOptions from scenario {parameter[:item]: level} and perturbation {parameter[:item]: scale}
    (or one scale for every parameter). defaults: {parameter: level} where the scenario sets none;
    seed: for its random draws; cost_dataset: reference costs."""
    if cost_dataset not in COST_DATASETS:
        raise ValueError(f"cost_dataset must be one of {COST_DATASETS}, got {cost_dataset!r}")
    if isinstance(perturbation, (int, float)):
        perturbation = {"all": perturbation}
    for name, spec in [("scenario", scenario), ("perturbation", perturbation)]:
        if spec is not None and not isinstance(spec, dict):
            raise TypeError(f"{name} must be a dict (or, for perturbation, a number), got {type(spec).__name__}")

    # Step 1 - expand each entry to (parameter, item) pairs, ranked by how specific it is
    entries = []
    for kind, spec in [("scenario", scenario or {}), ("perturbation", perturbation or {})]:
        for name, value in spec.items():
            name, _, items = name.partition(":")
            if name not in PARAMETERS and name not in PARAM_GROUPS:
                raise ValueError(f"unknown {kind} parameter {name!r}; parameters: {list(PARAMETERS)}, groups: {list(PARAM_GROUPS)}")
            parameters = PARAM_GROUPS.get(name, [name])
            items = items.split("|") if items else [None]
            if items != [None]:
                item_types = {PARAMETERS[p][1] for p in parameters}
                if len(item_types) != 1 or None in item_types:
                    raise ValueError(f"{name!r} takes no single type of item, so {name}:{'|'.join(items)} is ambiguous")
                unknown = set(items) - get_valid_items(item_types.pop())
                if unknown:
                    raise ValueError(f"unknown item(s) {sorted(unknown)} for {name!r}")
            rank = (items != [None], -len(parameters))  # for an item beats for every item; a smaller group beats a larger one
            entries += [(kind, rank, p, item, check(kind, p, value)) for item in items for p in parameters]

    # Step 2 - apply least specific first so more specific entries override; same rank must agree
    out = {"scenario": {(p, None): level for p, level in (defaults or {}).items()}, "perturbation": {}}
    set_at = {}
    for kind, rank, parameter, item, value in sorted(entries, key=lambda e: e[1]):
        if set_at.get((kind, parameter, item)) == rank and out[kind][(parameter, item)] != value:
            raise ValueError(f"conflicting {kind} values for {parameter}{':' + item if item else ''}")
        out[kind][(parameter, item)] = value
        set_at[(kind, parameter, item)] = rank
    return ScenarioOptions(scenario=out["scenario"], perturbation=out["perturbation"], rng=np.random.default_rng(seed),
                           cost_dataset=cost_dataset)


def check(kind, parameter, value):
    if kind == "scenario" and value not in PARAMETERS[parameter][0]:
        raise ValueError(f"{parameter!r} scenario level must be one of {PARAMETERS[parameter][0]}, got {value!r}")
    if kind == "perturbation" and not (isinstance(value, (int, float)) and 0 <= value <= 1):
        raise ValueError(f"{parameter!r} perturbation must be a number between 0 and 1, got {value!r}")
    return value


def sample_value(parameter, initial=None, scenario=None, perturbation=None, item=None, years=None, rng=None,
                 cost_dataset=DEFAULT_COST_DATASET):
    """Return a value for parameter, using its low / mid / high levels from get_parameter_levels
    (for item, e.g. a tech, and over years, for a cost trajectory, from cost_dataset).

    Step 1 - base value, from either an explicit scenario or the initial value:
        scenario "low" / "mid" / "high": that level's value
        scenario "random": a uniformly random value between low and high
        no scenario: initial (or the mid level's value if there's no initial value)
    Step 2 - perturbation (0 to 1) around the base value:
        uniform in [base - perturbation x (base - low), base + perturbation x (high - base)]
    A cost trajectory (an array over years) uses one random draw for every year, so it stays smooth.
    """
    rng = rng or np.random.default_rng()
    levels = get_parameter_levels(parameter, item, years, cost_dataset)
    low = np.minimum(np.minimum(levels["low"], levels["high"]), levels["mid"])
    high = np.maximum(np.maximum(levels["low"], levels["high"]), levels["mid"])

    if scenario == "random":
        value = low + rng.uniform() * (high - low)
    elif scenario is not None:
        value = levels[scenario]
    else:
        value = levels["mid"] if initial is None else initial

    if perturbation:
        below = value - perturbation * (value - np.minimum(low, value))
        above = value + perturbation * (np.maximum(high, value) - value)
        value = below + rng.uniform() * (above - below)
    return value


# ---------------------------------------------------------------- cost trajectories

COST_TABLES = {  # cost table -> (starts, pathways) files
    "tech": ("technology/tech_cost_starts.csv", "technology/tech_cost_pathways.csv"),
    "fuel": ("fuels/fuel_cost_starts.csv", "fuels/fuel_cost_pathways.csv"),
    "line": ("network/line_cost_starts.csv", "network/line_cost_pathways.csv"),
}


def get_cost_start_and_pathway(table, item, parameter, years, cost_dataset=DEFAULT_COST_DATASET):
    """For a tech, fuel or line cost (table "tech" / "fuel" / "line"; item, parameter e.g. "CCGT", "capex"), return the
    starting year cost and the cost pathway for each level (low, mid, high), interpolated onto years.
    None if the cost isn't in the tables. Each of the starts and pathways tables uses cost_dataset's rows
    for item, else the first dataset in COST_DATASETS that has them.

    start: the cost in years[0]. pathway: the cost relative to years[0] (1.0 in years[0]).
    """
    def scenario_curves(name):
        df = load_table(name)
        rows = select_dataset(df[(df[table] == item) & (df["parameter"] == parameter)], cost_dataset).set_index("scenario")
        table_years = [int(c) for c in rows.columns if c.isdigit()]
        curves = {}
        for level in ["low", "mid", "high"]:
            curve = pd.Series(rows.loc[level, [str(y) for y in table_years]].to_numpy(dtype=float), index=table_years)
            curves[level] = curve.reindex(sorted(set(table_years) | set(years))).interpolate(method="index").ffill().bfill().loc[years].to_numpy()
        return curves

    starts_file, pathways_file = COST_TABLES[table]
    starts = load_table(starts_file)
    if not ((starts[table] == item) & (starts["parameter"] == parameter)).any():
        return None

    # Step 1 - start (base-year cost) and pathway for each level, onto years
    start, pathway = scenario_curves(starts_file), scenario_curves(pathways_file)

    # Step 2 - determine starting year cost using reference data (2025 cost + mid cost trajectory);
    # renormalize trajectory to starting year
    start = {level: s[0] * pathway["mid"][0] for level, s in start.items()}
    pathway = {level: p / p[0] for level, p in pathway.items()}
    return start, pathway


def select_dataset(rows, cost_dataset):
    """rows from cost_dataset, else from the first dataset in COST_DATASETS that has any (a table with no
    dataset column is one dataset)."""
    if "dataset" not in rows.columns:
        return rows
    for dataset in [cost_dataset] + [d for d in COST_DATASETS if d != cost_dataset]:
        if (rows["dataset"] == dataset).any():
            return rows[rows["dataset"] == dataset]
    return rows


def get_vom_trajectory(tech, years):
    """tech's mid variable O&M trajectory ($/MWh); 0 if it has none."""
    levels = get_cost_start_and_pathway("tech", tech, "vom", years)
    return np.zeros(len(years)) if levels is None else levels[0]["mid"] * levels[1]["mid"]
