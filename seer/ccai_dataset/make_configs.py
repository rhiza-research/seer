"""Write one config per finalized country (configs/<country>.yaml), all from the same template, dropping only
the scenario elements a country's base instance gives nothing to act on:

    dr                 needs build options (it only weighs investment against operation)
    capex, capex:...   needs build options of those techs (with reference costs)
    demand_growth      needs more than one year
    re_profile         needs solar or wind profiles
    hydro              needs hydro profiles
    fuel               needs thermal generators with reference fuel prices
    voll, demand_profile   always apply

A grid that loses an element is dropped. A named scenario drops the elements that don't apply, and is itself
dropped if it loses one that defines it (NAMED_REQUIRES) or keeps only one element (then it's a one_at_a_time
case). A run left with no scenarios is dropped. Dispatch-only countries (no build options) perturb only demand,
renewables and fuel (marginal costs) in their random runs. Countries in SOLVE_YEARS solve their runs other than
baseline over only their first that many years.

Usage: python -m seer.ccai_dataset.make_configs [--models-dir ../seer-planning-models]
"""
import argparse
import copy
from pathlib import Path

import yaml

from ..schema import SeerInstance
from . import provenance
from .plan import base_values

CONFIG_DIR = Path(__file__).parent / "configs"
BASE_REPO = "https://github.com/rhiza-research/seer-planning-models"
BASE_DIR = "from_epm"
RENEWABLE_CAPEX = "capex:solar-utility|onwind|battery storage"

TEMPLATE = {
    "seed": 0,
    "significant_figures": 6,
    "cost_dataset": "wb_epm",
    "solve": {"model": "pyomo", "annualize_capex": True, "years": "all", "time_limit": 3600},
    "runs": [
        {"name": "baseline", "noise": 0, "solve": True},
        {"name": "one_at_a_time", "scenario": {"sweep_mode": "iterate", "elements": {
            element: ["low", "high"] for element in
            ["dr", "voll", "demand_growth", "demand_profile", "re_profile", "hydro", "capex", "fuel"]}},
         "noise": 0.02, "solve": True},
        {"name": "capex_x_fuel", "scenario": {"sweep_mode": "grid", "elements": {
            "capex": ["base", "low", "high"], "fuel": ["base", "low", "high"]}}, "noise": 0.02, "solve": True},
        {"name": "cheap_renewables", "scenario": {"elements": {RENEWABLE_CAPEX: "low", "fuel": "high"}},
         "noise": 0.02, "solve": True},
        {"name": "fossil_lock_in", "scenario": {"elements": {RENEWABLE_CAPEX: "high", "fuel": "low"}},
         "noise": 0.02, "solve": True},
        {"name": "high_demand_poor_resource", "scenario": {"elements": {
            "demand_growth": "high", "re_profile": "low", "hydro": "low"}}, "noise": 0.02, "solve": True},
        {"name": "expensive_capital", "scenario": {"elements": {"dr": "high", "capex": "high"}},
         "noise": 0.02, "solve": True},
        {"name": "random_solved", "perturbation": {"all": 0.5}, "iters": 20, "noise": 0.02, "solve": True},
        {"name": "random_low", "perturbation": {"all": 0.25}, "iters": 1190, "noise": 0.02, "solve": False},
        {"name": "random_mid", "perturbation": {"all": 0.5}, "iters": 1190, "noise": 0.02, "solve": False},
        {"name": "random_high", "perturbation": {"all": 1.0}, "iters": 1190, "noise": 0.02, "solve": False},
    ],
}
DISPATCH_ONLY_PERTURBATION = ["demand", "renewables", "fuel"]
SOLVE_YEARS = {"Ukraine": 10}  # country -> years its solved runs other than baseline solve (too slow at all years)
NAMED_REQUIRES = {  # named scenario -> the elements that define it
    "cheap_renewables": [RENEWABLE_CAPEX],
    "fossil_lock_in": [RENEWABLE_CAPEX],
    "high_demand_poor_resource": ["demand_growth"],
    "expensive_capital": ["capex"],
}


def applicable(instance):
    """The scenario elements instance gives something to act on (see the module docstring)."""
    years = [int(y) for y in instance.years["year"]]
    present = {(parameter, item) for parameter, item, base, _ in base_values(instance, years) if base is not None}
    capex_techs = {item for parameter, item in present if parameter == "capex_start"}
    out = {"voll", "demand_profile"}
    out |= {"dr"} if capex_techs else set()
    out |= {"capex"} if capex_techs else set()
    out |= {RENEWABLE_CAPEX} if capex_techs & set(RENEWABLE_CAPEX.split(":")[1].split("|")) else set()
    out |= {"demand_growth"} if ("demand_growth", None) in present else set()
    out |= {"re_profile"} if {("re_profile", "solar"), ("re_profile", "wind")} & present else set()
    out |= {"hydro"} if ("hydro", None) in present else set()
    out |= {"fuel"} if any(parameter == "fuel_start" for parameter, _ in present) else set()
    return out


def country_config(country, instance):
    """country's config: the template, less what its instance can't act on."""
    elements = applicable(instance)
    dispatch_only = "capex" not in elements
    runs = []
    for run in copy.deepcopy(TEMPLATE["runs"]):
        scenario = run.get("scenario")
        if scenario:
            kept = {k: v for k, v in scenario["elements"].items() if k in elements}
            if run["name"] in NAMED_REQUIRES:
                if any(k not in kept for k in NAMED_REQUIRES[run["name"]]) or len(kept) < 2:
                    continue
            elif not kept or (scenario.get("sweep_mode") == "grid" and len(kept) < len(scenario["elements"])):
                continue
            scenario["elements"] = kept
        if dispatch_only and "perturbation" in run:
            scale = run["perturbation"]["all"]
            run["perturbation"] = {group: scale for group in DISPATCH_ONLY_PERTURBATION}
        if country in SOLVE_YEARS and run["solve"] and run["name"] != "baseline":
            run["solve"] = {"years": SOLVE_YEARS[country]}
        runs.append(run)
    return {"country": country, "base": {"repo": BASE_REPO, "path": f"{BASE_DIR}/{country}"},
            **{k: v for k, v in TEMPLATE.items() if k != "runs"}, "runs": runs}


class ConfigDumper(yaml.SafeDumper):
    """Mappings in block style, lists of values inline ([low, high])."""


ConfigDumper.add_representer(list, lambda dumper, data: dumper.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=not any(isinstance(v, (dict, list)) for v in data)))


def to_yaml(config):
    """config as yaml, a blank line between runs."""
    text = yaml.dump(config, Dumper=ConfigDumper, sort_keys=False, default_flow_style=False, width=120, indent=2)
    return text.replace("\n- name:", "\n\n- name:").replace("\nsolve:", "\n\nsolve:").replace("\nruns:", "\n\nruns:")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, default=provenance.MODELS_DIR, help="local clone of seer-planning-models")
    args = parser.parse_args()
    for path in sorted((args.models_dir / BASE_DIR).iterdir()):
        if not path.is_dir():
            continue
        config = country_config(path.name, SeerInstance.read(path))
        (CONFIG_DIR / f"{path.name.replace(' ', '_')}.yaml").write_text(to_yaml(config))
        print(f"{path.name}: {', '.join(r['name'] for r in config['runs'])}")


if __name__ == "__main__":
    main()
