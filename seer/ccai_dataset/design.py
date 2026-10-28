"""What a run makes: the config's runs, expanded to one entry per perturbation.

Each run in a config:

    - name: capex_x_fuel          # unique; ids are <name>_00000, ...
      scenario:                   # optional; none = the base instance
        sweep_mode: grid          # grid: every combination of the elements' values
                                  # iterate: each element's values alone, the others at base
        elements:                 # perturb's scenario keys -> values (low / mid / high / random, a
          capex: [base, low, high]  # demand_profile shape, or base: the country's own value)
          fuel: [base, low, high]
      perturbation: {all: 0.5}    # optional; perturb's perturbation scales, applied after the scenario
      iters: 1                    # perturbations per scenario, each with its own seed
      noise: 0.02                 # relative noise on every parameter, after scenario and perturbation (0: none)
      solve: true

Every entry's seed derives from (config seed, run name, index), so adding, removing or reordering runs
never changes another run's entries.
"""
import itertools
import zlib

import numpy as np

SWEEP_MODES = ["grid", "iterate"]
RUN_KEYS = {"name", "scenario", "perturbation", "iters", "noise", "solve"}


def entry_seed(root_seed, run_name, index):
    return int(np.random.SeedSequence([root_seed, zlib.crc32(run_name.encode()), index]).generate_state(1, np.uint64)[0])


def scenarios(scenario):
    """[scenario dict] a run's scenario block expands to; base values are left out (the country's own)."""
    if not scenario:
        return [{}]
    mode, elements = scenario.get("sweep_mode", "grid"), scenario["elements"]
    if mode not in SWEEP_MODES:
        raise ValueError(f"sweep_mode must be one of {SWEEP_MODES}, got {mode!r}")
    elements = {key: values if isinstance(values, list) else [values] for key, values in elements.items()}
    if mode == "iterate":
        combos = [{key: value} for key, values in elements.items() for value in values]
    else:
        combos = [dict(zip(elements, values)) for values in itertools.product(*elements.values())]
    return [{key: value for key, value in combo.items() if value != "base"} for combo in combos]


def scenario_name(scenario):
    return ",".join(f"{key}={value}" for key, value in scenario.items()) or "base"


def build_entries(config):
    """Every perturbation config's runs make: [{id, run, scenario_name, scenario, perturbation, noise, seed, solve}]."""
    entries, names = [], set()
    for run in config["runs"]:
        unknown = set(run) - RUN_KEYS
        if unknown or "name" not in run or "noise" not in run:
            raise ValueError(f"run {run.get('name')!r}: needs a name and a noise; unknown keys {sorted(unknown)} "
                             f"(allowed: {sorted(RUN_KEYS)})")
        if run["name"] in names:
            raise ValueError(f"run name {run['name']!r} is used twice")
        names.add(run["name"])
        index = 0
        for scenario in scenarios(run.get("scenario")):
            for _ in range(run.get("iters", 1)):
                entries.append({"id": f"{run['name']}_{index:05d}", "run": run["name"], "scenario_name": scenario_name(scenario),
                                "scenario": scenario, "perturbation": run.get("perturbation", {}),
                                "noise": run["noise"], "seed": entry_seed(config["seed"], run["name"], index),
                                "solve": bool(run.get("solve", False))})
                index += 1
    return entries
