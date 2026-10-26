"""Sample and resample parameters in seer schema "settings".
"""
import pandas as pd

from .scenario_options import SCENARIO_DATA, sample_value

parameters = ["dr", "voll"]


def sample(options, rng):
    """settings.csv with dr and voll at options' levels and scales, starting from mid."""
    values = [float(sample_value(SCENARIO_DATA["settings"][name]["mid"], SCENARIO_DATA["settings"][name], rng, options.level(name), options.scale(name)))
              for name in parameters]
    return pd.DataFrame({"parameters": parameters, "value": values})


def resample(instance, options, rng):
    """Return a copy of instance with dr and voll set / perturbed per options, starting from the
    instance's own values (mid if absent)."""
    out = instance.copy()
    for name in parameters:
        if not options.active(name):
            continue
        levels = SCENARIO_DATA["settings"][name]
        rows = out.settings["parameters"] == name
        start = float(out.settings.loc[rows, "value"].iloc[0]) if rows.any() else levels["mid"]
        value = float(sample_value(start, levels, rng, options.level(name), options.scale(name)))
        if rows.any():
            out.settings.loc[rows, "value"] = value
        else:
            out.settings = pd.concat([out.settings, pd.DataFrame({"parameters": [name], "value": [value]})], ignore_index=True)
    return out
