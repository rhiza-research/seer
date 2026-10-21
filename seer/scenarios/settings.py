"""Sample and resample settings.csv's discount rate and value of lost load (low / mid / high from
reference_data/scenarios.yaml).
"""
import pandas as pd

from .spec import REFERENCE, sample_value


SETTINGS = ["dr", "voll"]


def sample(options, rng):
    """settings.csv with dr and voll sampled at the levels in options (random draws around mid)."""
    values = [sample_value(options[name], REFERENCE["settings"][name]["mid"], REFERENCE["settings"][name], rng, options["scale"]) for name in SETTINGS]
    return pd.DataFrame({"parameters": SETTINGS, "value": values})


def resample(instance, options, rng):
    """Resample dr and voll at the levels in options."""
    out = instance.copy()
    for name in SETTINGS:
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

