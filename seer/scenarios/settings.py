"""Sample and resample parameters in seer schema "settings". Parameters: dr, voll.
"""
import pandas as pd

parameters = ["dr", "voll"]


def sample(options, rng=None):
    """settings.csv per options."""
    return pd.DataFrame({"parameters": parameters, "value": [float(options.sample(p, rng=rng)) for p in parameters]})


def resample(instance, options, rng=None):
    """Copy of instance with settings set / perturbed per options, from the instance's own values."""
    out = instance.copy()
    for parameter in parameters:
        if not options.active(parameter):
            continue
        rows = out.settings["parameters"] == parameter
        initial = float(out.settings.loc[rows, "value"].iloc[0]) if rows.any() else None
        value = float(options.sample(parameter, initial=initial, rng=rng))
        if rows.any():
            out.settings.loc[rows, "value"] = value
        else:
            out.settings = pd.concat([out.settings, pd.DataFrame({"parameters": [parameter], "value": [value]})], ignore_index=True)
    return out
