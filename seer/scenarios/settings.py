"""Sample and resample the discount rate and value of lost load.

Parameter targets: dr, voll
Seer schema files: settings.csv
"""
import pandas as pd

# set of parameters in seer's settings that can be perturbed
parameters = ["dr", "voll"]


def sample(scenario_options):
    """settings.csv per scenario_options."""
    return pd.DataFrame({"parameters": parameters, "value": [float(scenario_options.sample(p)) for p in parameters]})


def resample(instance, scenario_options):
    """Resample parameters based on scenario_options around original instance values."""
    out = instance.copy()
    for parameter in parameters:
        # skip if user has not specified this parameter as target
        if not scenario_options.active(parameter):
            continue
        # get initial value
        rows = out.settings["parameters"] == parameter
        initial = float(out.settings.loc[rows, "value"].iloc[0])
        # set new value
        out.settings.loc[rows, "value"] = float(scenario_options.sample(parameter, initial=initial))
    return out
