"""Standardized long-format output table for storage state of charge on a solved pyomo seer
model (see seer/models/README.md for the output schema): storage_soc.
"""

import pandas as pd
import pyomo.environ as pyo


def build_storage_soc(instance):
    """Build storage_soc: hourly state of charge (MWh) for each storage unit.

    Returns a DataFrame with columns zone, tech, storage, year, hour, soc.
    """
    storage_zone = instance.storage_zone.extract_values()
    storage_carrier = instance.storage_carrier.extract_values()

    return pd.DataFrame([
        {
            "zone": storage_zone[s], "tech": storage_carrier[s], "storage": s,
            "year": year, "hour": hour, "soc": pyo.value(instance.soc[s, year, hour]),
        }
        for s in instance.storage for year, hour in instance.times
    ])
