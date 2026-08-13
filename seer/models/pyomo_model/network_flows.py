"""Standardized long-format output table for line flows on a solved pyomo seer model (see
seer/models/README.md for the output schema): network_flows.
"""

import pandas as pd
import pyomo.environ as pyo


def build_network_flows(instance):
    """Build network_flows: hourly flow (MW) on each line of a solved pyomo model instance.

    Returns a DataFrame with columns from, to, year, hour, flow. Positive flow = flowing from
    `from` to `to` (see abstract_model.py's line_flow variable); empty if the model has no lines.
    """
    line_from = instance.line_from.extract_values()
    line_to = instance.line_to.extract_values()

    return pd.DataFrame([
        {
            "from": line_from[line], "to": line_to[line],
            "year": year, "hour": hour, "flow": pyo.value(instance.line_flow[line, year, hour]),
        }
        for line in instance.lines for year, hour in instance.times
    ])
