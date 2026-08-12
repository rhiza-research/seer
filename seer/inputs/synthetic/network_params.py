"""Generates the seer schema file `network.csv`.

Topology: a random spanning path across zones guarantees every zone is connected (no isolated
zones), then every other possible zone pair is added as an extra line independently with
probability mesh_level (0 = just the path, 1 = fully connected).
"""

import itertools

import numpy as np
import pandas as pd

from .noise import noise_multiplier


def build_network(zones, peak_demand_mw, mesh_level, congestion_level, noise_pct):
    """Build network.csv: one row per transmission line (from, to, capacity).

    peak_demand_mw: {zone: peak demand (MW)}, used to size line capacity.
    mesh_level: probability [0, 1] that each possible line beyond the spanning path also exists.
    congestion_level: fraction [0, 1] by which a line's capacity is derated below the average of
    its two zones' peak demand.
    """
    if len(zones) < 2:
        return pd.DataFrame(columns=["from", "to", "capacity"])

    # spanning path: a random permutation of zones, connected in sequence, guarantees every
    # zone is reachable - no isolated zones
    order = np.random.permutation(zones)
    path_edges = {frozenset((order[i], order[i + 1])) for i in range(len(order) - 1)}

    # every other possible zone pair becomes a line independently with probability mesh_level
    other_edges = {
        frozenset(pair) for pair in itertools.combinations(zones, 2)
        if frozenset(pair) not in path_edges and np.random.uniform() < mesh_level
    }

    rows = []
    for edge in path_edges | other_edges:
        zone_from, zone_to = tuple(edge)
        avg_peak = (peak_demand_mw[zone_from] + peak_demand_mw[zone_to]) / 2
        capacity = avg_peak * (1 - congestion_level) * noise_multiplier(noise_pct)
        rows.append({"from": zone_from, "to": zone_to, "capacity": round(capacity, 3)})

    return pd.DataFrame(rows)
