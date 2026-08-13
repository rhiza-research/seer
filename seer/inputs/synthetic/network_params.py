"""Generates the seer schema file `network.csv`.

Creates randomized topology and line capacities with guaranteed connectivity.
"""

import itertools

import numpy as np
import pandas as pd

from .noise import noise_multiplier


def build_network(zones, peak_demand_mw, mesh_level, congestion_level, noise_pct):
    """Build network.
    A single random spanning path is created to guarantee zones are connected.
    Additional mesh connections are added with probability mesh_level.
    Line connecting zones 1 and 2 is given capacity proportional to their average peak demand scaled
    down by congestion_level. 

    mesh_level: probability [0, 1] that each possible line beyond the spanning path also exists.
    congestion_level: fraction [0, 1] by which a line's capacity is derated below the average of
    its two zones' peak demand.
    """
    # single node grid has no lines
    if len(zones) < 2:
        return pd.DataFrame(columns=["from", "to", "capacity"])

    # spanning path: a random permutation of zones, connected in sequence, guarantees that
    # no zone is an island.
    order = np.random.permutation(zones)
    path_edges = {frozenset((order[i], order[i + 1])) for i in range(len(order) - 1)}

    # add meshed lines with probability mesh_level
    # 0 = no meshed lines, 1 = fully connected
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
