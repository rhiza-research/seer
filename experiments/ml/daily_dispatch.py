"""PTDF construction and synthetic-topology sampling utilities.

The dispatch model itself (architecture, repair layers, forward pass, loss) has moved to
seer.models.ml.dispatch. Data generation and training now live in
experiments/ml/ml_dispatch_experiment.py. This file's remaining contents (compute_ptdf,
sample_network, sample_network_batch) are left here for now, to be relocated/consolidated with
experiments/ml/ptdf.py later.
"""

import itertools

import numpy as np
import torch

# =============================================================================
# PTDF construction utility
# =============================================================================

def compute_ptdf(n_nodes, branches, slack_bus=0):
    """
    Build the PTDF matrix from a simple branch list.

    branches: list of (from_bus, to_bus, susceptance, capacity)
    Returns:
        ptdf     : (E, n_nodes) numpy array
        line_cap : (E,) numpy array
    """
    E = len(branches)
    B_f = np.zeros((E, n_nodes))          # branch susceptance matrix
    line_cap = np.zeros(E)

    for e, (k, l, b_kl, cap) in enumerate(branches):
        B_f[e, k] += b_kl
        B_f[e, l] -= b_kl
        line_cap[e] = cap

    # Nodal susceptance (Laplacian-like) matrix B_bus = sum over branches
    B_bus = np.zeros((n_nodes, n_nodes))
    for (k, l, b_kl, _cap) in branches:
        B_bus[k, k] += b_kl
        B_bus[l, l] += b_kl
        B_bus[k, l] -= b_kl
        B_bus[l, k] -= b_kl

    # Remove slack bus row/col, invert, and re-insert a zero row/col for slack
    keep = [i for i in range(n_nodes) if i != slack_bus]
    B_reduced = B_bus[np.ix_(keep, keep)]
    B_inv_reduced = np.linalg.inv(B_reduced)

    B_inv_full = np.zeros((n_nodes, n_nodes))
    for i, ii in enumerate(keep):
        for j, jj in enumerate(keep):
            B_inv_full[ii, jj] = B_inv_reduced[i, j]
    # slack bus row/col stay zero -> slack bus injection has no angle effect

    ptdf = B_f @ B_inv_full   # (E, n_nodes)
    return ptdf, line_cap


def sample_network(n_nodes, mesh_level=0.3, capacity_range=(30.0, 80.0), slack_bus=0):
    """Sample one random connected topology over n_nodes nodes (same spanning-path + mesh
    construction as seer.inputs.synthetic.network_params.build_network): a random spanning path
    guarantees connectivity, then every other possible node pair becomes a line independently
    with probability mesh_level.

    Every possible node pair gets a fixed "slot" in the returned (E, n_nodes) ptdf / (E,)
    line_cap arrays (E = n_nodes*(n_nodes-1)/2) - inactive slots get capacity/susceptance 0, so
    they carry no flow and don't affect any other slot's PTDF row. This keeps ptdf/line_cap a
    fixed shape across instances even though which lines actually exist varies per instance.
    """
    all_pairs = list(itertools.combinations(range(n_nodes), 2))

    order = np.random.permutation(n_nodes)
    path_edges = {frozenset((order[i], order[i + 1])) for i in range(n_nodes - 1)}
    active_edges = {
        pair for pair in all_pairs
        if frozenset(pair) in path_edges or np.random.uniform() < mesh_level
    }

    branches = []
    for (k, l) in all_pairs:
        if (k, l) in active_edges:
            capacity = np.random.uniform(*capacity_range)
            branches.append((k, l, capacity, capacity))  # susceptance == capacity (same stand-in as ptdf.py)
        else:
            branches.append((k, l, 0.0, 0.0))  # inactive slot: no line, no flow, no effect on B_bus

    return compute_ptdf(n_nodes, branches, slack_bus=slack_bus)


def sample_network_batch(batch_size, n_nodes, device="cpu", **network_kwargs):
    """One independently-sampled topology per batch item. Returns ptdf (B, E, N) and
    line_cap (B, E) tensors, E = n_nodes*(n_nodes-1)//2."""
    ptdf_list, line_cap_list = [], []
    for _ in range(batch_size):
        ptdf_np, line_cap_np = sample_network(n_nodes, **network_kwargs)
        ptdf_list.append(ptdf_np)
        line_cap_list.append(line_cap_np)

    ptdf = torch.tensor(np.stack(ptdf_list), dtype=torch.float32, device=device)
    line_cap = torch.tensor(np.stack(line_cap_list), dtype=torch.float32, device=device)
    return ptdf, line_cap
