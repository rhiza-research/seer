"""Trains the dispatch model (seer.models.ml.dispatch) on SEER-generated synthetic data.
See generate_dispatch_data.py for how that data is generated/adapted."""

import numpy as np
import torch

from experiments.ml.dispatch.generate_dispatch_data import N_ZONES, build_dataset, sample_batch
from seer.models.ml.dispatch import DispatchNet, canonical_edge_pairs, dispatch_forward, dispatch_loss

T_STEPS = 24   # one day per training sample


# =============================================================================
# PTDF: a modeling choice (susceptance == capacity stand-in), not intrinsic to the dispatch
# data itself -- computed here from each instance's capacity_matrix, once, not stored with it.
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


def compute_ptdfs(dataset):
    """PTDF depends only on topology (capacity_matrix), not on which day is sampled -- compute
    it once per instance, not once per training batch."""
    n_zones = len(dataset[0].zones)
    ptdfs = []
    for instance in dataset:
        branches = [
            (i, j, instance.capacity_matrix[i, j], instance.capacity_matrix[i, j])  # stand-in
            for (i, j) in canonical_edge_pairs(n_zones)
        ]
        ptdf, _ = compute_ptdf(n_zones, branches, slack_bus=0)
        ptdfs.append(ptdf)
    return np.stack(ptdfs)   # (n_instances, E, N)


def train(n_epochs=200, n_instances=20, batch_size=64, lr=1e-3, device="cpu"):
    dataset = build_dataset(n_instances)
    ptdfs = compute_ptdfs(dataset)

    model = DispatchNet(N_ZONES, T_STEPS).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        injection, marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity, \
            voll, capacity_matrix, instance_indices = sample_batch(dataset, batch_size, t_steps=T_STEPS, device=device)
        ptdf = torch.tensor(ptdfs[instance_indices], dtype=torch.float32, device=device)

        outputs = dispatch_forward(model, injection, marginal_cost, gen_capacity,
                                    storage_energy_capacity, storage_power_capacity, voll, ptdf, capacity_matrix)
        loss, parts = dispatch_loss(outputs, marginal_cost, voll, capacity_matrix)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d} | loss {loss.item():10.2f} | "
                  f"gen_cost {parts['gen_cost']:9.2f} | "
                  f"ll_cost {parts['ll_cost']:9.2f} | "
                  f"line_penalty {parts['line_penalty']:9.2f}")

    return model, dataset, losses


if __name__ == "__main__":
    train()
