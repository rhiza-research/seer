"""Trains the dispatch model (seer.models.ml.dispatch) on real synthetic seer-schema data,
instead of the fully-synthetic sample_batch() the model previously used for a quick spin.

Problem scale (n_zones, years, t_steps) is fixed for the whole run (see SCALE below). Data
generation is a one-time up-front phase: `build_dataset()` calls
seer.inputs.synthetic.generate_all_data n_instances times (each a fresh, independently-noised
seer-schema dataset at the fixed scale), and each is adapted into the tensors
seer.models.ml.dispatch expects. Training then samples (instance, day) pairs from that
pre-built pool every step - generating a fresh seer dataset per training step would be far too
slow (each call writes/reads a full year of CSVs; we only want to pay that cost n_instances
times total, not once per epoch).

Known v1 simplifications in the seer -> dispatch adapter (documented, not hidden):
    - Only EXISTING (non-expandable) generators/storage are used - this is a dispatch-only
      model for a fixed build-out, not a capacity-expansion one, so build options are dropped.
    - All existing dispatchable (non-renewable) generators in a zone are aggregated into
      DispatchNet's single per-node (c, gbar): gbar = summed capacity, c = capacity-weighted
      average marginal cost. Same for storage (sbar summed energy capacity, rbar summed power
      capacity). DispatchNet has no notion of multiple techs/units per node.
    - Renewable (RETech) generators are netted into a single per-node injection instead
      (injection = RE - load), matching the model's own production-positive sign convention -
      they're not counted in gbar/c at all.
    - voll is a fixed constant (1e6, matching seer.models.pyomo_model's convention) for every
      instance - the seer schema has no per-zone/per-instance VOLL parameter.
    - years is fixed to a single year - multi-year capacity trajectories aren't relevant here
      (fixed build-out), we just need one full year of hourly profiles to sample days from.
"""

import itertools
import tempfile

import numpy as np
import pandas as pd
import torch

from seer.inputs.synthetic import generate_all_data
from seer.models.ml.dispatch import DispatchNet, dispatch_forward, dispatch_loss

# ---- fixed problem scale for the whole run ----
N_ZONES = 3
T_STEPS = 24          # one day per training sample
YEAR = 2025
VOLL = 1_000_000.0    # matches seer.models.pyomo_model.abstract_model's fixed convention


# =============================================================================
# PTDF construction (a data-prep step: computed once per instance from network.csv,
# then passed straight through to dispatch_forward as a fixed, non-learned tensor)
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


class SeerDispatchInstance:
    """Everything needed to draw dispatch training samples from one generated seer-schema
    dataset: per-zone technical/cost aggregates (fixed for the whole year) plus a full year of
    hourly net injection, from which a random 24h window is sliced per training sample."""

    def __init__(self, zones, c, gbar, sbar, rbar, ptdf, line_cap, injection_by_hour):
        self.zones = zones                        # list[str], length N
        self.c = c                                 # (N,) marginal cost
        self.gbar = gbar                           # (N,) dispatchable capacity
        self.sbar = sbar                           # (N,) storage energy capacity
        self.rbar = rbar                           # (N,) storage power capacity
        self.ptdf = ptdf                           # (E, N)
        self.line_cap = line_cap                   # (E,)
        self.injection_by_hour = injection_by_hour  # (N, 8760) RE - load, MW

    @property
    def hours_in_year(self):
        return self.injection_by_hour.shape[1]


def _aggregate_existing(constants_df, costs_df, name_col, capacity_col, zones, retech_only=None):
    """Shared logic for generator_constants/storage_constants: sum capacity_col and
    capacity-weight-average marginal_cost per zone, existing (non-expandable) units only.
    retech_only: None (no filter), True (renewables only), or False (non-renewables only).
    """
    existing = constants_df[~constants_df["expandable"]]
    if retech_only is not None and "RETech" in existing.columns:
        existing = existing[existing["RETech"].astype(bool) == retech_only]

    marginal_cost = costs_df[costs_df["cost_type"] == "marginal_cost"].set_index(name_col)[str(YEAR)]

    capacity_by_zone = np.zeros(len(zones))
    cost_by_zone = np.zeros(len(zones))
    for zone_idx, zone in enumerate(zones):
        rows = existing[existing["zone"] == zone]
        if len(rows) == 0:
            continue
        capacities = rows[capacity_col].to_numpy(dtype=float)
        capacity_by_zone[zone_idx] = capacities.sum()
        costs = marginal_cost.reindex(rows[name_col]).fillna(0.0).to_numpy(dtype=float)
        weights = capacities / max(capacities.sum(), 1e-9)
        cost_by_zone[zone_idx] = float((costs * weights).sum())

    return capacity_by_zone, cost_by_zone


def load_seer_instance(data_path, n_zones=N_ZONES):
    """Read one generated seer-schema dataset and adapt it into a SeerDispatchInstance."""
    data_path = str(data_path).rstrip("/")

    zones_df = pd.read_csv(f"{data_path}/zones.csv")
    zones = zones_df["zone"].tolist()
    zone_index = {zone: i for i, zone in enumerate(zones)}
    assert len(zones) == n_zones, f"expected {n_zones} zones, got {len(zones)}"

    generator_constants = pd.read_csv(f"{data_path}/generator_constants.csv")
    generator_costs = pd.read_csv(f"{data_path}/generator_costs.csv")
    storage_constants = pd.read_csv(f"{data_path}/storage_constants.csv")
    storage_costs = pd.read_csv(f"{data_path}/storage_costs.csv")

    # ---- dispatchable generator aggregate (non-renewable, existing only) ----
    gbar, c = _aggregate_existing(
        generator_constants, generator_costs, "generator", "capacity", zones, retech_only=False
    )

    # ---- storage aggregate (existing only; energy + power capacity) ----
    sbar, _ = _aggregate_existing(
        storage_constants, storage_costs, "storage", "capacity_mwh", zones, retech_only=None
    )
    rbar, _ = _aggregate_existing(
        storage_constants, storage_costs, "storage", "capacity_mw", zones, retech_only=None
    )

    # ---- full-year hourly demand per zone (MW) ----
    demandforecast = pd.read_csv(f"{data_path}/demandforecast.csv")
    demandprofile = pd.read_csv(f"{data_path}/demandprofile.csv")
    peak_by_zone = demandforecast[
        (demandforecast["type"] == "peak") & (demandforecast["year"] == YEAR)
    ].set_index("zone")["value"]

    load_by_hour = np.zeros((n_zones, 8760))
    for zone, group in demandprofile.groupby("zone", sort=False):
        # rows are already in (season, day, hour) order = chronological order within the year
        load_by_hour[zone_index[zone], :] = group["value"].to_numpy() * peak_by_zone[zone]

    # ---- full-year hourly renewable generation per zone (MW), netted out of demand ----
    re_gbar, _ = _aggregate_existing(
        generator_constants, generator_costs, "generator", "capacity", zones, retech_only=True
    )
    renewable_profiles = pd.read_csv(f"{data_path}/renewable_profiles.csv")
    re_gen_by_hour = np.zeros((n_zones, 8760))
    if len(renewable_profiles) > 0:
        for zone, group in renewable_profiles.groupby("zone", sort=False):
            # multiple techs per zone would double-count re_gbar (which sums all RETech
            # capacity, not per-tech) - fine for now since the synthetic generator only has one
            # RE tech (PV); flagged here in case that changes
            group = group.sort_values(["month", "day", "hour"])
            re_gen_by_hour[zone_index[zone], :] = group["value"].to_numpy() * re_gbar[zone_index[zone]]

    injection_by_hour = re_gen_by_hour - load_by_hour  # (N, 8760), can be < 0

    # ---- network -> fixed-slots ptdf/line_cap (same convention as sample_network) ----
    network_df = pd.read_csv(f"{data_path}/network.csv")
    capacity_by_pair = {}
    for row in network_df.itertuples(index=False):
        i, j = zone_index[row[0]], zone_index[row[1]]  # columns: from, to, capacity
        capacity_by_pair[(min(i, j), max(i, j))] = row[2]

    all_pairs = list(itertools.combinations(range(n_zones), 2))
    branches = [
        (i, j, capacity_by_pair.get((i, j), 0.0), capacity_by_pair.get((i, j), 0.0))
        for (i, j) in all_pairs
    ]
    ptdf, line_cap = compute_ptdf(n_zones, branches, slack_bus=0)

    return SeerDispatchInstance(zones, c, gbar, sbar, rbar, ptdf, line_cap, injection_by_hour)


def build_dataset(n_instances, n_zones=N_ZONES, year=YEAR):
    """Generate n_instances fresh seer-schema datasets (the "make calls to generate the
    training data" step) and adapt each into a SeerDispatchInstance. Done once, up front."""
    instances = []
    for i in range(n_instances):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = generate_all_data(n_zones=n_zones, years=[year], output_path=tmp)
            instances.append(load_seer_instance(data_path, n_zones=n_zones))
        print(f"Generated training instance {i + 1}/{n_instances}", flush=True)
    return instances


def sample_batch(dataset, batch_size, t_steps=T_STEPS, device="cpu"):
    """Draw batch_size (instance, random day) pairs from a pre-built dataset."""
    n_zones = len(dataset[0].zones)
    d = np.zeros((batch_size, n_zones, t_steps))
    c = np.zeros((batch_size, n_zones))
    gbar = np.zeros((batch_size, n_zones))
    sbar = np.zeros((batch_size, n_zones))
    rbar = np.zeros((batch_size, n_zones))
    ptdf = np.zeros((batch_size, *dataset[0].ptdf.shape))
    line_cap = np.zeros((batch_size, *dataset[0].line_cap.shape))

    for b in range(batch_size):
        instance = dataset[np.random.randint(len(dataset))]
        max_start = instance.hours_in_year - t_steps
        start = np.random.randint(0, max_start + 1)

        d[b] = instance.injection_by_hour[:, start:start + t_steps]
        c[b] = instance.c
        gbar[b] = instance.gbar
        sbar[b] = instance.sbar
        rbar[b] = instance.rbar
        ptdf[b] = instance.ptdf
        line_cap[b] = instance.line_cap

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    voll = torch.full((batch_size,), VOLL, dtype=torch.float32, device=device)
    return to_tensor(d), to_tensor(c), to_tensor(gbar), to_tensor(sbar), to_tensor(rbar), voll, \
        to_tensor(ptdf), to_tensor(line_cap)


def train(n_epochs=200, n_instances=20, batch_size=64, lr=1e-3, device="cpu"):
    dataset = build_dataset(n_instances)

    model = DispatchNet(N_ZONES, T_STEPS).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        d, c, gbar, sbar, rbar, voll, ptdf, line_cap = sample_batch(dataset, batch_size, device=device)

        outputs = dispatch_forward(model, d, c, gbar, sbar, rbar, voll, ptdf, line_cap)
        loss, parts = dispatch_loss(outputs, c, voll, line_cap)

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
