"""Generates SEER synthetic data and adapts it into dispatch-model (seer.models.ml.dispatch)
training tensors.

One SEER instance = one full year, one network topology. Since a year has 365 days and the
dispatch model trains on single-day (T=24) windows, one instance yields 365 training examples
-- so a handful of instances (each its own random network/generator mix) is enough to sample
from.

Known v1 simplifications (documented, not hidden):
    - Only EXISTING (non-expandable) generators/storage are used -- this is a dispatch-only
      model for a fixed build-out, not a capacity-expansion one, so build options are dropped.
    - Exactly ONE non-renewable generator is kept per zone, chosen at random from whichever
      existing units are there (not aggregated/blended) -- this lets different zones end up
      with genuinely different marginal costs, which is what we want to see the model learn to
      exploit, without needing multiple generator "slots" per node in DispatchNet.
    - Storage units at a zone ARE summed (capacity_mwh, capacity_mw) -- storage has no cost in
      this model, so blending units together doesn't hide anything interesting the way blending
      generators of different cost would.
    - Renewable (PV) generation is netted out of load into a single per-zone net_load, then
      flipped to injection (production-positive, matching the model's own sign convention).
    - voll is a fixed constant (1e6, matching seer.models.pyomo_model's convention).
    - years is fixed to a single year -- we just need one full year of hourly profiles.
"""

import pickle
import tempfile

import numpy as np
import pandas as pd
import torch

from seer.inputs.synthetic import generate_all_data

N_ZONES = 3
YEAR = 2025
VOLL = 1_000_000.0    # matches seer.models.pyomo_model.abstract_model's fixed convention


# =============================================================================
# SEER -> dispatch adapter
# =============================================================================

class DispatchInstance:
    """One SEER-generated year/topology, adapted into per-zone dispatch tensors. injection_by_hour
    covers the full year, so 365 non-overlapping days of training data can be sliced from it.
    No PTDF here -- that's derived from capacity_matrix at experiment time, not stored (see
    ml_dispatch_experiment.py); it's a modeling choice, not intrinsic to the data itself."""

    def __init__(self, zones, marginal_cost, gen_capacity, storage_energy_capacity,
                 storage_power_capacity, capacity_matrix, injection_by_hour):
        self.zones = zones                                     # list[str], length N
        self.marginal_cost = marginal_cost                     # (N,)
        self.gen_capacity = gen_capacity                       # (N,)
        self.storage_energy_capacity = storage_energy_capacity  # (N,)
        self.storage_power_capacity = storage_power_capacity   # (N,)
        self.capacity_matrix = capacity_matrix                 # (N, N) symmetric, zero diagonal
        self.injection_by_hour = injection_by_hour             # (N, 8760)

    @property
    def n_days(self):
        return self.injection_by_hour.shape[1] // 24


def dispatch_adaptor(data_path, n_zones=N_ZONES, rng=None):
    """Reads one generated SEER dataset and adapts it into a DispatchInstance."""
    data_path = str(data_path).rstrip("/")
    rng = rng or np.random.default_rng()

    zones_df = pd.read_csv(f"{data_path}/zones.csv")
    zones = zones_df["zone"].tolist()
    zone_index = {zone: i for i, zone in enumerate(zones)}
    assert len(zones) == n_zones, f"expected {n_zones} zones, got {len(zones)}"

    generator_constants = pd.read_csv(f"{data_path}/generator_constants.csv")
    generator_costs = pd.read_csv(f"{data_path}/generator_costs.csv")
    storage_constants = pd.read_csv(f"{data_path}/storage_constants.csv")

    marginal_cost_by_gen = generator_costs[
        generator_costs["cost_type"] == "marginal_cost"
    ].set_index("generator")[str(YEAR)]

    # ---- one non-renewable generator per zone, picked at random (not blended) -- so different
    #      zones can end up with genuinely different marginal costs. Capacity is still the SUM
    #      of the whole existing fleet at that zone, though -- only the cost rate is randomly
    #      picked, not the amount of capacity, otherwise we'd throw away most of the zone's
    #      actual capacity and create an artificial supply shortfall (see conversation).
    existing_nonre = generator_constants[
        (~generator_constants["expandable"]) & (~generator_constants["RETech"])
    ]
    gen_capacity = np.zeros(n_zones)
    marginal_cost = np.zeros(n_zones)
    for zone, group in existing_nonre.groupby("zone"):
        chosen = group.iloc[rng.integers(len(group))]   # random pick, for its cost only
        zi = zone_index[zone]
        gen_capacity[zi] = group["capacity"].sum()      # full existing fleet capacity
        marginal_cost[zi] = marginal_cost_by_gen.get(chosen["generator"], 0.0)

    # ---- storage: sum existing capacity per zone (storage is free to use, so blending units
    #      together doesn't hide anything the way blending generator costs would)
    existing_storage = storage_constants[~storage_constants["expandable"]]
    storage_energy_capacity = np.zeros(n_zones)
    storage_power_capacity = np.zeros(n_zones)
    for zone, group in existing_storage.groupby("zone"):
        zi = zone_index[zone]
        storage_energy_capacity[zi] = group["capacity_mwh"].sum()
        storage_power_capacity[zi] = group["capacity_mw"].sum()

    # ---- full-year hourly load per zone (MW)
    demandforecast = pd.read_csv(f"{data_path}/demandforecast.csv")
    demandprofile = pd.read_csv(f"{data_path}/demandprofile.csv")
    peak_by_zone = demandforecast[
        (demandforecast["type"] == "peak") & (demandforecast["year"] == YEAR)
    ].set_index("zone")["value"]

    load_by_hour = np.zeros((n_zones, 8760))
    for zone, group in demandprofile.groupby("zone", sort=False):
        # rows are already in (season, day, hour) order = chronological order within the year
        load_by_hour[zone_index[zone], :] = group["value"].to_numpy() * peak_by_zone[zone]

    # ---- full-year hourly PV generation per zone (MW)
    pv_capacity_by_zone = generator_constants[
        (~generator_constants["expandable"]) & (generator_constants["RETech"])
    ].groupby("zone")["capacity"].sum()
    renewable_profiles = pd.read_csv(f"{data_path}/renewable_profiles.csv")
    pv_by_hour = np.zeros((n_zones, 8760))
    if len(renewable_profiles) > 0:
        for zone, group in renewable_profiles.groupby("zone", sort=False):
            group = group.sort_values(["month", "day", "hour"])
            pv_by_hour[zone_index[zone], :] = group["value"].to_numpy() * pv_capacity_by_zone.get(zone, 0.0)

    # net_load = demand - PV (consumption-positive); injection = -net_load (production-positive,
    # matching seer.models.ml.dispatch's sign convention)
    net_load_by_hour = load_by_hour - pv_by_hour
    injection_by_hour = -net_load_by_hour

    # ---- network -> NxN capacity matrix (symmetric, zero diagonal; canonical representation,
    #      see seer.models.ml.dispatch's module docstring) + PTDF
    network_df = pd.read_csv(f"{data_path}/network.csv")
    capacity_matrix = np.zeros((n_zones, n_zones))
    for row in network_df.itertuples(index=False):
        i, j = zone_index[row[0]], zone_index[row[1]]  # columns: from, to, capacity
        capacity_matrix[i, j] = capacity_matrix[j, i] = row[2]

    return DispatchInstance(zones, marginal_cost, gen_capacity, storage_energy_capacity,
                             storage_power_capacity, capacity_matrix, injection_by_hour)


def build_dataset(n_instances, n_zones=N_ZONES, year=YEAR, seed=None):
    """Generate n_instances fresh SEER datasets (each its own random network/generator mix) and
    adapt each into a DispatchInstance. Done once, up front -- training then samples
    (instance, day) pairs from this pool."""
    rng = np.random.default_rng(seed)
    instances = []
    for i in range(n_instances):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = generate_all_data(n_zones=n_zones, years=[year], output_path=tmp)
            instances.append(dispatch_adaptor(data_path, n_zones=n_zones, rng=rng))
        print(f"Generated instance {i + 1}/{n_instances}", flush=True)
    return instances


def sample_batch(dataset, batch_size, t_steps=24, device="cpu"):
    """Draw batch_size (instance, random day) pairs -- each instance covers a full year, so it
    yields 365 non-overlapping days of dispatch training data. No PTDF here -- that's a
    per-instance, capacity_matrix-derived quantity computed at experiment time (see
    ml_dispatch_experiment.py), not part of the raw dispatch data itself."""
    n_zones = len(dataset[0].zones)
    injection = np.zeros((batch_size, n_zones, t_steps))
    marginal_cost = np.zeros((batch_size, n_zones))
    gen_capacity = np.zeros((batch_size, n_zones))
    storage_energy_capacity = np.zeros((batch_size, n_zones))
    storage_power_capacity = np.zeros((batch_size, n_zones))
    capacity_matrix = np.zeros((batch_size, n_zones, n_zones))
    instance_indices = np.zeros(batch_size, dtype=int)   # so callers can align a precomputed PTDF

    for b in range(batch_size):
        idx = np.random.randint(len(dataset))
        instance = dataset[idx]
        day = np.random.randint(instance.n_days)      # one of the 365 days, not an arbitrary window
        start = day * t_steps

        instance_indices[b] = idx
        injection[b] = instance.injection_by_hour[:, start:start + t_steps]
        marginal_cost[b] = instance.marginal_cost
        gen_capacity[b] = instance.gen_capacity
        storage_energy_capacity[b] = instance.storage_energy_capacity
        storage_power_capacity[b] = instance.storage_power_capacity
        capacity_matrix[b] = instance.capacity_matrix

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    voll = torch.full((batch_size,), VOLL, dtype=torch.float32, device=device)
    return (to_tensor(injection), to_tensor(marginal_cost), to_tensor(gen_capacity),
            to_tensor(storage_energy_capacity), to_tensor(storage_power_capacity), voll,
            to_tensor(capacity_matrix), instance_indices)


# =============================================================================
# Save/load a generated dataset to/from disk (SEER generation is slow -- do it once, then
# reuse the pickled dataset across training runs)
# =============================================================================

def save_dataset(dataset, path):
    with open(path, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_and_save(path, n_instances, n_zones=N_ZONES, year=YEAR, seed=None):
    """Generate n_instances DispatchInstances and pickle them to path, for later reuse via
    load_dataset (see ml_dispatch_experiment.py)."""
    dataset = build_dataset(n_instances, n_zones=n_zones, year=year, seed=seed)
    save_dataset(dataset, path)
    return dataset
