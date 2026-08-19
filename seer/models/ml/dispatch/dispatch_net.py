"""Daily dispatch model with repair layers for feasibility.

Hourly dispatch model incorporating network, storage, and lost load.

-----------------------------------------------------------------------------
Architecture Summary
-----------------------------------------------------------------------------
B = batch size
N = number of nodes
T = number of time steps
E = number of lines
Each node is assumed to have a single dispatchable generator, storage unit, and aggregate load.

Sign convention: production/injection is POSITIVE, consumption/extraction is NEGATIVE. So
injection < 0 means net consumption (more load than RE), and storage realized_delta > 0 means
discharging (< 0 = charging). gen_output/lost_load are always >= 0 because they can only
produce power.

Inputs (per instance, batched):
    injection               : (B, N, T)  net injection at each node (RE - load), can be < 0
    marginal_cost           : (B, N)     marginal cost of each dispatchable generator
    gen_capacity            : (B, N)     capacity of each dispatchable generator
    storage_energy_capacity : (B, N)     energy capacity of each storage unit
    storage_power_capacity  : (B, N)     power (rate) capacity of each storage unit
    voll                    : (B,)       value of lost load; identical for all space-time
                                         (can be thought of as cost of a lost load "generator")
    line_capacity           : (B, E)     thermal line capacities, one per instance

Outputs:
    gen_output : (B, N, T)  dispatchable generator output
    lost_load  : (B, T)     aggregate lost-load
    soc        : (B, N, T)  storage state of charge
    line_flows : (B, E, T)  line flows

Repair sequence:
    1. Storage repair: tanh-bound the raw output to a power-capacity fraction in [-1, 1],
       rescale into SOC-space (divide by duration_hours = energy_capacity/power_capacity),
       cumsum over time, then bound into (0, 1) with a sharpened sigmoid to get a guaranteed-
       feasible SOC trajectory (starting from SOC=0 at midnight; no constraint forces it back to
       0 by the end -- see storage_repair's docstring). Realized deltas are backed out by
       differencing, then everything is scaled from fractional to absolute MW/MWh in
       dispatch_forward.
    2. Aggregate power balance repair (paper's Theorem-1-style layer,
       generalized to N+1 "generators": N dispatchable + 1 aggregate
       lost-load unit). This is done PER TIMESTEP, treating storage's
       already-fixed realized output as part of the effective balance target.
    3. PTDF-based flow reconstruction from nodal net injections. Because
       aggregate balance holds exactly (sum injections = 0 every t), the
       resulting flows automatically satisfy KCL -- no repair needed there.
       Line capacity is a SOFT constraint (penalized in the loss), matching
       the original paper's treatment of thermal limits.

Known v1 simplifications (documented, not hidden):
    - Balance is enforced in AGGREGATE (system-wide) only, not nodally.
      Excess dispatchable generation is only discouraged via cost, not
      hard-repaired (per user's design choice) -- deficit side is exact,
      surplus side is cost-incentivized only.
    - Lost load is a single AGGREGATE generator (bound = system demand at
      time t), then allocated back to nodes pro-rata by demand share for
      PTDF purposes. This is a modeling simplification -- a nodal lost
      load formulation would remove the need for this allocation step.
    - DispatchNet now sees a per-node topology summary too (each node's total
      connected line capacity, derived from line_capacity via a fixed node-line
      incidence matrix - see below) alongside marginal_cost/gen_capacity/
      storage_energy_capacity/storage_power_capacity, in addition to
      line_capacity/ptdf still driving the PTDF flow reconstruction and
      line-capacity penalty as before. This was Option 2 of a few considered
      (see conversation): cheap, contained to DispatchNet, but still a plain
      MLP with fixed N and only a coarse per-node summary (not the full graph
      structure - which line connects to which specific neighbor is not
      visible, just each node's total connected capacity). A GNN/PTDF-
      conditioned encoder taking the full graph (and able to generalize
      across varying N) remains the follow-on step if this coarser summary
      isn't enough - not done here. Not yet re-validated empirically against
      the earlier topology-blind run (see conversation for that baseline:
      fixed topology converged to line_penalty flat at 0.00, varying topology
      never settled, bouncing roughly 0-1000 epoch to epoch) - that
      comparison should be re-run with this change in.
    - Initial SOC is fixed at 0 for every instance (start of the midnight-to-midnight
      horizon); there is no constraint requiring the battery to return to any
      particular SOC by the end of the horizon.
    - The sigmoid used to bound the SOC trajectory into (0, 1) reshapes the whole
      trajectory somewhat, not just out-of-range excursions (see storage_repair's
      docstring) -- accepted for v1; the network can learn raw outputs that
      compensate for this during training.
-----------------------------------------------------------------------------
"""

import itertools

import torch
import torch.nn as nn


def _build_incidence(n_nodes):
    """(N, E) 0/1 incidence matrix, E = n_nodes*(n_nodes-1)//2. incidence[i, e] = 1 if line slot
    e touches node i. Slot order must match every other fixed-slots user in this codebase
    (itertools.combinations(range(n_nodes), 2) - see experiments/ml/daily_dispatch.py's
    sample_network and experiments/ml/ml_dispatch_experiment.py's network adapter)."""
    all_pairs = list(itertools.combinations(range(n_nodes), 2))
    incidence = torch.zeros(n_nodes, len(all_pairs))
    for e, (i, j) in enumerate(all_pairs):
        incidence[i, e] = 1.0
        incidence[j, e] = 1.0
    return incidence


# =============================================================================
# Model
# =============================================================================

class DispatchNet(nn.Module):
    def __init__(self, n_nodes, t_steps, hidden=256, n_layers=4):
        super().__init__()
        self.n = n_nodes
        self.t = t_steps

        # fixed (not learned), moves with .to(device) since it's a buffer
        self.register_buffer("incidence", _build_incidence(n_nodes))  # (N, E)

        input_dim = (
            n_nodes * t_steps      # injection
            + n_nodes               # marginal_cost
            + n_nodes               # gen_capacity
            + n_nodes               # storage_energy_capacity
            + n_nodes               # storage_power_capacity
            + n_nodes               # connected_capacity (topology summary, see module docstring)
            + 1                      # voll
        )
        # output: (n+1) x T generator-raw logits + n x T storage-raw deltas
        output_dim = (n_nodes + 1) * t_steps + n_nodes * t_steps

        layers = []
        dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1)]
            dim = hidden
        layers += [nn.Linear(dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, injection, marginal_cost, gen_capacity, storage_energy_capacity,
                storage_power_capacity, voll, line_capacity):
        """line_capacity : (B, E) - see dispatch_forward. E must match this instance's n_nodes
        (E = n_nodes*(n_nodes-1)//2), i.e. n_nodes is fixed for a given trained model."""
        B = injection.shape[0]
        connected_capacity = torch.einsum('ne,be->bn', self.incidence, line_capacity)  # (B, N)

        # build input tensor
        x = torch.cat([
            injection.reshape(B, -1),
            marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity,
            connected_capacity,
            voll.reshape(B, 1),
        ], dim=1)
        out = self.net(x)

        n, t = self.n, self.t
        gen_raw = out[:, : (n + 1) * t].reshape(B, n + 1, t)
        storage_raw = out[:, (n + 1) * t:].reshape(B, n, t)
        return gen_raw, storage_raw


# =============================================================================
# Repair layers
# =============================================================================

def storage_repair(storage_delta_raw, storage_energy_capacity, storage_power_capacity, sigmoid_sharpness=4.0):
    """
    Repairs the network's raw storage output into a guaranteed-feasible SOC trajectory over the
    day (midnight to midnight), and the deltas that produce it. The battery is assumed to start
    the horizon at SOC = 0; there is no constraint forcing it back to 0 by the end -- it ends up
    wherever the repaired trajectory puts it (in practice, since using storage is free in this
    model, the network should tend to want to discharge whenever it can).

    Everything here is in FRACTIONAL terms (fraction of storage_power_capacity /
    storage_energy_capacity respectively) -- converting to absolute MW/MWh units happens outside,
    in dispatch_forward, at the same point generation's fraction -> absolute scaling happens.

    storage_delta_raw       : (B, N, T) unconstrained raw output from the network
    storage_energy_capacity : (B, N) energy capacity
    storage_power_capacity  : (B, N) power/rate capacity
    sigmoid_sharpness       : steepness of the sigmoid used to bound SOC into (0, 1); default 4.0
                              gives max slope 1 at SOC = 0.5, matching the max magnitude of a
                              single hour's power-fraction delta (see module docstring).

    Sign convention (see module docstring): production/injection is POSITIVE, consumption/
    extraction is NEGATIVE. Discharging INJECTS power into the grid (production-like), so it is
    POSITIVE. Charging EXTRACTS power from the grid (consumption-like), so it is NEGATIVE.
    storage_delta_raw / the returned realized_delta both follow this convention.

    Returns:
        soc            : (B, N, T) state of charge, as a FRACTION of storage_energy_capacity, in (0, 1)
        realized_delta : (B, N, T) the delta (fraction of storage_power_capacity, same units as
                         storage_delta_raw once bounded) that actually produces this soc trajectory
    """
    # Step 1: bound the raw output to [-1, 1] -- a fraction of storage_power_capacity.
    delta_frac = torch.tanh(storage_delta_raw)  # (B, N, T)

    # rate converting between "fraction of power capacity" and "fraction of energy capacity"
    # (SOC space) for a one-hour step: at hourly resolution, running at full power for one hour
    # moves power_capacity MWh, i.e. power_capacity/energy_capacity = 1/duration_hours of SOC
    # (duration_hours = storage_energy_capacity / storage_power_capacity)
    soc_rate = (storage_power_capacity / storage_energy_capacity).unsqueeze(-1)  # (B, N, 1)

    # Step 2: SOC starts at 0 (midnight); accumulate the raw (unbounded) SOC-space deltas, then
    # bound into (0, 1) with a sharpened sigmoid. This guarantees feasibility for any input, but
    # note it reshapes the WHOLE trajectory somewhat (not just out-of-range excursions), since a
    # plain sigmoid maps input=0 to output=0.5, not to the actual SOC=0 starting point -- the
    # network can (and must) learn whatever raw deltas compensate for this during training.
    soc_cumsum = torch.cumsum(-delta_frac * soc_rate, dim=-1)  # (B, N, T), negated: SOC moves
                                                                # opposite to delta_frac (discharge drains it)
    soc = torch.sigmoid(sigmoid_sharpness * soc_cumsum)  # (B, N, T)

    # Step 3: back out the realized deltas from the repaired trajectory (soc0 = 0), negated back
    # into delta_frac's power-fraction convention for the same reason as the cumsum above
    soc_prev = torch.cat([torch.zeros_like(soc[:, :, :1]), soc[:, :, :-1]], dim=-1)
    realized_delta = -(soc - soc_prev) / soc_rate

    return soc, realized_delta


def balance_repair(dispatch_raw, dispatch_capacity, target_dispatch):
    """
    Generalized version of the paper's Theorem-1 power balance repair
    layer (Eq. 4-5), applied to an arbitrary number of "generators" with
    lower bound 0, at a single timestep (vectorized over batch and time).

    dispatch_raw      : (B, G, T) proposed dispatch (already within [0, dispatch_capacity])
    dispatch_capacity : (B, G, T) upper bound for each generator at each time
                        (allows the lost-load "generator"'s bound to vary with time)
    target_dispatch   : (B, T) target aggregate dispatch (net demand + storage charge)

    Returns:
        dispatch_repaired : (B, G, T) with sum_g dispatch_repaired = target_dispatch exactly,
                     WHENEVER target_dispatch is feasible i.e. 0 <= target_dispatch <=
                     sum_g dispatch_capacity. If target_dispatch is outside that range,
                     dispatch is pinned at the corresponding bound (0 or dispatch_capacity)
                     and a residual imbalance remains -- this should not occur in this v1
                     because the lost-load generator's bound is set to always make
                     target_dispatch feasible; kept here as a safety clamp regardless.
    """
    dispatch_sum = dispatch_raw.sum(dim=1)          # (B, T)
    capacity_sum = dispatch_capacity.sum(dim=1)      # (B, T)

    eps = 1e-8
    need_up = (dispatch_sum < target_dispatch)
    scale_up = torch.clamp((target_dispatch - dispatch_sum) / (capacity_sum - dispatch_sum + eps), 0.0, 1.0)
    scale_down = torch.clamp((dispatch_sum - target_dispatch) / (dispatch_sum - 0.0 + eps), 0.0, 1.0)

    scale_up = scale_up.unsqueeze(1)      # (B,1,T) for broadcasting over G
    scale_down = scale_down.unsqueeze(1)
    need_up = need_up.unsqueeze(1)

    dispatch_up = (1 - scale_up) * dispatch_raw + scale_up * dispatch_capacity
    dispatch_down = (1 - scale_down) * dispatch_raw + scale_down * 0.0

    dispatch_repaired = torch.where(need_up, dispatch_up, dispatch_down)
    return dispatch_repaired


# =============================================================================
# Full forward pass
# =============================================================================

def dispatch_forward(model, injection, marginal_cost, gen_capacity, storage_energy_capacity,
                      storage_power_capacity, voll, ptdf, line_capacity):
    """
    injection, marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity :
        as described in the module docstring
    voll           : (B,)
    ptdf           : (B, E, N) tensor - one topology per batch item (see
                     experiments/ml/daily_dispatch.py's sample_network_batch, or a seer-schema
                     network.csv converted the same way)
    line_capacity  : (B, E) tensor - one topology per batch item

    Returns dict with gen_output, lost_load, soc, line_flows, injections.
    """
    B, N, T = injection.shape

    # line_capacity also feeds the model itself now (a per-node connected-capacity summary, see
    # DispatchNet's docstring) - not just the post-hoc PTDF flow reconstruction/penalty below.
    gen_raw, storage_raw = model(
        injection, marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity,
        voll, line_capacity,
    )

    # ---- clamp generator outputs to capacity (paper's sigmoid-scale trick)
    # Real generators: static bound gen_capacity. Lost-load "generator": dynamic
    # bound = system-wide consumption at each t (can't shed more than exists).
    z = torch.sigmoid(gen_raw)                      # (B, N+1, T)
    z_gen, z_ll = z[:, :N, :], z[:, N:, :]           # split

    dispatch_capacity_gen = gen_capacity.unsqueeze(-1).expand(B, N, T)   # (B, N, T)
    gen_output_raw = z_gen * dispatch_capacity_gen

    # system-wide consumption need = -injection (only the deficit, i.e. injection < 0, is
    # sheddable -- an injection surplus has nothing to shed)
    consumption_total_t = (-injection.sum(dim=1)).clamp(min=0.0)  # (B, T)
    dispatch_capacity_ll = consumption_total_t.unsqueeze(1)       # (B, 1, T)
    lost_load_raw = z_ll * dispatch_capacity_ll                   # (B, 1, T)

    dispatch_capacity_all = torch.cat([dispatch_capacity_gen, dispatch_capacity_ll], dim=1)  # (B, N+1, T)
    dispatch_raw_all = torch.cat([gen_output_raw, lost_load_raw], dim=1)  # (B, N+1, T)

    # ---- storage repair (tanh -> sigmoid-bounded SOC trajectory, guaranteed feasible); returned
    #      in fractional terms, scaled to absolute MW/MWh here, same as generation's z_gen above
    soc_frac, realized_delta_frac = storage_repair(storage_raw, storage_energy_capacity, storage_power_capacity)
    soc = soc_frac * storage_energy_capacity.unsqueeze(-1)
    realized_delta = realized_delta_frac * storage_power_capacity.unsqueeze(-1)

    # ---- aggregate balance target for balance_repair (a consumption-positive "deficit to be
    #      met by generators + lost load" magnitude -- the opposite convention from injection/
    #      realized_delta above; NOTE: this sign flip is a placeholder pending further discussion,
    #      see conversation)
    target_dispatch_t = -(injection.sum(dim=1) + realized_delta.sum(dim=1))   # (B, T)

    # ---- aggregate power balance repair (generators + lost load only;
    #      storage already fixed, entering only via target_dispatch_t)
    dispatch_repaired_all = balance_repair(dispatch_raw_all, dispatch_capacity_all, target_dispatch_t)  # (B, N+1, T)
    gen_output = dispatch_repaired_all[:, :N, :]
    lost_load = dispatch_repaired_all[:, N, :]       # (B, T) aggregate

    # ---- allocate aggregate lost load back to nodes pro-rata by consumption share, purely for
    #      PTDF injection purposes (see module docstring)
    consumption_pos = (-injection).clamp(min=0.0)
    consumption_pos_sum = consumption_pos.sum(dim=1).clamp(min=1e-6)      # (B, T)
    lost_load_node = lost_load.unsqueeze(1) * (consumption_pos / consumption_pos_sum.unsqueeze(1))  # (B, N, T)

    # ---- nodal net injections, for the PTDF -- a plain sum, everything here is already
    #      production-positive (see module docstring)
    injections = gen_output + lost_load_node + injection + realized_delta   # (B, N, T)

    # ---- PTDF flows (KCL holds exactly since sum_i injections = 0 by
    #      construction of the balance repair layer)
    # ptdf: (B, E, N), injections: (B, N, T) -> line_flows: (B, E, T)
    line_flows = torch.einsum('ben,bnt->bet', ptdf, injections)

    return {
        "gen_output": gen_output,
        "lost_load": lost_load,
        "soc": soc,
        "realized_delta": realized_delta,
        "injections": injections,
        "line_flows": line_flows,
    }


# =============================================================================
# Loss function
# =============================================================================

def dispatch_loss(outputs, marginal_cost, voll, line_capacity, m_th=1500.0):
    """
    cost = sum_t sum_i marginal_cost_i * gen_output(i,t)  +  sum_t voll * lost_load(t)
    line_penalty = m_th * sum_e sum_t relu(|line_flows(e,t)| - line_capacity(e))

    line_capacity : (B, E) - one topology per batch item (see dispatch_forward)
    """
    gen_output = outputs["gen_output"]   # (B, N, T)
    lost_load = outputs["lost_load"]     # (B, T)
    line_flows = outputs["line_flows"]   # (B, E, T)

    gen_cost = (marginal_cost.unsqueeze(-1) * gen_output).sum(dim=(1, 2))   # (B,)
    ll_cost = (voll.unsqueeze(-1) * lost_load).sum(dim=1)                   # (B,)

    overflow = torch.relu(line_flows.abs() - line_capacity.unsqueeze(-1))  # (B, E, 1) broadcast over T
    line_penalty = m_th * overflow.sum(dim=(1, 2))                 # (B,)

    total = gen_cost + ll_cost + line_penalty
    return total.mean(), {
        "gen_cost": gen_cost.mean().item(),
        "ll_cost": ll_cost.mean().item(),
        "line_penalty": line_penalty.mean().item(),
    }
