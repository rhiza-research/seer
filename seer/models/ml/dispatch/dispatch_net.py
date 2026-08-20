"""Daily dispatch model.

-----------------------------------------------------------------------------
Architecture Summary
-----------------------------------------------------------------------------
B = batch size
N = number of nodes
T = number of time steps
E = number of lines
Each node is assumed to have a single dispatchable generator, storage unit, and aggregate load.

Sign convention: production/injection is POSITIVE, consumption/extraction is NEGATIVE.

Inputs (per instance, batched):                                                              range
    injection               : (B, N, T)  net injection at each node                         signed (+prod / -consum)
    marginal_cost           : (B, N)     marginal cost, per generator                       >= 0
    gen_capacity            : (B, N)     capacity, per generator                            >= 0
    storage_energy_capacity : (B, N)     energy capacity, per storage                       > 0 (divisor, see storage_repair)
    storage_power_capacity  : (B, N)     power/rate capacity, per storage                   >= 0
    voll                    : (B,)       value of lost load                                 >= 0
    line_capacity           : (B, N, N)  thermal line capacities, symmetric, zero diagonal   >= 0 (0 = no line)

Outputs:                                                            range
    gen_output     : (B, N, T)  dispatchable generator output       [0, gen_capacity]
    lost_load      : (B, T)     aggregate lost load                 signed (+shed / -curtail)
    soc            : (B, N, T)  storage state of charge             [0, storage_energy_capacity]
    storage_output : (B, N, T)  storage power output                signed, |.| <= storage_power_capacity
    injections     : (B, N, T)  nodal net injection (post-repair)   signed
    line_flows     : (B, E, T)  line flows                          signed

Model structure:
    1. DispatchNet (MLP) outputs unconstrained gen & storage dispatch.
    2. Capacity repair: generator dispatch clamped to generator capacity via sigmoid scaling.
    3. Storage repair: tanh + cumsum + clamp turns storage_raw into a guaranteed-feasible SOC
       trajectory and the storage_output that produces it (see storage_repair). Runs before
       generator balance repair because storage_output feeds directly into the target that
       repair balances against.
    4. Generator balance repair: balance_repair forces the N (capacity-clamped) generators to
       sum exactly to a target aggregate dispatch that already accounts for storage_output
       (pre-clamped to be feasible). Lost load is the residual beyond generator capacity -- a
       deterministic fallback, not a learned output (see dispatch_forward).
    5. Nodal injections + PTDF: gen_output, allocated lost load, injection, and storage_output
       combine into nodal net injections (summing to exactly 0, by construction), which the
       PTDF converts into line_flows -- KCL holds automatically, no repair needed there.
    6. Loss: generation cost + shedding cost (VOLL, positive lost_load only) + a soft penalty
       for exceeding line_capacity.
-----------------------------------------------------------------------------
"""

import itertools

import torch
import torch.nn as nn


# =============================================================================
# Model
# =============================================================================

class DispatchNet(nn.Module):
    def __init__(self, n_nodes, t_steps, hidden=256, n_layers=4):
        super().__init__()
        self.n = n_nodes
        self.t = t_steps

        input_dim = (
            n_nodes * t_steps      # injection
            + n_nodes               # marginal_cost
            + n_nodes               # gen_capacity
            + n_nodes               # storage_energy_capacity
            + n_nodes               # storage_power_capacity
            + n_nodes               # connected_capacity
            + 1                      # voll
        )
        # output: n x T gen_raw + n x T storage_raw.
        output_dim = n_nodes * t_steps + n_nodes * t_steps

        layers = []
        dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1)]
            dim = hidden
        layers += [nn.Linear(dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, injection, marginal_cost, gen_capacity, storage_energy_capacity,
                storage_power_capacity, voll, line_capacity):
        B = injection.shape[0]
        # connectivity information currently passed as total connected line capacity
        connected_capacity = line_capacity.sum(dim=-1)

        # input tensor
        x = torch.cat([
            injection.reshape(B, -1),
            marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity,
            connected_capacity,
            voll.reshape(B, 1),
        ], dim=1)
        out = self.net(x)

        n, t = self.n, self.t
        gen_raw = out[:, : n * t].reshape(B, n, t)
        storage_raw = out[:, n * t:].reshape(B, n, t)
        return gen_raw, storage_raw


# =============================================================================
# Repair layers
# =============================================================================

def storage_repair(storage_raw, storage_energy_capacity, storage_power_capacity):
    """
    Returns corrected hourly storage outputs so power and SOC capacity constraints are
    respected exactly; operates on fractional values.

        1. storage_raw is tanh-clamped to a fraction in [-1, 1].
        2. converted to fractional state of charge and
           accumulated over time, assuming soc(t=0) = 0.
        3. resulting soc is clamped to [0, 1].
        4. storage_output is the first difference of the repaired
           soc trajectory, converted to power fraction units.

    storage_raw              : (B, N, T) unconstrained raw output from the network
    storage_energy_capacity  : (B, N) energy capacity
    storage_power_capacity   : (B, N) power/rate capacity

    Returns:
        soc            : (B, N, T) state of charge, fraction of storage_energy_capacity, in [0, 1]
        storage_output : (B, N, T) power, fraction of storage_power_capacity; guaranteed |.| <= 1
    """
    # bound raw storage dispatch to [-1, 1] of storage_power_capacity
    power_frac = torch.tanh(storage_raw)  # (B, N, T)
    # convert fractional power output to fractional state of charge
    soc_rate = (storage_power_capacity / storage_energy_capacity).unsqueeze(-1)  # (B, N, 1)
    soc_cumsum = torch.cumsum(-power_frac * soc_rate, dim=-1)  # (B, N, T)
    # clamp soc between 0 and 1
    soc = torch.clamp(soc_cumsum, 0.0, 1.0)  # (B, N, T)
    # determine corrected storage output
    soc_prev = torch.cat([torch.zeros_like(soc[:, :, :1]), soc[:, :, :-1]], dim=-1)
    storage_output = -(soc - soc_prev) / soc_rate

    return soc, storage_output


def balance_repair(dispatch_raw, dispatch_capacity, target_dispatch):
    """
    Generalized version of the paper's Theorem-1 power balance repair
    layer (Eq. 4-5), applied to an arbitrary number of "generators" with
    lower bound 0, at a single timestep (vectorized over batch and time).

    dispatch_raw      : (B, G, T) proposed dispatch (already within [0, dispatch_capacity])
    dispatch_capacity : (B, G, T) upper bound for each generator at each time
    target_dispatch   : (B, T) target aggregate dispatch

    Returns:
        dispatch_repaired : (B, G, T) with sum_g dispatch_repaired = target_dispatch exactly,
                     WHENEVER target_dispatch is feasible i.e. 0 <= target_dispatch <=
                     sum_g dispatch_capacity. If target_dispatch is outside that range,
                     dispatch is pinned at the corresponding bound (0 or dispatch_capacity)
                     and a residual imbalance remains -- callers should pre-clamp
                     target_dispatch to [0, sum_g dispatch_capacity] if they need this to never
                     trigger (e.g. dispatch_forward's target_dispatch_gen); kept here as a
                     safety clamp regardless.
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
                     experiments/ml/ml_dispatch_experiment.py's compute_ptdf/network adapter)
    line_capacity  : (B, N, N) tensor - one topology per batch item; symmetric, zero diagonal
                     (see capacity_matrix_to_edges for the edge-indexed form PTDF/the loss need)

    Returns dict with gen_output, lost_load, soc, line_flows, injections.
    """
    B, N, T = injection.shape

    # get raw storage and generator dispatch
    gen_raw, storage_raw = model(
        injection, marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity,
        voll, line_capacity,
    )

    # generator dispatch is clamped between 0 and capacity with sigmoid scaling
    z_gen = torch.sigmoid(gen_raw)                                       # (B, N, T)
    gen_capacity_t = gen_capacity.unsqueeze(-1).expand(B, N, T)          # (B, N, T)
    gen_output_raw = z_gen * gen_capacity_t

    # ---- storage repair (tanh -> sigmoid-bounded SOC trajectory, guaranteed feasible); returned
    #      in fractional terms, scaled to absolute MW/MWh here, same as generation's z_gen above
    soc_frac, storage_output_frac = storage_repair(storage_raw, storage_energy_capacity, storage_power_capacity)
    soc = soc_frac * storage_energy_capacity.unsqueeze(-1)
    storage_output = storage_output_frac * storage_power_capacity.unsqueeze(-1)

    # ---- aggregate balance target for balance_repair: a consumption-positive "deficit to be
    #      met by generators" magnitude -- the opposite convention from injection/storage_output
    #      above. This flip is unavoidable (not just a leftover inconsistency): generators are
    #      non-negative magnitudes, not signed quantities, so their target has to be stated as
    #      "how much is needed" rather than "net signed position".
    target_dispatch_t = -(injection.sum(dim=1) + storage_output.sum(dim=1))   # (B, T)

    # ---- real generators absorb the target first, up to their combined capacity
    gen_capacity_sum = gen_capacity_t.sum(dim=1)                                      # (B, T)
    target_dispatch_gen = torch.clamp(target_dispatch_t, min=torch.zeros_like(gen_capacity_sum), max=gen_capacity_sum)

    # ---- power balance repair among real generators only (storage is already fixed, entering
    #      only via target_dispatch_t above)
    gen_output = balance_repair(gen_output_raw, gen_capacity_t, target_dispatch_gen)  # (B, N, T)

    # ---- lost load is whatever's left over -- a deterministic residual, not a learned output
    #      (see DispatchNet's docstring). Computed from gen_output's ACTUAL realized sum, not
    #      target_dispatch_gen, so it exactly absorbs any floating-point slippage from
    #      balance_repair above and injections sums to zero exactly, not just approximately.
    #      Negative lost load means curtailment (target < 0, i.e. more RE + storage discharge
    #      than needed, and generators are already floored at 0) -- not a learned choice either,
    #      just the unavoidable sign of the same residual.
    lost_load = target_dispatch_t - gen_output.sum(dim=1)                            # (B, T), can be < 0

    # ---- allocate aggregate lost load back to nodes, purely for PTDF injection purposes (see
    #      module docstring). Basis: the nodal injection BEFORE lost load is added -- i.e. where
    #      the deficit/surplus that lost_load is covering actually sits, once generators and
    #      storage have already made their decisions. Shed (lost_load > 0) goes to deficit nodes
    #      (pre_lost_load < 0), curtailment (lost_load < 0) goes to surplus nodes (pre_lost_load >
    #      0), each pro-rata by magnitude. Since pre_lost_load.sum(dim=1) == -lost_load exactly
    #      (gen_output.sum(dim=1) == target_dispatch_gen by construction), whichever basis is
    #      active is guaranteed to have a nonzero sum -- e.g. lost_load > 0 forces
    #      pre_lost_load.sum(dim=1) < 0, so not every node can be >= 0.
    pre_lost_load = gen_output + injection + storage_output                      # (B, N, T)
    eps = 1e-8
    shed_alloc = torch.relu(-pre_lost_load) / torch.relu(-pre_lost_load).sum(dim=1, keepdim=True).clamp(min=eps)
    curtail_alloc = torch.relu(pre_lost_load) / torch.relu(pre_lost_load).sum(dim=1, keepdim=True).clamp(min=eps)
    lost_load_t = lost_load.unsqueeze(1)  # (B, 1, T)
    lost_load_node = torch.where(lost_load_t > 0, lost_load_t * shed_alloc, lost_load_t * curtail_alloc)

    # ---- nodal net injections, for the PTDF -- a plain sum, everything here is already
    #      production-positive (see module docstring)
    injections = gen_output + lost_load_node + injection + storage_output   # (B, N, T)

    # ---- PTDF flows (KCL holds exactly since sum_i injections = 0 by
    #      construction of the balance repair layer)
    # ptdf: (B, E, N), injections: (B, N, T) -> line_flows: (B, E, T)
    line_flows = torch.einsum('ben,bnt->bet', ptdf, injections)

    return {
        "gen_output": gen_output,
        "lost_load": lost_load,
        "soc": soc,
        "storage_output": storage_output,
        "injections": injections,
        "line_flows": line_flows,
    }


# =============================================================================
# Loss function
# =============================================================================

def dispatch_loss(outputs, marginal_cost, voll, line_capacity, m_th=1500.0):
    """
    cost = sum_t sum_i marginal_cost_i * gen_output(i,t)  +  sum_t voll * relu(lost_load(t))
    line_penalty = m_th * sum_e sum_t relu(|line_flows(e,t)| - line_capacity(e))

    line_capacity : (B, N, N) - one topology per batch item, symmetric, zero diagonal (see
                    dispatch_forward); converted to edge-indexed form here to compare against
                    line_flows (which is edge-indexed, from the PTDF)
    """
    gen_output = outputs["gen_output"]   # (B, N, T)
    lost_load = outputs["lost_load"]     # (B, T)
    line_flows = outputs["line_flows"]   # (B, E, T)
    N = gen_output.shape[1]
    line_capacity_edges = capacity_matrix_to_edges(line_capacity, N)   # (B, E)

    gen_cost = (marginal_cost.unsqueeze(-1) * gen_output).sum(dim=(1, 2))   # (B,)
    # only the shedding part (lost_load > 0) is costed at VOLL -- negative lost_load is
    # curtailment (see dispatch_forward), which is free for now (no reward, no penalty)
    ll_cost = (voll.unsqueeze(-1) * torch.relu(lost_load)).sum(dim=1)       # (B,)

    overflow = torch.relu(line_flows.abs() - line_capacity_edges.unsqueeze(-1))  # (B, E, 1) broadcast over T
    line_penalty = m_th * overflow.sum(dim=(1, 2))                 # (B,)

    total = gen_cost + ll_cost + line_penalty
    return total.mean(), {
        "gen_cost": gen_cost.mean().item(),
        "ll_cost": ll_cost.mean().item(),
        "line_penalty": line_penalty.mean().item(),
    }


def canonical_edge_pairs(n_nodes):
    """Define a fixed edge ordering for the network"""
    return list(itertools.combinations(range(n_nodes), 2))


def capacity_matrix_to_edges(capacity_matrix, n_nodes):
    """Extract the canonically-ordered (..., E) edge values from an (..., N, N) symmetric
    capacity matrix, using canonical_edge_pairs' ordering."""
    pairs = canonical_edge_pairs(n_nodes)
    idx_i = torch.tensor([i for i, j in pairs], device=capacity_matrix.device)
    idx_j = torch.tensor([j for i, j in pairs], device=capacity_matrix.device)
    return capacity_matrix[..., idx_i, idx_j]
