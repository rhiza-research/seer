"""Daily dispatch model.

-----------------------------------------------------------------------------
Architecture Summary
-----------------------------------------------------------------------------
B = batch size
N = number of nodes
T = number of time steps
E = number of lines
Each node is assumed to have a single dispatchable generator, storage unit, and aggregate load.

Sign convention: production is POSITIVE, consumption is NEGATIVE.

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
       sum as closely as possible to a target aggregate dispatch that already accounts for
       storage_output -- pinning to 0 or combined capacity if the target is infeasible, no
       separate pre-clamp needed (see balance_repair). Lost load is whatever's left over -- a
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
            # voll deliberately NOT included -- it's currently a constant (1e6) across every
            # training example, so it carries no information, and its raw scale was dwarfing
            # every other feature and destabilizing training (see conversation).
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

    # quick-and-dirty input-only rescale for power/energy-type features (hundreds of MW/MWh),
    # purely to keep the MLP's own input conditioned to a smaller scale -- this ONLY affects
    # what's fed into the network here; gen_capacity/storage capacities/etc. are used at their
    # true scale everywhere else (dispatch_forward, storage_repair, balance_repair, dispatch_loss
    # all see the real, unscaled values), so nothing downstream needs to change (see conversation)
    INPUT_SCALE = 1000.0

    def forward(self, injection, marginal_cost, gen_capacity, storage_energy_capacity,
                storage_power_capacity, voll, line_capacity):
        # voll is accepted for call-site/signature compatibility with dispatch_forward, but
        # deliberately not used as a model input (see input_dim's comment above)
        B = injection.shape[0]
        # connectivity information currently passed as total connected line capacity
        connected_capacity = line_capacity.sum(dim=-1)

        # input tensor
        s = self.INPUT_SCALE
        x = torch.cat([
            (injection / s).reshape(B, -1),
            marginal_cost, gen_capacity / s, storage_energy_capacity / s, storage_power_capacity / s,
            connected_capacity / s,
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
    Theorem 1 balance repair layer (Eq. 6) from Chen, Tanneau & Van Hentenryck, "End-to-End
    Feasible Optimization Proxies for Large-Scale Economic Dispatch" (arXiv:2304.11726),
    generalized to an arbitrary number of resources with lower bound 0. Proportionally scales
    dispatch_raw toward 0 or dispatch_capacity (whichever direction closes the gap) so the G
    resources sum EXACTLY to target_dispatch.

    dispatch_raw      : (B, G, T) proposed dispatch, already within [0, dispatch_capacity]
    dispatch_capacity : (B, G, T) upper bound per resource, per time
    target_dispatch   : (B, T) the exact aggregate value dispatch_repaired must sum to.

    Returns:
        dispatch_repaired : (B, G, T), sum_g dispatch_repaired == target_dispatch exactly, as
                     long as target_dispatch is feasible (in [0, sum_g dispatch_capacity]).
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


def dispatch_forward(model, injection, marginal_cost, gen_capacity, storage_energy_capacity,
                      storage_power_capacity, voll, ptdf, line_capacity):
    """
    Returns dict with gen_output, lost_load, soc, line_flows, injections.
    """
    B, N, T = injection.shape

    # get raw storage and generator dispatch
    gen_raw, storage_raw = model(
        injection, marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity,
        voll, line_capacity,
    )

    gen_capacity_t = gen_capacity.unsqueeze(-1).expand(B, N, T)
    # repair individual generator dispatch by clamping to capacity via sigmoid scaling
    gen_fraction = torch.sigmoid(gen_raw)
    gen_output_raw = gen_fraction * gen_capacity_t

    # repair storage dispatch
    soc_frac, storage_output_frac = storage_repair(storage_raw, storage_energy_capacity, storage_power_capacity)
    soc = soc_frac * storage_energy_capacity.unsqueeze(-1)
    storage_output = storage_output_frac * storage_power_capacity.unsqueeze(-1)

    # dispatch target
    target_dispatch_t = -(injection.sum(dim=1) + storage_output.sum(dim=1))   # (B, T)

    # repair generator dispatch by balancing against target dispatch as far as possible
    gen_output = balance_repair(gen_output_raw, gen_capacity_t, target_dispatch_t)  # (B, N, T)

    # any remaining residual must be lost load
    lost_load = target_dispatch_t - gen_output.sum(dim=1)                            # (B, T), can be < 0

    # allocate lost load to nodes so it is accounted for in line flows
    pre_lost_load = gen_output + injection + storage_output                      # (B, N, T)
    eps = 1e-8
    shed_alloc = torch.relu(-pre_lost_load) / torch.relu(-pre_lost_load).sum(dim=1, keepdim=True).clamp(min=eps)
    curtail_alloc = torch.relu(pre_lost_load) / torch.relu(pre_lost_load).sum(dim=1, keepdim=True).clamp(min=eps)
    lost_load_t = lost_load.unsqueeze(1)  # (B, 1, T)
    lost_load_node = torch.where(lost_load_t > 0, lost_load_t * shed_alloc, lost_load_t * curtail_alloc)
    # nodal net injections
    injections = gen_output + lost_load_node + injection + storage_output   # (B, N, T)

    # ---- PTDF flows (KCL holds exactly since sum_i injections = 0 by
    #      construction of the balance repair layer)
    # ptdf: (B, E, N), injections: (B, N, T) -> line_flows: (B, E, T); a per-batch matmul
    line_flows = ptdf @ injections

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

    # generation cost
    gen_cost = (marginal_cost.unsqueeze(-1) * gen_output).sum(dim=(1, 2))   # (B,)
    # only load shedding (lost_load > 0) is costed at VOLL
    # RE curtailment (lost_load < 0) is not penalized for now
    ll_cost = (voll.unsqueeze(-1) * torch.relu(lost_load)).sum(dim=1)       # (B,)

    # penalize line flows that exceed capacity
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
