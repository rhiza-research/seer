"""
End-to-end feasible dispatch proxy (E2ELR-style) for a small multi-zone
24h economic dispatch problem with storage and lost load.

Fixed problem size for this prototype: N = 3 zones, T = 24 timesteps. The network
topology (which lines exist, their capacities) now varies per training instance
by default (see train(vary_network=...)) - N itself is still fixed.

-----------------------------------------------------------------------------
DESIGN SUMMARY
-----------------------------------------------------------------------------
Inputs (per instance, batched):
    d        : (B, N, T)  net demand at each node (load - RE), can be < 0
    c        : (B, N)     marginal cost of each dispatchable generator
    gbar     : (B, N)     capacity of each dispatchable generator
    sbar     : (B, N)     energy capacity of each storage unit
    rbar     : (B, N)     power (rate) capacity of each storage unit
    voll     : (B,)       value of lost load (cost of the aggregate lost
                           load "generator"; could also be per-node/time,
                           kept scalar-per-instance here for simplicity)
    ptdf     : (B, E, N)  PTDF matrix, one per instance (see sample_network_batch) -
                           topology now varies per instance; N is still fixed,
                           E = N*(N-1)/2 fixed "slots" (inactive lines get
                           capacity/susceptance 0, see note below)
    line_cap : (B, E)     thermal line capacities, one per instance

Outputs:
    p_real   : (B, N, T)  dispatchable generator output
    lost_load: (B, T)     aggregate lost-load "generator" output (system-wide)
    soc      : (B, N, T)  storage state of charge
    flows    : (B, E, T)  line flows

Repair sequence:
    1. Storage repair (sequential over time): clamp charging rate, then
       clamp SOC to [0, sbar], recompute realized deltas.
    2. Aggregate power balance repair (paper's Theorem-1-style layer,
       generalized to N+1 "generators": N dispatchable + 1 aggregate
       lost-load unit). This is done PER TIMESTEP, treating storage's
       already-fixed realized net charge as part of the effective demand.
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
    - PTDF and line capacities now vary per instance (train(vary_network=True),
      the default) - a fresh random topology is sampled per batch item every
      step (sample_network_batch), fixed node count N but a random set of
      active lines/capacities (guaranteed connected, same construction as
      seer.inputs.synthetic.network_params.build_network). Pass
      vary_network=False to fall back to the original single fixed topology.
    - IMPORTANT: even with vary_network=True, the network topology is only
      used in the PTDF flow reconstruction and the line-capacity penalty -
      DispatchNet itself still doesn't take topology as an input, so it has
      no way to know which lines exist/are constrained for a given instance.
      Confirmed empirically (150 epochs, otherwise identical settings): fixed
      topology converges to line_penalty flat at 0.00 (with flows at ~50-80%
      of the tightest line's capacity - a real but moderate margin, not a
      slack constraint); with vary_network=True, line_penalty never settles,
      bouncing roughly 0-1000 epoch to epoch - the model can't condition its
      dispatch on a topology it never sees, so it can't reliably respect
      constraints that move under it. Feeding topology into the model itself
      (e.g. a GNN/PTDF-conditioned encoder) is the next step, not done here.
      Separately, note total loss itself is noisy/non-monotonic past ~epoch
      10 in BOTH regimes (bouncing ~650k-870k over the remaining 140 epochs) -
      that looks like a pre-existing training/tuning issue independent of
      this change (more epochs, a lr schedule, loss re-weighting), not
      something introduced by varying the topology.
    - Initial SOC is fixed at a config-level fraction of sbar (default
      50%) for every instance; not learned or input-conditioned in v1.
-----------------------------------------------------------------------------
"""

import itertools

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PTDF construction utility
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


# =============================================================================
# 2. Repair layers
# =============================================================================

def storage_repair(delta_raw, sbar, rbar, soc0_frac=0.5):
    """
    Sequential storage repair. Enforces both rate and energy bounds exactly.

    delta_raw : (B, N, T) unconstrained deltas from the network
    sbar      : (B, N) energy capacity
    rbar      : (B, N) power/rate capacity
    Returns:
        soc            : (B, N, T) state of charge, in [0, sbar]
        realized_delta : (B, N, T) actual SOC(t) - SOC(t-1), i.e. actual
                         charge(+)/discharge(-) power in each interval
    """
    B, N, T = delta_raw.shape

    # Step A: clamp raw deltas to the power/rate limit (elementwise, cheap)
    delta_clamped = torch.clamp(delta_raw, min=-rbar.unsqueeze(-1), max=rbar.unsqueeze(-1))

    soc0 = soc0_frac * sbar  # (B, N)
    soc_list = []
    prev_soc = soc0
    for t in range(T):
        cand = prev_soc + delta_clamped[:, :, t]
        # Step B: clamp SOC to [0, sbar]. Because delta was already
        # rate-clamped, this clamp can only shrink |realized delta|,
        # never grow it beyond rbar.
        soc_t = torch.clamp(cand, min=torch.zeros_like(sbar), max=sbar)
        soc_list.append(soc_t)
        prev_soc = soc_t

    soc = torch.stack(soc_list, dim=-1)  # (B, N, T)
    soc_prev = torch.cat([soc0.unsqueeze(-1), soc[:, :, :-1]], dim=-1)
    realized_delta = soc - soc_prev
    return soc, realized_delta


def balance_repair(p_raw, pbar, D):
    """
    Generalized version of the paper's Theorem-1 power balance repair
    layer (Eq. 4-5), applied to an arbitrary number of "generators" with
    lower bound 0, at a single timestep (vectorized over batch and time).

    p_raw : (B, G, T) proposed dispatch (already within [0, pbar])
    pbar  : (B, G, T) upper bound for each generator at each time
            (allows the lost-load "generator"'s bound to vary with time)
    D     : (B, T) target aggregate dispatch (net demand + storage charge)

    Returns:
        p_repaired : (B, G, T) with sum_g p_repaired = D exactly,
                     WHENEVER D is feasible i.e. 0 <= D <= sum_g pbar.
                     If D is outside that range, dispatch is pinned at
                     the corresponding bound (0 or pbar) and a residual
                     imbalance remains -- this should not occur in this
                     v1 because the lost-load generator's bound is set
                     to always make D feasible; kept here as a safety
                     clamp regardless.
    """
    sum_p = p_raw.sum(dim=1)          # (B, T)
    sum_pbar = pbar.sum(dim=1)        # (B, T)

    eps = 1e-8
    need_up = (sum_p < D)
    eta_up = torch.clamp((D - sum_p) / (sum_pbar - sum_p + eps), 0.0, 1.0)
    eta_down = torch.clamp((sum_p - D) / (sum_p - 0.0 + eps), 0.0, 1.0)

    eta_up = eta_up.unsqueeze(1)      # (B,1,T) for broadcasting over G
    eta_down = eta_down.unsqueeze(1)
    need_up = need_up.unsqueeze(1)

    p_up = (1 - eta_up) * p_raw + eta_up * pbar
    p_down = (1 - eta_down) * p_raw + eta_down * 0.0

    p_repaired = torch.where(need_up, p_up, p_down)
    return p_repaired


# =============================================================================
# 3. Network
# =============================================================================

class DispatchNet(nn.Module):
    def __init__(self, n_nodes, t_steps, hidden=256, n_layers=4):
        super().__init__()
        self.n = n_nodes
        self.t = t_steps

        input_dim = (
            n_nodes * t_steps      # d
            + n_nodes               # c
            + n_nodes               # gbar
            + n_nodes               # sbar
            + n_nodes               # rbar
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

    def forward(self, d, c, gbar, sbar, rbar, voll):
        B = d.shape[0]
        x = torch.cat([
            d.reshape(B, -1),
            c, gbar, sbar, rbar,
            voll.reshape(B, 1),
        ], dim=1)
        out = self.net(x)

        n, t = self.n, self.t
        gen_raw = out[:, : (n + 1) * t].reshape(B, n + 1, t)
        storage_raw = out[:, (n + 1) * t:].reshape(B, n, t)
        return gen_raw, storage_raw


# =============================================================================
# 4. Full forward pass with repair layers
# =============================================================================

def dispatch_forward(model, d, c, gbar, sbar, rbar, voll, ptdf, line_cap):
    """
    d, c, gbar, sbar, rbar : as described above
    voll     : (B,)
    ptdf     : (B, E, N) tensor - one topology per batch item (see sample_network_batch)
    line_cap : (B, E) tensor - one topology per batch item

    Returns dict with p_real, lost_load, soc, flows, injections.
    """
    B, N, T = d.shape

    gen_raw, storage_raw = model(d, c, gbar, sbar, rbar, voll)

    # ---- clamp generator outputs to capacity (paper's sigmoid-scale trick)
    # Real generators: static bound gbar. Lost-load "generator": dynamic
    # bound = system-wide net demand at each t (can't shed more than exists).
    z = torch.sigmoid(gen_raw)                      # (B, N+1, T)
    z_real, z_ll = z[:, :N, :], z[:, N:, :]          # split

    pbar_real = gbar.unsqueeze(-1).expand(B, N, T)   # (B, N, T)
    p_real_raw = z_real * pbar_real

    demand_total_t = d.sum(dim=1).clamp(min=0.0)     # (B, T) system net demand (only pos part sheddable)
    pbar_ll = demand_total_t.unsqueeze(1)            # (B, 1, T)
    lost_load_raw = z_ll * pbar_ll                   # (B, 1, T)

    pbar_all = torch.cat([pbar_real, pbar_ll], dim=1)     # (B, N+1, T)
    p_raw_all = torch.cat([p_real_raw, lost_load_raw], dim=1)  # (B, N+1, T)

    # ---- storage repair (sequential, exact rate + energy bounds)
    soc, realized_delta = storage_repair(storage_raw, sbar, rbar)

    # ---- aggregate balance target: net demand + net storage charging
    D_t = d.sum(dim=1) + realized_delta.sum(dim=1)   # (B, T)

    # ---- aggregate power balance repair (generators + lost load only;
    #      storage already fixed, entering only via D_t)
    p_all_repaired = balance_repair(p_raw_all, pbar_all, D_t)   # (B, N+1, T)
    p_real = p_all_repaired[:, :N, :]
    lost_load = p_all_repaired[:, N, :]              # (B, T) aggregate

    # ---- allocate aggregate lost load back to nodes pro-rata by demand
    #      share, purely for PTDF injection purposes (see module docstring)
    d_pos = d.clamp(min=0.0)
    d_pos_sum = d_pos.sum(dim=1, keepdim=True).clamp(min=1e-6)   # (B,1,T)... 
    # NOTE: shape fix below
    d_pos_sum = d_pos.sum(dim=1).clamp(min=1e-6)      # (B, T)
    lost_load_node = lost_load.unsqueeze(1) * (d_pos / d_pos_sum.unsqueeze(1))  # (B, N, T)

    # ---- nodal net injections: generation - consumption
    #      + : dispatchable gen, + lost load served as if it were generation
    #      - : net demand, - storage charging (charging is a load; realized
    #          delta > 0 means charging)
    injections = p_real + lost_load_node - d - realized_delta   # (B, N, T)

    # ---- PTDF flows (KCL holds exactly since sum_i injections = 0 by
    #      construction of the balance repair layer)
    # ptdf: (B, E, N), injections: (B, N, T) -> flows: (B, E, T)
    flows = torch.einsum('ben,bnt->bet', ptdf, injections)

    return {
        "p_real": p_real,
        "lost_load": lost_load,
        "soc": soc,
        "realized_delta": realized_delta,
        "injections": injections,
        "flows": flows,
    }


# =============================================================================
# 5. Loss function
# =============================================================================

def dispatch_loss(outputs, c, voll, line_cap, m_th=1500.0):
    """
    cost = sum_t sum_i c_i * p_real(i,t)  +  sum_t voll * lost_load(t)
    line_penalty = m_th * sum_e sum_t relu(|flow(e,t)| - line_cap(e))

    line_cap : (B, E) - one topology per batch item (see sample_network_batch)
    """
    p_real = outputs["p_real"]           # (B, N, T)
    lost_load = outputs["lost_load"]     # (B, T)
    flows = outputs["flows"]             # (B, E, T)

    gen_cost = (c.unsqueeze(-1) * p_real).sum(dim=(1, 2))          # (B,)
    ll_cost = (voll.unsqueeze(-1) * lost_load).sum(dim=1)          # (B,)

    overflow = torch.relu(flows.abs() - line_cap.unsqueeze(-1))   # (B, E, 1) broadcast over T
    line_penalty = m_th * overflow.sum(dim=(1, 2))                 # (B,)

    total = gen_cost + ll_cost + line_penalty
    return total.mean(), {
        "gen_cost": gen_cost.mean().item(),
        "ll_cost": ll_cost.mean().item(),
        "line_penalty": line_penalty.mean().item(),
    }


# =============================================================================
# 6. Synthetic data generator (for a quick spin before you have real data)
# =============================================================================

def sample_batch(batch_size, n_nodes, t_steps, device="cpu"):
    """
    Very simple synthetic instance sampler. Replace with real data loading
    once you have it -- this only exists so the training loop below runs
    end to end.
    """
    B, N, T = batch_size, n_nodes, t_steps

    hours = torch.linspace(0, 2 * np.pi, T, device=device)
    base_shape = 1.0 + 0.5 * torch.sin(hours - np.pi / 2)  # crude daily shape

    d = torch.zeros(B, N, T, device=device)
    for i in range(N):
        scale = 50.0 + 30.0 * torch.rand(B, 1, device=device)
        noise = 0.05 * torch.randn(B, T, device=device)
        d[:, i, :] = scale * (base_shape.unsqueeze(0) + noise)

    c = 20.0 + 40.0 * torch.rand(B, N, device=device)
    gbar = 60.0 + 40.0 * torch.rand(B, N, device=device)
    sbar = 20.0 + 30.0 * torch.rand(B, N, device=device)
    rbar = 10.0 + 15.0 * torch.rand(B, N, device=device)
    voll = 2000.0 + 1000.0 * torch.rand(B, device=device)

    return d, c, gbar, sbar, rbar, voll


# =============================================================================
# 7. Training loop
# =============================================================================

def train(n_epochs=200, batch_size=64, lr=1e-3, device="cpu", vary_network=True):
    N, T = 3, 24

    # --- fixed toy topology: a 3-node triangle with modest susceptances
    # (only used when vary_network=False, to reproduce the original fixed-topology behavior)
    fixed_branches = [
        (0, 1, 5.0, 60.0),   # (from, to, susceptance, capacity)
        (1, 2, 5.0, 60.0),
        (0, 2, 3.0, 40.0),
    ]
    fixed_ptdf_np, fixed_line_cap_np = compute_ptdf(N, fixed_branches, slack_bus=0)

    model = DispatchNet(N, T).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        d, c, gbar, sbar, rbar, voll = sample_batch(batch_size, N, T, device)

        # network is passed in as an input to this step, used only in the PTDF-based flow
        # reconstruction (dispatch_forward) and the line-capacity penalty (dispatch_loss) - the
        # model itself still doesn't see it (see module docstring's "Known v1 simplifications")
        if vary_network:
            # a fresh, independently-sampled topology per batch item, every step
            ptdf, line_cap = sample_network_batch(batch_size, N, device=device)
        else:
            ptdf = torch.tensor(fixed_ptdf_np, dtype=torch.float32, device=device).expand(batch_size, -1, -1)
            line_cap = torch.tensor(fixed_line_cap_np, dtype=torch.float32, device=device).expand(batch_size, -1)

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

    return model, ptdf, line_cap, losses


if __name__ == "__main__":
    model, ptdf, line_cap, losses = train()
    plt.plot(losses)
    plt.show()
    breakpoint()