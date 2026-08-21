# Dispatch model -- ideas to try

Running list of design questions/alternatives raised during review, not yet implemented or
tested. Revisit once we can actually train and compare runs.

## 1. Storage repair: "debt carryover" from the global-clamp approach

**Current design** (`storage_repair` in `seer/models/ml/dispatch/dispatch_net.py`): tanh-bound
raw output, convert to SOC-space deltas, `cumsum` over the full day, then a single hard clamp
to `[0, 1]` at the end. `storage_output` is the first difference of the clamped trajectory.

**Issue found**: because the clamp is applied to the *cumulative* raw path rather than
per-step, an overshoot at one hour doesn't just get capped there -- it leaves a "debt" in the
underlying (unclamped) cumsum that suppresses the realized output of *later* hours too, even
when those later hours command an action in the opposite direction. Concretely: if the network
commands three hours of strong charging (way more than needed to fill the battery), and then
commands a strong discharge, the discharge can come out as ~0 realized output, because the raw
cumulative sum is still far above the ceiling and the discharge command isn't enough to pay that
down. See conversation for two worked examples (Case 1 and Case 2 in the smoke test).

**Alternative**: a sequential/recursive clamp -- `soc[t] = clamp(soc[t-1] + delta_raw[t], 0, 1)`,
using the *actual clamped* `soc[t-1]` as the base for the next step, not a running raw sum. This
is (a refined version of) what the very first pre-rework implementation did (a Python loop over
T). It should remove the debt-carryover effect entirely, since `soc[t-1]` is always already in
`[0, 1]` by induction -- there's no unclamped "overshoot" to accumulate. The same rate-bound
guarantee we verified for the global clamp (`|storage_output| <= power_capacity` exactly, every
hour including the first) should still hold, by the same 1-Lipschitz argument, applied at every
step instead of once.

**Tradeoffs to weigh before switching**:
- Requires a genuinely sequential computation (a loop or scan over T), not a single vectorized
  cumsum + clamp. Likely a small cost for T=24, but worth confirming.
- A chain of 24 sequential clamps could have its own gradient-flow considerations -- a
  saturation event at some hour could block gradient signal between hours on either side of it
  (each clamp's local gradient is exactly 0 when saturated, exactly 1 otherwise, and this
  compounds multiplicatively across the chain). Not obviously worse than the global-clamp
  version's dead-zone issue, but a different failure shape -- worth watching for empirically.

**To try**: implement the sequential version, verify the debt-carryover cases from the
conversation now behave as expected (discharge commands after an overshoot period actually
produce nonzero output), and compare training dynamics (does convergence speed/stability
differ meaningfully from the current global-clamp version?).

## 2. Passing richer line/topology information into the model

**Current design**: `DispatchNet` only sees a coarse per-node summary of the network -- each
node's total connected line capacity (`line_capacity.sum(dim=-1)`), not the full topology. PTDF
itself (the actual electrical sensitivities) is never seen by the network at all -- it's only
used post-hoc, after dispatch decisions are made, to compute `line_flows` for the loss penalty
(see module docstring's "Known v1 simplifications" and conversation). A plain MLP has no
structural way to exploit the full NxN/PTDF relational structure even if given it directly (no
weight sharing / permutation-equivariance the way a GNN has -- see conversation for the full
argument), so this isn't just "feed it more data," it likely needs a different architecture.

**To try**: a GNN (or PTDF-conditioned encoder) that takes the full graph structure (node
features + per-line capacities/PTDF) and can, in principle, generalize across varying N as well
-- unlike the current fixed-N MLP. Open questions to resolve when actually trying this:
- What's the right node/edge feature set -- raw line capacities, PTDF rows, both?
- Message-passing depth vs. training cost/data needs for the scale we care about.
- Whether to replace DispatchNet's whole input encoding, or just augment the current per-node
  summary with a learned GNN-produced embedding.

**Baseline to compare against**: the existing topology-blind vs. topology-summary comparison
noted in the module docstring (fixed topology converged to `line_penalty` flat at 0.00; varying
topology never settled, bouncing ~0-1000 epoch to epoch) -- re-run that comparison with the
current NxN/per-node-summary version in place before drawing conclusions about whether a GNN is
actually needed.
