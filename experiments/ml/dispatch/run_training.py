"""Full training run: 15 SEER instances (4 zones) for training (15*365 dispatch days), 1 held-out
instance for testing (365 dispatch days). Trains DispatchNet, plots the loss curves, then
evaluates on the held-out instance's full year.

Run directly: python -m experiments.ml.dispatch.run_training
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.ml.dispatch.generate_dispatch_data import VOLL, build_dataset, sample_batch
from experiments.ml.dispatch.ml_dispatch_experiment import compute_ptdfs
from seer.models.ml.dispatch import DispatchNet, dispatch_forward, dispatch_loss

N_ZONES = 4
T_STEPS = 24
N_TRAIN_INSTANCES = 15
N_EPOCHS = 3000
BATCH_SIZE = 64
LR = 1e-4
MAX_GRAD_NORM = 1.0   # gradient clipping -- early gradients were exploding (up to ~2.6e7) and
                      # permanently saturating the network's clamps/sigmoids, killing all
                      # further learning (see conversation)


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def evaluate_full_year(model, instance, ptdf, device="cpu"):
    """Run the model (no grad) over every one of the instance's 365 non-overlapping days at
    once (as a size-365 batch), for a clean, deterministic held-out evaluation."""
    n_days = instance.n_days
    injection = np.stack([
        instance.injection_by_hour[:, d * T_STEPS:(d + 1) * T_STEPS] for d in range(n_days)
    ])
    marginal_cost = np.tile(instance.marginal_cost, (n_days, 1))
    gen_capacity = np.tile(instance.gen_capacity, (n_days, 1))
    storage_energy_capacity = np.tile(instance.storage_energy_capacity, (n_days, 1))
    storage_power_capacity = np.tile(instance.storage_power_capacity, (n_days, 1))
    capacity_matrix = np.tile(instance.capacity_matrix, (n_days, 1, 1))
    ptdf_batch = np.tile(ptdf, (n_days, 1, 1))
    voll = np.full(n_days, VOLL)

    with torch.no_grad():
        outputs = dispatch_forward(
            model, to_tensor(injection, device), to_tensor(marginal_cost, device),
            to_tensor(gen_capacity, device), to_tensor(storage_energy_capacity, device),
            to_tensor(storage_power_capacity, device), to_tensor(voll, device),
            to_tensor(ptdf_batch, device), to_tensor(capacity_matrix, device),
        )
        loss, parts = dispatch_loss(outputs, to_tensor(marginal_cost, device),
                                     to_tensor(voll, device), to_tensor(capacity_matrix, device))

    # $/MWh actually realized, for comparison against the instance's own marginal costs
    total_cost = (to_tensor(marginal_cost, device).unsqueeze(-1) * outputs["gen_output"]).sum().item()
    total_mwh = outputs["gen_output"].sum().item()
    parts["gen_cost_per_mwh"] = total_cost / max(total_mwh, 1e-6)
    return outputs, loss.item(), parts


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"generating {N_TRAIN_INSTANCES} training instances ({N_ZONES} zones each)...")
    train_dataset = build_dataset(N_TRAIN_INSTANCES, n_zones=N_ZONES, seed=0)
    train_ptdfs = compute_ptdfs(train_dataset)

    print("generating 1 held-out test instance...")
    test_dataset = build_dataset(1, n_zones=N_ZONES, seed=12345)   # different seed -- independent draw
    test_instance = test_dataset[0]
    test_ptdf = compute_ptdfs(test_dataset)[0]

    # marginal costs across all training instances, for the $/MWh reference lines later
    train_marginal_costs = np.concatenate([inst.marginal_cost for inst in train_dataset])
    print(f"training generator marginal costs: min={train_marginal_costs.min():.2f} "
          f"mean={train_marginal_costs.mean():.2f} max={train_marginal_costs.max():.2f}")

    model = DispatchNet(N_ZONES, T_STEPS)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"loss": [], "gen_cost": [], "gen_cost_per_mwh": [], "ll_cost": [], "line_penalty": []}

    print(f"training for {N_EPOCHS} epochs...")
    for epoch in range(N_EPOCHS):
        injection, marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity, \
            voll, capacity_matrix, instance_indices = sample_batch(train_dataset, BATCH_SIZE, t_steps=T_STEPS)
        ptdf = to_tensor(train_ptdfs[instance_indices], "cpu")

        outputs = dispatch_forward(model, injection, marginal_cost, gen_capacity,
                                    storage_energy_capacity, storage_power_capacity, voll, ptdf, capacity_matrix)
        loss, parts = dispatch_loss(outputs, marginal_cost, voll, capacity_matrix)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optim.step()

        total_cost = (marginal_cost.unsqueeze(-1) * outputs["gen_output"]).sum().item()
        total_mwh = outputs["gen_output"].sum().item()

        history["loss"].append(loss.item())
        history["gen_cost"].append(parts["gen_cost"])
        history["gen_cost_per_mwh"].append(total_cost / max(total_mwh, 1e-6))
        history["ll_cost"].append(parts["ll_cost"])
        history["line_penalty"].append(parts["line_penalty"])

        if epoch % 200 == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:4d} | loss {loss.item():10.2f} | gen_cost {parts['gen_cost']:9.2f} | "
                  f"ll_cost {parts['ll_cost']:9.2f} | line_penalty {parts['line_penalty']:9.2f}")

    print("\nevaluating on held-out test instance (full 365-day year)...")
    test_outputs, test_loss, test_parts = evaluate_full_year(model, test_instance, test_ptdf)
    print(f"test loss (mean over 365 days): {test_loss:.2f}")
    print(f"test gen_cost: {test_parts['gen_cost']:.2f} | gen_cost/MWh: {test_parts['gen_cost_per_mwh']:.2f} | "
          f"ll_cost: {test_parts['ll_cost']:.2f} | line_penalty: {test_parts['line_penalty']:.2f}")
    print(f"test instance marginal costs: {test_instance.marginal_cost}")

    # ---- plot ----
    def smooth(x, window=30):
        x = np.array(x)
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(history["loss"], alpha=0.3, color="C0")
    axes[0, 0].plot(smooth(history["loss"]), color="C0")
    axes[0, 0].set_title("total loss")
    axes[0, 0].set_xlabel("epoch")

    axes[0, 1].plot(history["gen_cost"], alpha=0.3, color="C1")
    axes[0, 1].plot(smooth(history["gen_cost"]), color="C1")
    axes[0, 1].set_title("gen_cost ($)")
    axes[0, 1].set_xlabel("epoch")

    axes[0, 2].plot(history["gen_cost_per_mwh"], alpha=0.3, color="C2")
    axes[0, 2].plot(smooth(history["gen_cost_per_mwh"]), color="C2")
    axes[0, 2].axhline(train_marginal_costs.min(), color="gray", linestyle="--", linewidth=1, label="min mc")
    axes[0, 2].axhline(train_marginal_costs.mean(), color="black", linestyle="--", linewidth=1, label="mean mc")
    axes[0, 2].axhline(train_marginal_costs.max(), color="gray", linestyle="--", linewidth=1, label="max mc")
    axes[0, 2].set_title("gen_cost ($/MWh) vs. training marginal costs")
    axes[0, 2].set_xlabel("epoch")
    axes[0, 2].legend(fontsize=7)

    axes[1, 0].plot(history["ll_cost"], alpha=0.3, color="C3")
    axes[1, 0].plot(smooth(history["ll_cost"]), color="C3")
    axes[1, 0].set_title("ll_cost (VOLL-weighted shedding)")
    axes[1, 0].set_xlabel("epoch")

    axes[1, 1].plot(history["line_penalty"], alpha=0.3, color="C4")
    axes[1, 1].plot(smooth(history["line_penalty"]), color="C4")
    axes[1, 1].set_title("line_penalty")
    axes[1, 1].set_xlabel("epoch")

    # train (last 200 epochs avg) vs held-out test instance
    last_n = 200
    train_avg = {
        "gen_cost/MWh": np.mean(history["gen_cost_per_mwh"][-last_n:]),
        "ll_cost": np.mean(history["ll_cost"][-last_n:]),
        "line_penalty": np.mean(history["line_penalty"][-last_n:]),
    }
    test_avg = {
        "gen_cost/MWh": test_parts["gen_cost_per_mwh"],
        "ll_cost": test_parts["ll_cost"],
        "line_penalty": test_parts["line_penalty"],
    }
    labels = list(train_avg.keys())
    x = np.arange(len(labels))
    axes[1, 2].bar(x - 0.2, [train_avg[k] for k in labels], width=0.4, label=f"train (last {last_n} epochs)")
    axes[1, 2].bar(x + 0.2, [test_avg[k] for k in labels], width=0.4, label="held-out test instance")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels, fontsize=8)
    axes[1, 2].set_yscale("symlog")
    axes[1, 2].set_title("train vs. held-out test")
    axes[1, 2].legend(fontsize=7)

    fig.tight_layout()
    out_path = "/tmp/dispatch_training_run.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nplot saved to {out_path}")
    return model, train_dataset, test_instance, history, out_path


if __name__ == "__main__":
    main()
