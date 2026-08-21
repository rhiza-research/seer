"""Trains the dispatch model (seer.models.ml.dispatch) on SEER-generated synthetic data.
See generate_dispatch_data.py for how that data is generated/adapted."""

import torch

from experiments.ml.dispatch.generate_dispatch_data import N_ZONES, build_dataset, sample_batch
from seer.models.ml.dispatch import DispatchNet, dispatch_forward, dispatch_loss

T_STEPS = 24   # one day per training sample


def train(n_epochs=200, n_instances=20, batch_size=64, lr=1e-3, device="cpu"):
    dataset = build_dataset(n_instances)

    model = DispatchNet(N_ZONES, T_STEPS).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        injection, marginal_cost, gen_capacity, storage_energy_capacity, storage_power_capacity, \
            voll, ptdf, capacity_matrix = sample_batch(dataset, batch_size, t_steps=T_STEPS, device=device)

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
