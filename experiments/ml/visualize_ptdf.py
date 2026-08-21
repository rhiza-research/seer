"""Small exploration script: visualize a PTDF matrix (imshow) next to the network it came from
(edge width = line capacity, arrows = resulting line flows), for a given set of line capacities
and nodal injections. Reuses ptdf.py's PTDF (capacity-as-admittance, not impedance) and
plot_network as-is -- this file is just a thin wrapper combining the two into one figure.

Run directly for a demo: python experiments/ml/visualize_ptdf.py
"""

import numpy as np
import matplotlib.pyplot as plt

from experiments.ml.ptdf import compute_ptdf, compute_flow_changes, plot_network


def visualize_ptdf(n_nodes, lines, injections, slack_node=0, node_positions=None, figsize=(14, 6),
                    savepath=None):
    """
    lines       : list of (i, j, capacity) -- one entry per physical line, i-j only
    injections  : (n_nodes,) array of nodal injections (should sum to ~0)
    slack_node  : reference node (PTDF has n_nodes-1 columns -- every node except this one)

    Plots side by side:
        left  - PTDF as a heatmap (line x non-slack node)
        right - the network: edge width = capacity, arrows = resulting flow direction/magnitude
                (see ptdf.py's plot_network for exactly how that's drawn)

    Returns (fig, ptdf, flows) so you can keep playing with the numbers in a notebook/REPL.
    """
    injections = np.asarray(injections, dtype=float)
    assert len(injections) == n_nodes

    ptdf = compute_ptdf(n_nodes, lines, slack_node=slack_node)
    flows = compute_flow_changes(ptdf, injections, slack_node=slack_node)

    fig, (ax_ptdf, ax_net) = plt.subplots(1, 2, figsize=figsize)

    # ---- left: PTDF heatmap ----
    vmax = np.abs(ptdf).max() or 1.0
    im = ax_ptdf.imshow(ptdf, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    non_slack = [n for n in range(n_nodes) if n != slack_node]
    ax_ptdf.set_xticks(range(len(non_slack)))
    ax_ptdf.set_xticklabels(non_slack)
    ax_ptdf.set_yticks(range(len(lines)))
    ax_ptdf.set_yticklabels([f"{i}-{j}" for i, j, _ in lines])
    ax_ptdf.set_xlabel(f"node (slack = {slack_node}, omitted)")
    ax_ptdf.set_ylabel("line")
    ax_ptdf.set_title("PTDF")
    fig.colorbar(im, ax=ax_ptdf, shrink=0.8, label="flow per unit injection")

    # ---- right: network, edge width = capacity, arrows = flow ----
    plot_network(ax_net, n_nodes, lines, flow_changes=flows, injections=injections,
                 node_positions=node_positions)
    ax_net.set_title("edge width = capacity, arrow = flow (direction + magnitude)")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"saved to {savepath}")
    return fig, ptdf, flows


if __name__ == "__main__":
    # ---- play with these ----
    n_nodes = 4
    lines = [(0, 1, 40), (1, 2, 15), (2, 3, 25), (3, 0, 10), (0, 2, 20)]
    injections = np.array([10, 0, -10, 0], dtype=float)  # +10 at node 0, -10 (extraction) at node 2

    fig, ptdf, flows = visualize_ptdf(n_nodes, lines, injections, slack_node=0)
    print("PTDF:\n", ptdf)
    print("flows:", flows)
    plt.show()
