import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

FLOW_CMAP = LinearSegmentedColormap.from_list("light_to_dark_red", ["#FFD6D6", "#8B0000"])


def build_susceptance_matrix(n_nodes, lines):
    """lines: list of (i, j, capacity) tuples, one entry per physical line (i-j only, never both
    directions). Capacity stands in for admittance (as in the original design): higher capacity
    -> higher admittance. Returns the symmetric (n_nodes, n_nodes) susceptance matrix B."""
    B = np.zeros((n_nodes, n_nodes))
    for i, j, capacity in lines:
        B[i, j] -= capacity
        B[j, i] -= capacity  # the fix: a line's admittance is symmetric, so both entries need it
    for i in range(n_nodes):
        B[i, i] = -B[i, :].sum()
    return B


def build_line_sensitivity_matrix(n_nodes, lines):
    """One row per physical line (not per ordered pair, so no phantom zero-capacity rows)."""
    Bline = np.zeros((len(lines), n_nodes))
    for idx, (i, j, capacity) in enumerate(lines):
        Bline[idx, i] = capacity
        Bline[idx, j] = -capacity
    return Bline


def compute_ptdf(n_nodes, lines, slack_node=0):
    """The simple entry point: pass the node count and a list of (i, j, capacity) lines, get back
    the (n_lines, n_nodes - 1) PTDF matrix (columns correspond to every node except slack_node)."""
    B = build_susceptance_matrix(n_nodes, lines)
    Bline = build_line_sensitivity_matrix(n_nodes, lines)

    non_slack = [n for n in range(n_nodes) if n != slack_node]
    B_reduced = B[np.ix_(non_slack, non_slack)]
    Bline_reduced = Bline[:, non_slack]

    return Bline_reduced @ np.linalg.inv(B_reduced)


def compute_flow_changes(ptdf, injections, slack_node=0):
    """injections: full-length (n_nodes,) vector of injection changes (should sum to zero).
    Returns the resulting change in flow on each line (n_lines,)."""
    non_slack = [n for n in range(len(injections)) if n != slack_node]
    return ptdf @ injections[non_slack]  # matrix-vector product, not element-wise


def circular_layout(n_nodes):
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    return {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}


def plot_network(ax, n_nodes, lines, flow_changes=None, injections=None, node_positions=None):
    """Draw branches as pipes, width proportional to capacity. If flow_changes is given (one
    value per line, matching `lines` order), each branch is drawn as an arrow pointing in the
    actual flow direction (reversed from the line's defined (i, j) orientation when the value is
    negative), colored and labeled by the flow *magnitude* (always non-negative - direction is
    carried by the arrow, not by sign). If injections is given (one value per node), a small
    external arrow at each node shows |p|, pointing inward for injections (p > 0) and outward for
    extractions (p < 0) - same "positive magnitude, direction carries the sign" convention."""
    node_positions = node_positions or circular_layout(n_nodes)

    capacities = np.array([capacity for _, _, capacity in lines])
    max_capacity = capacities.max() if len(capacities) else 1.0

    if flow_changes is not None:
        magnitudes = np.abs(flow_changes)
        max_magnitude = max(magnitudes.max(), 1e-9)
        norm = plt.Normalize(vmin=0, vmax=max_magnitude)
        cmap = FLOW_CMAP  # sequential (light -> dark red), since color now encodes magnitude only

    for idx, (i, j, capacity) in enumerate(lines):
        linewidth = 2 + 8 * (capacity / max_capacity)

        if flow_changes is not None:
            magnitude = abs(flow_changes[idx])
            src, dst = (i, j) if flow_changes[idx] >= 0 else (j, i)
            color = cmap(norm(magnitude))
        else:
            src, dst = i, j
            color = "steelblue"

        start, end = node_positions[src], node_positions[dst]
        arrow = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=20,
            linewidth=linewidth, color=color, shrinkA=15, shrinkB=15, zorder=1,
        )
        ax.add_patch(arrow)

        if flow_changes is not None:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(
                mx, my, f"{magnitude:.2f} | {capacity:g}", fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
                zorder=3,
            )

    if injections is not None:
        centroid = np.mean(list(node_positions.values()), axis=0)
        max_injection = max(np.abs(injections).max(), 1e-9)

        for node, (x, y) in node_positions.items():
            p = injections[node]
            if abs(p) < 1e-9:
                continue

            direction = np.array([x, y]) - centroid
            direction /= np.linalg.norm(direction) + 1e-12
            outer_point = np.array([x, y]) + direction * (0.3 + 0.4 * abs(p) / max_injection)

            if p > 0:
                # injection: arrow points inward, from outside the network into the node
                start, end, shrinkA, shrinkB = outer_point, (x, y), 2, 15
            else:
                # extraction: arrow points outward, from the node out of the network
                start, end, shrinkA, shrinkB = (x, y), outer_point, 15, 2

            arrow = FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=15,
                linewidth=2, color="black", shrinkA=shrinkA, shrinkB=shrinkB, zorder=0,
            )
            ax.add_patch(arrow)

            label_point = outer_point + direction * 0.15
            ax.text(label_point[0], label_point[1], f"{abs(p):.2f}", fontsize=8, ha="center", va="center", zorder=5)

    for node, (x, y) in node_positions.items():
        ax.plot(x, y, "o", markersize=20, color="black", zorder=2)
        ax.text(x, y, str(node), color="white", fontsize=10, ha="center", va="center", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    n_nodes = 3
    lines = [(0, 1, 10), (1, 2, 1), (2, 0, 30)]  # (from, to, capacity) - i-j only

    ptdf = compute_ptdf(n_nodes, lines, slack_node=0)

    # perturbation: inject +1 at src_node, withdraw -1 at dst_node
    src_node, dst_node = 0, 1
    injections = np.zeros(n_nodes)
    injections[src_node] = 1
    injections[dst_node] = -1

    flow_changes = compute_flow_changes(ptdf, injections, slack_node=0)
    print(flow_changes)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_network(ax, n_nodes, lines, flow_changes=flow_changes, injections=injections)
    plt.show()

    breakpoint()
