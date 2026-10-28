"""Noise: a small random jitter on every parameter of a seer-schema instance, applied after any scenario
levels and perturbations. Each value is scaled by (1 + u), u ~ U(-noise, noise), drawn as follows:

    dr, voll                 one u each
    demand growth            one u per zone: its average annual growth rate x (1 + u), trajectory regrown
    demand profile           one u per hour, except each zone's (and year's) peak hour, which no other hour
                             may exceed: the peak demand stays where demand growth puts it
    renewable profiles       one u per hour (solar, wind, hydro ...), clipped to [0, 1]
    capital costs            per tech (shared by its units in every zone): start x (1 + u_start), and the
                             pathway's change from the start x (1 + u_pathway)
    marginal costs           per fuel, as capital costs (fuel and variable O&M together)
    line capital costs       one u per line

Pathways: a cost c_t with start c_0 becomes c_0 (1 + u_start) (1 + (c_t / c_0 - 1)(1 + u_pathway)), so it
keeps its shape and still starts at c_0 (1 + u_start).
"""
import numpy as np

SETTINGS = ["dr", "voll"]


def add_noise(instance, noise, rng):
    """Copy of instance with every parameter jittered by relative noise (see the module docstring)."""
    out = instance.copy()
    if not noise:
        return out

    def u(size=None):
        return rng.uniform(-noise, noise, size)

    # dr, voll
    rows = out.settings["parameters"].isin(SETTINGS)
    out.settings.loc[rows, "value"] *= 1 + u(rows.sum())

    # demand growth: per zone, the trajectory's average annual growth rate x (1 + u), regrown in its own shape
    df = out.demand_forecast
    for _, group in df.groupby(["zone", "type"]):
        group = group.sort_values("year")
        years = group["year"] - group["year"].iloc[0]
        if years.iloc[-1] == 0:
            continue  # a single year has no growth
        growth_rate = (group["value"].iloc[-1] / group["value"].iloc[0]) ** (1 / years.iloc[-1]) - 1
        if abs(growth_rate) < 1e-9:
            continue  # a flat trajectory: no growth to jitter
        ratio = group["value"] / group["value"].iloc[0]
        ratio = ratio ** (np.log(1 + growth_rate * (1 + u())) / np.log(1 + growth_rate))
        df.loc[group.index, "value"] = group["value"].iloc[0] * ratio

    # demand profile: per hour, each zone's (and year's) peak hour kept and never exceeded
    df = out.demand_profile
    keys = [c for c in ["zone", "year"] if c in df.columns]
    peak = df.groupby(keys)["value"].transform("max")
    jitter = 1 + u(len(df))
    jitter[df.index.get_indexer(df.groupby(keys)["value"].idxmax())] = 1
    df["value"] = (df["value"] * jitter).clip(upper=peak)

    # renewable profiles: per hour
    df = out.renewable_profiles
    df["value"] = (df["value"] * (1 + u(len(df)))).clip(0, 1)

    # capital costs per tech, marginal costs per fuel
    for costs, constants, id_col in [(out.generator_costs, out.generator_constants, "generator"),
                                     (out.storage_costs, out.storage_constants, "storage")]:
        group_of = {"capital": constants.set_index(id_col)["tech"]}
        if id_col == "generator":
            group_of["marginal"] = constants.set_index(id_col)["fuel"]
        for cost_type, group in group_of.items():
            rows = costs["cost_type"] == cost_type
            add_cost_noise(costs, rows, costs.loc[rows, id_col].map(group), u)

    # line capital costs: per line
    if "capital" in out.network.columns:
        out.network["capital"] *= 1 + u(len(out.network))
    return out


def add_cost_noise(costs, rows, groups, u):
    """costs' rows (cost trajectories over its year columns) jittered, one (u_start, u_pathway) per group."""
    year_cols = [c for c in costs.columns if str(c).isdigit()]
    for group in sorted(groups.dropna().unique()):
        u_start, u_pathway = u(), u()
        index = groups.index[groups == group]
        trajectory = costs.loc[index, year_cols].to_numpy(dtype=float)
        start = trajectory[:, :1]
        pathway = np.divide(trajectory, start, out=np.ones_like(trajectory), where=start != 0)
        costs.loc[index, year_cols] = start * (1 + u_start) * (1 + (pathway - 1) * (1 + u_pathway))
