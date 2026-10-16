"""Solve a built PyPSA seer model and print a summary of the results."""

import time

import xarray as xr

SOLVER_NAME = "highs"  # matches the HiGHS solver used on the pyomo side


def add_total_build_limits(model):
    """Bound each build option's (generator, line or storage unit) capacity summed across all its build-year
    vintages by its total_build_limit (recorded in model.meta by build.build_generators /
    build.build_network) -- PyPSA's own
    p_nom_max only bounds one component at a time."""
    for component, key in [
        ("Generator", "total_build_limits"),
        ("Link", "line_total_build_limits"),
        ("StorageUnit", "storage_total_build_limits"),
    ]:
        limits = model.meta.get(key, {})
        if not limits:
            continue
        p_nom = model.model.variables[f"{component}-p_nom"]
        for name, spec in limits.items():
            model.model.add_constraints(
                p_nom.sel(name=spec["components"]).sum() <= spec["limit"],
                name=f"total_build_limit-{component}-{name}",
            )


def add_daily_storage_balance(model):
    """Each storage unit's state of charge must end every day where it started: net stored
    energy (after charging efficiency, minus discharge before discharging efficiency) sums to
    zero over each day. Same as the pyomo backend's daily cyclic rings. Without it, with
    weighted representative days (times.csv `weight`), storage could charge on a lightly
    weighted day and discharge on a heavily weighted one, creating energy in the weighted
    totals. Storage units are cyclic per period (see build.build_storage), so the shared
    start-of-day level is free."""
    if model.storage_units.empty:
        return
    m = model.model
    p_store, p_dispatch = m.variables["StorageUnit-p_store"], m.variables["StorageUnit-p_dispatch"]
    units = model.storage_units.loc[p_store.coords["name"].values]
    eff_store = xr.DataArray(units["efficiency_store"].values, coords={"name": units.index})
    eff_dispatch = xr.DataArray(units["efficiency_dispatch"].values, coords={"name": units.index})
    net = eff_store * p_store - (1 / eff_dispatch) * p_dispatch

    snapshots = model.snapshots
    day = [f"{period}-{ts.date()}" for period, ts in snapshots]
    day = xr.DataArray(day, coords={"snapshot": p_store.coords["snapshot"]}, dims="snapshot", name="day")
    m.add_constraints(net.groupby(day).sum() == 0, name="StorageUnit-daily_balance")


def solve(model, solver_name=SOLVER_NAME, verbose=True, **optimize_kwargs):
    """Solve a PyPSA Network in place and print a summary of the results.
    Returns (status, condition, solve_time)."""
    user_extra = optimize_kwargs.pop("extra_functionality", None)

    def extra_functionality(n, snapshots):
        add_total_build_limits(n)
        add_daily_storage_balance(n)
        if user_extra is not None:
            user_extra(n, snapshots)

    start_time = time.perf_counter()
    status, condition = model.optimize(
        solver_name=solver_name, extra_functionality=extra_functionality, **optimize_kwargs
    )
    solve_time = time.perf_counter() - start_time

    if condition != "optimal":
        raise RuntimeError(f"Solve did not reach optimality: status={status}, condition={condition}")

    if verbose:
        print(f"Solve time: {solve_time:.2f}s\n")
        print_summary(model)

    return status, condition, solve_time


def print_summary(model):
    print(f"Objective: {model.objective:,.0f}\n")

    print("Optimized generator capacity (MW):")
    for name, p_nom_opt in model.generators["p_nom_opt"].items():
        print(f"  {name:<24} {p_nom_opt:8.1f} MW")

    print("\nOptimized storage capacity (MW):")
    for name, p_nom_opt in model.storage_units["p_nom_opt"].items():
        print(f"  {name:<24} {p_nom_opt:8.1f} MW")
