"""Solve a built PyPSA seer model and print a summary of the results."""

import time

import pandas as pd

SOLVER_NAME = "highs"  # matches the HiGHS solver used on the pyomo side


def solve(model, solver_name=SOLVER_NAME, verbose=True, **optimize_kwargs):
    """Solve a PyPSA Network in place and print a summary of the results.
    Returns (status, condition, solve_time)."""
    start_time = time.perf_counter()
    status, condition = model.optimize(solver_name=solver_name, **optimize_kwargs)
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


def _base_name(name, build_year):
    """Strip the "_<build_year>" suffix used for per-vintage new-build components (see
    build_generators.py/build_storage.py: `gen_year_name = name + "_" + str(year)`), so multiple
    vintages of the same build option collapse back into one logical name, matching the fixed
    per-option names used on the pyomo side."""
    suffix = f"_{int(build_year)}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _capacity_by_year(model, static_df):
    """static_df: model.generators or model.storage_units (must have build_year, lifetime,
    p_nom_opt). Returns a DataFrame indexed by investment year, one column per logical (base)
    name, summing p_nom_opt across every vintage active in that year."""
    base_names = pd.Series(
        [_base_name(name, row.build_year) for name, row in static_df.iterrows()],
        index=static_df.index,
    )

    rows = {}
    for year in model.investment_periods:
        active = (static_df["build_year"] <= year) & (year < static_df["build_year"] + static_df["lifetime"])
        rows[year] = static_df.loc[active, "p_nom_opt"].groupby(base_names[active]).sum()

    df = pd.DataFrame(rows).T.fillna(0.0)
    df.index.name = None
    df.columns.name = None
    return df


def _timeseries_by_base_name(static_df, t_df):
    """t_df: e.g. model.generators_t.p, indexed by model.snapshots (year, timestamp), columns =
    component names. Collapses vintage-specific columns to their logical base name (summing any
    that overlap) and reindexes to a plain (year, hour) index, hour = 1-indexed position within
    each year - matching the pyomo side's convention."""
    base_name_map = {name: _base_name(name, static_df.loc[name, "build_year"]) for name in t_df.columns}
    renamed = t_df.rename(columns=base_name_map)
    renamed = renamed.T.groupby(level=0).sum().T

    years = renamed.index.get_level_values(0)
    hours = pd.Series(range(len(renamed)), index=renamed.index).groupby(years).cumcount() + 1
    renamed.index = pd.MultiIndex.from_arrays([years, hours.values], names=["year", "hour"])
    renamed.columns.name = None
    return renamed


def _genmap(static_df):
    """static_df: model.generators or model.storage_units. Returns a DataFrame indexed by
    logical (base) name, columns=[tech, zone], one row per distinct build option (collapsing
    vintages, same as _capacity_by_year/_timeseries_by_base_name)."""
    base_names = [_base_name(name, row.build_year) for name, row in static_df.iterrows()]
    df = pd.DataFrame({"tech": static_df["carrier"].values, "zone": static_df["bus"].values}, index=base_names)
    return df[~df.index.duplicated()]


def build_results_dataframes(model):
    """Return solved-model outputs as a dict of DataFrames (same shape as the pyomo backend):
    - capacity_df: index=years, columns=generators, values=total capacity (MW)
    - generation_df: index=(year, hour), columns=generators, values=output (MW)
    - storage_capacity_df: index=years, columns=storage, values=total energy capacity (MWh)
    - storage_change_df: index=(year, hour), columns=storage, values=charge(+)/discharge(-) (MW)
    - soc_df: index=(year, hour), columns=storage, values=state of charge (MWh)
    - genmap_df: index=generators+storage, columns=[tech, zone]
    """
    capacity_df = _capacity_by_year(model, model.generators)
    generation_df = _timeseries_by_base_name(model.generators, model.generators_t.p)

    storage_capacity_df = _capacity_by_year(model, model.storage_units)
    # PyPSA's StorageUnit.p convention is the opposite of the pyomo side's: positive = discharging.
    # Negate so positive = charging here too.
    storage_change_df = -_timeseries_by_base_name(model.storage_units, model.storage_units_t.p)
    soc_df = _timeseries_by_base_name(model.storage_units, model.storage_units_t.state_of_charge)

    genmap_df = pd.concat([_genmap(model.generators), _genmap(model.storage_units)])
    genmap_df.index.name = None

    return {
        "capacity_df": capacity_df,
        "generation_df": generation_df,
        "storage_capacity_df": storage_capacity_df,
        "storage_change_df": storage_change_df,
        "soc_df": soc_df,
        "genmap_df": genmap_df,
    }
