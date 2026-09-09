"""Build generator_constants.csv and generator_costs.csv from EPM input data."""
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# sentinel for "no known retirement year" -- well beyond any realistic modeled horizon
NO_RETIREMENT_YEAR = 9999


def build_generator_params(data_path="data/Gambia/"):
    """Combines disparate generator data sources into two seer schema files:
    generator_constants.csv: a dataframe of generator constants (parameters, build limits, etc.)
    generator_costs.csv: a dataframe of generator costs (capex and marginal cost)
    Returns two dataframes: generator_constants and generator_costs.
    """
    input_data = load_input_data(data_path)
    all_generators = input_data["all_generators"]
    default_gen_params = input_data["default_gen_params"]
    custom_gen_params = input_data["custom_gen_params"]
    default_gen_capex = input_data["default_gen_capex"]
    custom_gen_capex = input_data["custom_gen_capex"]
    fuel_prices = input_data["fuel_prices"]

    """Setup generator constants.
    Use generator list, default parameters, and custom parameters to fill in a single dataframe of generator constants.
    """
    # Capex and HeatRate are needed only to compute generator_costs below (capex scaling
    # multiplier, marginal cost from heat_rate x fuel price); dropped from the final output,
    # since they're already baked into generator_costs.csv by the time this returns.
    param_list = [
        "Capacity", "Capex", "RampDnRate", "RampUpRate", "Life",
        "HeatRate", "RETech", "RetrYr", "StYr", "UnitSize", "BuildLimitperYear",
    ]
    # Initialize generator_constants dataframe
    generator_constants = pd.DataFrame(columns=["generator", "zone", "tech", "fuel"] + param_list)
    generator_constants[["generator", "zone", "tech", "fuel"]] = all_generators[["generator", "zone", "tech", "fuel"]]

    # Fill in parameters from defaults first
    for idx, gen in generator_constants.iterrows():
        for param in param_list:
            # find default row for this zone/tech/fuel/parameter
            mask = (
                (default_gen_params["zone"] == gen.zone) &
                (default_gen_params["tech"] == gen.tech) &
                (default_gen_params["fuel"] == gen.fuel) &
                (default_gen_params["parameters"] == param)
            )
            if mask.any():
                generator_constants.loc[idx, param] = default_gen_params.loc[mask, "value"].values[0]
            # override with custom parameters where available
            mask = (
                (custom_gen_params["generator"] == gen.generator) &
                (custom_gen_params["parameters"] == param)
            )
            if mask.any():
                generator_constants.loc[idx, param] = custom_gen_params.loc[mask, "value"].values[0]
    
    
    # Create standard columns PyPSA expects
    # this is a build option if buildlimitperyear is not nan
    generator_constants["expandable"] = generator_constants["BuildLimitperYear"].notna()
    # rename columns to PyPSA conventions
    generator_constants.rename(columns={
        "BuildLimitperYear": "annual_build_limit",
        "RampUpRate": "ramp_up_rate",
        "RampDnRate": "ramp_down_rate",
        "HeatRate": "heat_rate",
        "StYr": "start_year",
        "RetrYr": "retirement_year",
        "UnitSize": "unit_size",
        "RETech": "isre",
    }, inplace=True)
    # isre is a boolean flag -- no RETech/VRETech value found means "not a variable-renewable
    # tech", not "unknown"; must be a real bool here, since bool(float("nan")) is True in
    # Python and would otherwise silently misclassify every such generator as renewable
    generator_constants["isre"] = generator_constants["isre"].infer_objects(copy=False).fillna(0).astype(bool)
    # no RetrYr value found means "no known retirement" -- default to a sentinel year well
    # beyond any realistic modeled horizon, not NaN. A NaN retirement_year makes every
    # `year <= retirement_year` comparison False (both model backends' active-year checks work
    # this way), which silently drops the generator/its capacity from every modeled year
    # instead of leaving it active for the full horizon as intended.
    generator_constants["retirement_year"] = (
        generator_constants["retirement_year"].infer_objects(copy=False).fillna(NO_RETIREMENT_YEAR)
    )
    # change all capital leters to lowercase in column names
    generator_constants.columns = generator_constants.columns.str.lower()


    """Setup generator costs.
    Use generator constants, default capex trajectories, and custom capex trajectories to fill in a single dataframe of generator costs.
    Result has two rows per generator (capex and marginal cost) and years as columns.
    """
    # setup cost dataframe
    # (years normally comes from the default capex trajectory; some regions have no
    # default capex data at all, so fall back to whatever years the custom trajectory covers)
    if not default_gen_capex.empty:
        years = sorted(default_gen_capex["year"].unique())
    elif not custom_gen_capex.empty:
        years = sorted(custom_gen_capex["year"].unique())
    else:
        years = []
    rows = []

    for _, gen in generator_constants.iterrows():

        # capex
        row_capex = {"generator": gen.generator, "cost_type": "capex"}

        # get capex trajectory - custom if it exists, otherwise default
        if gen.generator in custom_gen_capex["plant"].values:
            capex_traj = custom_gen_capex[
                custom_gen_capex["plant"] == gen.generator
            ][["year", "value"]].set_index("year")
        else:
            capex_traj = default_gen_capex[
                (default_gen_capex["zone"] == gen.zone) &
                (default_gen_capex["tech"] == gen.tech) &
                (default_gen_capex["fuel"] == gen.fuel)
            ][["year", "value"]].set_index("year")

        capex_traj = gen.capex * capex_traj["value"]

        for y in years:
            row_capex[y] = capex_traj.get(y, pd.NA)

        rows.append(row_capex)

        # marginal cost -- always write a row (0 when there's no heat_rate at all, e.g.
        # renewables that don't burn fuel), so every generator has both capex and
        # marginal_cost rows in generator_costs.csv
        row_mc = {"generator": gen.generator, "cost_type": "marginal_cost"}
        if pd.notna(gen.heat_rate):
            # fuel prices are zone-specific (pFuelPrice.csv has a zone column) -- filtering
            # by fuel alone matches every zone's price row for that fuel/year, producing a
            # multi-row Series instead of a scalar per year
            fuel_price = fuel_prices[
                (fuel_prices["fuel"] == gen.fuel) & (fuel_prices["zone"] == gen.zone)
            ][["year", "value"]].set_index("year")

            mc = fuel_price["value"] * gen.heat_rate

            for y in years:
                row_mc[y] = mc.get(y, pd.NA)
        else:
            for y in years:
                row_mc[y] = 0

        rows.append(row_mc)

    generator_cost = pd.DataFrame(rows, columns=["generator", "cost_type"] + years)

    # capex/heat_rate were only needed above (capex scaling multiplier, marginal-cost
    # computation) -- already baked into generator_cost, so drop from the final output
    generator_constants = generator_constants.drop(columns=["capex", "heat_rate"])
    return generator_constants, generator_cost


def _read_csv_or_empty(path, columns):
    """Read a csv, or return an empty dataframe with the given columns if it doesn't exist
    (some regions' raw gdx simply has no data for certain symbols)."""
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def load_input_data(data_path):
    """Load EPM data for generators from the data path.
    Files that are read containing generator data:
    - gmap.csv: list of all generators including name, zone, fuel (if available — some
      regions' raw gdx has no gmap symbol at all, see below)
    - pGenDataInputDefault.csv: generator parameters (by zone, tech, fuel)
    - pGenDataInputCustom.csv: generator parameters (by generator)
    - pCapexTrajectoriesDefault.csv: generator capex (by zone, tech, fuel)
    - pCapexTrajectoriesCustom.csv: generator capex (by generator)
    - pFuelPrice.csv: fuel prices

    Returns a dictionary of dataframes.
    """
    # generator parameters (by zone, tech, fuel) -- some regions have no default params at
    # all (no such symbol in their raw gdx); treat as empty rather than failing, so those
    # params just stay NaN on generator_constants (to be filled in properly later)
    default_gen_params = _read_csv_or_empty(
        data_path + "pGenDataInputDefault.csv", ["zone", "tech", "fuel", "parameters", "value"]
    )
    custom_gen_params = pd.read_csv(data_path + "pGenDataInputCustom.csv")

    # list of all generators including name, zone, fuel
    gmap_path = Path(data_path + "gmap.csv")
    if gmap_path.exists():
        all_generators = pd.read_csv(gmap_path)
    else:
        # This region's raw gdx has no gmap symbol at all. gdx's enrich_gen_data already
        # backfills zone/tech/fuel into pGenDataInputCustom per generator in this case
        # (via pZoneIndex/ftfindex), so derive the generator list from there instead.
        all_generators = (
            custom_gen_params[["generator", "zone", "tech", "fuel"]]
            .drop_duplicates(subset="generator")
            .reset_index(drop=True)
        )

    # generator names must be unique -- they're the primary key joining generator_constants.csv
    # <-> generator_costs.csv, and are used for component naming in both model backends. Some
    # regions' raw gmap.csv reuses the same name for two conceptually different plants (e.g. a
    # hydro dam and a co-located PV installation both literally called "Lauca" -- see CAPP).
    # Keep only the first occurrence and warn, rather than silently producing duplicate/
    # ambiguous rows downstream.
    dupes = all_generators["generator"].duplicated(keep=False)
    if dupes.any():
        dup_names = sorted(all_generators.loc[dupes, "generator"].unique())
        warnings.warn(
            f"Duplicate generator names found in {data_path!r} (keeping first occurrence of "
            f"each, dropping the rest): {dup_names}"
        )
        all_generators = all_generators.drop_duplicates(subset="generator", keep="first")

    # generator capex (by zone, tech, fuel) -- same story: treat a missing default or
    # custom capex trajectory as empty rather than failing, so build_generator_params
    # falls back to whichever of the two is actually available
    default_gen_capex = _read_csv_or_empty(
        data_path + "pCapexTrajectoriesDefault.csv", ["zone", "tech", "fuel", "year", "value"]
    )
    custom_gen_capex = _read_csv_or_empty(
        data_path + "pCapexTrajectoriesCustom.csv", ["plant", "year", "value"]
    )
    # fuel prices
    fuel_prices = pd.read_csv(data_path + "pFuelPrice.csv")

    return {
        "all_generators": all_generators,
        "default_gen_params": default_gen_params,
        "custom_gen_params": custom_gen_params,
        "default_gen_capex": default_gen_capex,
        "custom_gen_capex": custom_gen_capex,
        "fuel_prices": fuel_prices,
    }
