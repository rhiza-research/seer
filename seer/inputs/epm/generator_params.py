"""Build generator_constants.csv and generator_costs.csv from EPM input data."""
import uuid
import warnings
from pathlib import Path

import pandas as pd

from ..utils import standardize_tech, standardize_fuel, default_fuel_for_tech
from .data_checks import check_generators, check_capex_trajectory, check_profile_time_coverage
from .re_potentials import build_re_potentials
from .utils import init_df, pivot_wide, pivot_years, align

# sentinel for "no known retirement year" -- well beyond any realistic modeled horizon
NO_RETIREMENT_YEAR = 9999

# required generator parameters
PARAM_LIST = [
    "Capacity", "Capex", "RampDnRate", "RampUpRate", "Life",
    "HeatRate", "RETech", "RetrYr", "StYr", "UnitSize", "BuildLimitperYear", "MaxTotalBuild",
]


def build_generator_params(input_dir, valid_zones=None, model_years=None, times=None):
    """Builds generator_constants.csv and generator_costs.csv from EPM's raw generator files."""

    default_gen_params = init_df(input_dir + "pGenDataInputDefault.csv",
                                ["zone", "tech", "fuel", "parameters", "value"])
    custom_gen_params = pd.read_csv(f"{input_dir}pGenDataInputCustom.csv")

    # get the list of generators; either from gmap or inferred from custom_gen_params
    gmap_path = Path(f"{input_dir}gmap.csv")
    if gmap_path.exists():
        all_generators = pd.read_csv(gmap_path)
    else:
        all_generators = custom_gen_params[["generator", "zone", "tech", "fuel"]]\
            .drop_duplicates(subset="generator")\
            .reset_index(drop=True)


    # generator cost trajectories
    default_gen_capex = init_df(f"{input_dir}pCapexTrajectoriesDefault.csv",
                                ["zone", "tech", "fuel", "year", "value"])
    custom_gen_capex = init_df(f"{input_dir}pCapexTrajectoriesCustom.csv",
                                ["plant", "year", "value"])
    fuel_prices = pd.read_csv(f"{input_dir}pFuelPrice.csv")
    # storage plants - used only to exclude generators that are actually storage
    storage_plants = init_df(f"{input_dir}pStorageDataInput.csv",
                            ["plant", "year", "parameters", "value"])

    generators = check_generators(all_generators, storage_plants, valid_zones)

    """fill generator parameters from various sources"""
    gztf = ["generator", "zone", "tech", "fuel"]
    ztf = ["zone", "tech", "fuel"]
    tf = ["tech", "fuel"]
    # match generators with defaults, more specific to less specific
    defaults_ztf = generators.merge(pivot_wide(default_gen_params, ztf, PARAM_LIST), on=ztf, how="left")[PARAM_LIST]
    defaults_tf = generators.merge(pivot_wide(default_gen_params, tf, PARAM_LIST), on=tf, how="left")[PARAM_LIST]
    defaults = defaults_ztf.combine_first(defaults_tf)
    # match generators with custom
    customs = generators.merge(pivot_wide(custom_gen_params, "generator", PARAM_LIST), on="generator", how="left")[PARAM_LIST]
    # fill with custom, then defaults
    filled = customs.combine_first(defaults).reset_index(drop=True)
    generator_constants = pd.concat([generators[gztf].reset_index(drop=True), filled], axis=1)

    generator_constants = _derive_flags(generator_constants, input_dir)

    """build generator costs"""
    generator_cost = build_costs(generator_constants, default_gen_capex, custom_gen_capex, fuel_prices, model_years)
    # capex/heat_rate/maxtotalbuild were only needed to build generator_cost and can now be dropped
    generator_constants = generator_constants.drop(columns=["capex", "heat_rate", "maxtotalbuild"])

    """standardize tech & fuel names"""
    generator_constants["tech"] = standardize_tech(generator_constants["tech"], generator_constants["fuel"])
    generator_constants["fuel"] = generator_constants["fuel"].replace("CustomFuel", pd.NA)
    generator_constants["fuel"] = standardize_fuel(generator_constants["fuel"])
    missing_fuel = generator_constants["fuel"].isna()
    if missing_fuel.any():
        generator_constants.loc[missing_fuel, "fuel"] = (
            generator_constants.loc[missing_fuel, "tech"].map(default_fuel_for_tech)
        )

    """match renewable profiles"""
    profiles_raw = build_re_potentials(input_dir)
    generator_constants, renewable_profiles = match_re_profiles(generator_constants, profiles_raw, times)

    # reorder for readability
    generator_constants = generator_constants.sort_values(["expandable", "fuel", "tech"]).reset_index(drop=True)

    # generator_costs.csv rows should follow the same generator order as generator_constants.csv
    generator_cost = align(generator_cost, generator_constants, "generator")

    return generator_constants, generator_cost, renewable_profiles


def _derive_flags(gc, data_path):
    """Derive expandable/isre/retirement_year/life, and null out columns only meaningful
    for one of expandable/existing."""
    # MaxTotalBuild fallback when both BuildLimitperYear and Capacity are missing (see
    # data/processed_gams/README.md re: this vs. the Status-based fixes)
    no_limit_or_cap = gc["BuildLimitperYear"].isna() & gc["Capacity"].isna()
    gc.loc[no_limit_or_cap, "BuildLimitperYear"] = gc.loc[no_limit_or_cap, "MaxTotalBuild"]

    # expandable: a real nonzero build limit, or no capacity but real cost data to build from
    gc["expandable"] = (
        (gc["BuildLimitperYear"].notna() & (gc["BuildLimitperYear"] != 0))
        | (gc["Capacity"].isna() & gc["Capex"].notna())
    )
    gc.loc[gc["expandable"], "Capacity"] = pd.NA   # only existing generators have a capacity
    gc.loc[~gc["expandable"], "UnitSize"] = pd.NA  # only expandable generators have a unit_size
    gc["RampUpRate"] = gc["RampUpRate"].fillna(1)  # default: fully flexible
    gc["RampDnRate"] = gc["RampDnRate"].fillna(-1)

    gc.rename(columns={
        "BuildLimitperYear": "annual_build_limit",
        "RampUpRate": "ramp_up_rate",
        "RampDnRate": "ramp_down_rate",
        "HeatRate": "heat_rate",
        "StYr": "start_year",
        "RetrYr": "retirement_year",
        "UnitSize": "unit_size",
        "RETech": "isre",
    }, inplace=True)

    # isre must be a real bool -- bool(nan) is True in Python, so a plain fillna+astype would
    # misclassify every generator with no RETech value as renewable
    gc["isre"] = gc["isre"].infer_objects(copy=False).fillna(0).astype(bool)
    # no retirement year -> sentinel far beyond the horizon, not NaN (NaN fails every
    # `year <= retirement_year` check, dropping the generator from every modeled year)
    gc["retirement_year"] = gc["retirement_year"].infer_objects(copy=False).fillna(NO_RETIREMENT_YEAR)

    # no Life -> derive from the generator's own known active window
    missing_life = gc["Life"].isna()
    gc.loc[missing_life, "Life"] = gc.loc[missing_life, "retirement_year"] - gc.loc[missing_life, "start_year"]
    still_missing = gc["Life"].isna()
    if still_missing.any():
        warnings.warn(f"No Life value in {data_path!r} for generators: {gc.loc[still_missing, 'generator'].tolist()}")

    gc.columns = gc.columns.str.lower()
    return gc


def build_costs(generator_constants, default_gen_capex, custom_gen_capex, fuel_prices, model_years):
    """generator_costs.csv: two rows per generator (capex and marginal cost), years as
    columns. capex = capex trajectory x capex scalar. marginal_cost = heat_rate x fuel price."""

    ztf = ["zone", "tech", "fuel"]
    gc = generator_constants.set_index("generator")

    default_wide = pivot_years(default_gen_capex, ztf, sorted(default_gen_capex["year"].unique()))
    custom_gen_capex = custom_gen_capex.rename(columns={"plant": "generator"})
    custom_wide = pivot_years(custom_gen_capex, "generator", sorted(custom_gen_capex["year"].unique()))

    """build capex trajectory"""
    flat = check_capex_trajectory(default_wide, custom_wide, model_years)
    if flat:
        capex_traj = pd.DataFrame({y: 1.0 for y in model_years}, index=gc.index)
    else:
        # downselect both trajectories to just the modeled years
        default_wide = default_wide.reindex(columns=model_years)
        custom_wide = custom_wide.reindex(columns=model_years)
        custom_traj = custom_wide.reindex(gc.index)
        default_traj = gc.reset_index().merge(default_wide, on=ztf, how="left").set_index("generator")[model_years]
        has_custom = pd.Series(gc.index.isin(custom_gen_capex["generator"]), index=gc.index)
        capex_traj = custom_traj.where(has_custom, default_traj)
    # scale normalized trajectory by generator's own capex
    capex_cost = capex_traj.mul(gc["capex"], axis=0)

    """build marginal cost"""
    fuel_price_traj = gc.reset_index().merge(
        pivot_years(fuel_prices, ["fuel", "zone"], model_years), on=["fuel", "zone"], how="left"
    ).set_index("generator")[model_years]
    marginal_cost = fuel_price_traj.mul(gc["heat_rate"], axis=0)
    # marginal cost is 0 for generators with no heat_rate
    marginal_cost = marginal_cost.where(gc["heat_rate"].fillna(0) != 0, 0.0)

    """concatenate capex and marginal cost"""
    # row order is finalized later, to match generator_constants.csv
    generator_cost = pd.concat([
        capex_cost.assign(cost_type="capex"),
        marginal_cost.assign(cost_type="marginal_cost"),
    ]).reset_index()
    generator_cost = generator_cost[["generator", "cost_type"] + model_years]
    return generator_cost


def match_re_profiles(gc, profiles_raw, times):
    """Resolve each generator to at most one renewable profile, by priority: generator-
    specific, then zone+tech generic, then tech-only fallback (see seer/inputs/README.md).
    Every distinct profile gets its own profile_id, independent of that priority order --
    identity is just "which (zone, tech, generator) this is" -- shared by every generator that
    resolves to it, so matching downstream is a plain id lookup, not a re-derived
    zone/tech/generator priority chain. isre is recomputed from whether a match was found, not
    the raw RETech flag. Returns (gc, renewable_profiles)."""
    profiles_raw = check_profile_time_coverage(profiles_raw, times)

    # keep only profiles that actually reference something real in gc: generator-specific,
    # zone+tech generic, or tech-only fallback
    has_tech = profiles_raw["tech"].notna()
    has_zone = profiles_raw["zone"].notna()
    zone_tech = profiles_raw[has_tech & has_zone].merge(
        gc[["zone", "tech"]].drop_duplicates(), on=["zone", "tech"], how="inner"
    )
    tech_only = profiles_raw[has_tech & ~has_zone]
    tech_only = tech_only[tech_only["tech"].isin(gc["tech"])]
    specific = profiles_raw[~has_tech & profiles_raw["generator"].isin(gc["generator"])]
    profiles_raw = pd.concat([zone_tech, tech_only, specific], ignore_index=True)

    # assign a unique profile_id to each distinct (zone, tech, generator)
    keys = profiles_raw[["zone", "tech", "generator"]].drop_duplicates()
    keys["profile_id"] = [str(uuid.uuid4()) for _ in range(len(keys))]
    profiles_raw = profiles_raw.merge(keys, on=["zone", "tech", "generator"])

    # resolve each generator to (at most) one profile_id: try generator-specific, then
    # zone+tech generic, then tech-only fallback, keeping the first match found
    lookups = [
        (keys[keys["generator"].notna()], ["generator"]),
        (keys[keys["tech"].notna() & keys["zone"].notna()], ["zone", "tech"]),
        (keys[keys["tech"].notna() & keys["zone"].isna()], ["tech"]),
    ]
    profile_id = pd.Series(pd.NA, index=gc.index, dtype="object")
    for tier_keys, key_cols in lookups:
        ids = tier_keys.drop_duplicates(key_cols)[key_cols + ["profile_id"]]
        matched = gc[key_cols].reset_index().merge(ids, on=key_cols, how="left").set_index("index")["profile_id"]
        profile_id = profile_id.where(profile_id.notna(), matched)

    claimed_but_unmatched = gc["isre"] & profile_id.isna()
    for _, gen in gc[claimed_but_unmatched].iterrows():
        warnings.warn(
            f"Generator {gen.generator!r} (zone={gen.zone}, tech={gen.tech}) is flagged "
            "renewable but has no matching profile -- marking as non-renewable."
        )

    gc = gc.copy()
    gc["profile_id"] = profile_id
    gc["isre"] = profile_id.notna()
    renewable_profiles = profiles_raw[["profile_id", "zone", "tech", "generator", "time", "value"]]
    return gc, renewable_profiles


