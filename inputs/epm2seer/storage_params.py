"""Build storage_constants.csv and storage_costs.csv from EPM input data."""
import warnings

import pandas as pd

from .generator_params import NO_RETIREMENT_YEAR
from .standard_names import standardize_tech


def build_storage_params(data_path="data/Gambia/", years=None, valid_zones=None):
    """
    Processes storage input data into two seer schema files:
    - storage_constants.csv
    - storage_costs.csv

    Files that are read containing storage data:
    - pStorageDataInput.csv: storage data (by plant)
    - pGenDataInputCustom.csv: generator parameters (by generator)
    Returns two dataframes: storage_constants and storage_costs.

    `years` should be the same modeled years as years.csv (the model's investment periods) --
    storage_costs.csv must have a column for every one of them, since build_storage's
    per-vintage capex lookup indexes by those exact year values. Defaults to 2025-2050 for
    standalone use, but epm_to_seer always passes the region's actual years explicitly.
    """
    if years is None:
        years = list(range(2025, 2051))

    storage_data = pd.read_csv(data_path + "pStorageDataInput.csv")
    custom_gen_params = pd.read_csv(data_path + "pGenDataInputCustom.csv")

    """Setup storage constants.
    Use storage data to fill in a single dataframe of storage constants.
    """
    # pivot parameters into columns
    storage_constants = (
        storage_data
        .pivot(index="plant", columns="parameters", values="value")
        .reset_index()
    )

    # select and rename parameters (reindex, not [], since some regions have no "Life"
    # parameter at all for any storage plant -- that column just comes back all-NaN)
    param_list = ["plant", "CapacityMWh", "CapexMWh", "Efficiency", "Life"]
    param_names = ["storage", "capacity_mwh", "capex_mwh", "efficiency", "life"]
    storage_constants = storage_constants.reindex(columns=param_list).rename(columns=dict(zip(param_list, param_names)))

    """Pull the remaining attributes from the generator custom-parameters file - storage plants
    are just generators with tech="Storage" there - matched here by name, same as generator_constants.
    """
    storage_gen_params = custom_gen_params[custom_gen_params["generator"].isin(storage_constants["storage"])].copy()

    # some storage plants have no "CapexMWh" in pStorageDataInput.csv at all, but DO have a real
    # "Capex" value on their own generator-side record (e.g. Serbia's BESS2025_4h_Ser, Turkiye's
    # Gokcekaya_PHP, Lesotho's Monont'sa_HY_LSO) -- the same generator-side record zone/tech
    # already gets pulled from below. Fall back to it here rather than leaving a real, known
    # cost sitting unused in the wrong source file.
    missing_capex = storage_constants["capex_mwh"].isna()
    if missing_capex.any():
        gen_side_capex = storage_gen_params[storage_gen_params["parameters"] == "Capex"].set_index("generator")["value"]
        storage_constants.loc[missing_capex, "capex_mwh"] = (
            storage_constants.loc[missing_capex, "storage"].map(gen_side_capex)
        )

    # zone and tech (the storage carrier, e.g. "battery storage"/"PHS") are plain columns
    # rather than "parameters" rows. The raw "fuel" column here is the actual carrier
    # (Battery/Water/Pumped/...); the raw "tech" column (Storage/STO HY/...) is itself an
    # ambiguous placeholder that only disambiguates Water/Pumped/Hydro (pumped-hydro storage)
    # from Battery once paired with it -- see epm2seer.standard_names's compound tech-fuel synonyms.
    storage_gen_params["tech"] = standardize_tech(storage_gen_params["tech"], storage_gen_params["fuel"])
    # some regions' storage plants carry the same tech=Water/fuel=Hydro pair ordinary
    # reservoir/ROR hydro *generators* use (e.g. South Africa's Drakensberg/Ingula/Palmiet/
    # Steenbras pumped-storage schemes, named "...PH_HY_ZAF") -- standardize_tech can't tell
    # these apart from real hydro generation on the pair alone, so it resolves to "hydro"
    # (the correct call in a generator context). But every row here came from
    # pStorageDataInput.csv's plant list to begin with -- reservoir/ROR hydro never appears
    # there, so "hydro" in this storage-only context can only mean pumped-hydro storage.
    storage_gen_params.loc[storage_gen_params["tech"] == "hydro", "tech"] = "PHS"
    zone_tech = storage_gen_params[["generator", "zone", "tech"]].drop_duplicates().rename(
        columns={"generator": "storage"}
    )
    storage_constants = storage_constants.merge(zone_tech, on="storage", how="left")

    # drop storage plants with no zone (no matching pGenDataInputCustom row at all) or a zone
    # not in zones.csv -- storage_costs is built from storage_constants below, so filtering
    # here keeps both files consistent automatically.
    bad_zone = ~storage_constants["zone"].isin(valid_zones)
    if bad_zone.any():
        warnings.warn(
            f"Excluding storage plant(s) with no zone, or a zone not in zones.csv, in "
            f"{data_path!r}: {sorted(storage_constants.loc[bad_zone, 'storage'])}"
        )
        storage_constants = storage_constants[~bad_zone]

    # fill missing lifetimes (some regions have no "Life" value at all for any storage
    # plant) with a default by tech.
    default_lifetime = {
        "battery storage": 15,
        "PHS": 50,
    }
    missing_life = storage_constants["life"].isna()
    storage_constants.loc[missing_life, "life"] = storage_constants.loc[missing_life, "tech"].map(default_lifetime)

    # fill in the remaining GAMS parameters (same names generator_constants pulls)
    gams_param_list = ["Capacity", "RetrYr", "StYr", "UnitSize", "BuildLimitperYear", "MaxTotalBuild"]
    for param in gams_param_list:
        param_values = storage_gen_params[storage_gen_params["parameters"] == param].set_index("generator")["value"]
        storage_constants[param] = storage_constants["storage"].map(param_values)

    # fall back to MaxTotalBuild (a real, per-unit lifetime build ceiling) wherever BOTH
    # BuildLimitperYear and Capacity are missing -- same reasoning and same restriction as
    # generator_params.py (MaxTotalBuild has the same "duplicates existing Capacity" artifact
    # otherwise)
    missing_bly_and_capacity = storage_constants["BuildLimitperYear"].isna() & storage_constants["Capacity"].isna()
    storage_constants.loc[missing_bly_and_capacity, "BuildLimitperYear"] = (
        storage_constants.loc[missing_bly_and_capacity, "MaxTotalBuild"]
    )
    storage_constants = storage_constants.drop(columns=["MaxTotalBuild"])

    # build option if BuildLimitperYear (or its MaxTotalBuild fallback) is given and non-zero -
    # same signal generator_constants uses (a zero build limit means "not buildable at all",
    # same as no value given)
    storage_constants["expandable"] = (
        storage_constants["BuildLimitperYear"].notna() & (storage_constants["BuildLimitperYear"] != 0)
    )
    # unit_size (modular build increment) is only meaningful for build options -- an existing
    # unit isn't built in discrete increments, it just has its own fixed capacity (same as
    # generator_params.py)
    storage_constants.loc[~storage_constants["expandable"], "UnitSize"] = pd.NA
    # rename to seer-schema column names, matching generator_constants conventions
    storage_constants.rename(columns={
        "Capacity": "capacity_mw",
        "RetrYr": "retirement_year",
        "StYr": "start_year",
        "UnitSize": "unit_size_mwh",
        "BuildLimitperYear": "annual_build_limit_mwh",
    }, inplace=True)
    # no RetrYr value found means "no known retirement" -- default to a sentinel year well
    # beyond any realistic modeled horizon, not NaN (see generator_params.py for why: a NaN
    # retirement_year makes every `year <= retirement_year` check False, silently dropping
    # the unit from every modeled year instead of leaving it active for the full horizon).
    storage_constants["retirement_year"] = (
        storage_constants["retirement_year"].infer_objects(copy=False).fillna(NO_RETIREMENT_YEAR)
    )

    # capacity_mw (power) and capacity_mwh (energy) together give duration_hours -- there's no
    # duration/hours parameter anywhere in the raw data itself (see data/processed_gams/
    # README.md), so a unit missing either one has no real basis for a duration assumption in
    # code. Fail loudly rather than guessing a duration (e.g. "4h for batteries") here -- that
    # kind of physical assumption belongs in a documented, per-unit manual fix in
    # data/processed_gams/<region>/ (see e.g. Eswatini's/Timor Leste's Battery_Generic fixes),
    # not silently baked into the pipeline.
    missing_capacity_mw = storage_constants["capacity_mw"].isna() & storage_constants["capacity_mwh"].notna()
    if missing_capacity_mw.any():
        raise ValueError(
            f"{data_path!r}: storage unit(s) have CapacityMWh but no Capacity (MW), so "
            f"duration_hours can't be computed: "
            f"{sorted(storage_constants.loc[missing_capacity_mw, 'storage'])}. "
            "Add a Capacity value for these in pGenDataInputCustom.csv (with a documented "
            "duration assumption in data/processed_gams/README.md) rather than guessing one "
            "here."
        )

    storage_constants["duration_hours"] = storage_constants["capacity_mwh"] / storage_constants["capacity_mw"]

    # capacity_mw is only meaningful for existing (non-expandable) storage -- for build options
    # it's whatever raw "Capacity" value happened to be given (often a placeholder ceiling, same
    # pattern as generator_constants' capacity), and it's never read by either model backend for
    # expandable storage (only annual_build_limit_mwh is). Null it out now, after it's already
    # been used above to derive duration_hours (a real, meaningful property for every storage
    # unit, expandable or not) -- see seer/schema/README.md: "ignored/NaN for build options".
    storage_constants.loc[storage_constants["expandable"], "capacity_mw"] = pd.NA

    # reorder for readability
    storage_constants = storage_constants.sort_values(["expandable", "tech"]).reset_index(drop=True)

    """Setup storage costs.
    Use storage constants to fill in a single dataframe of storage costs.
    """
    rows = []

    for _, storage in storage_constants.iterrows():

        row_capex = {"storage": storage.storage, "cost_type": "capex"}
        for y in years:
            row_capex[y] = storage.capex_mwh
        rows.append(row_capex)

        # marginal cost -- always write a row (0 fallback), same pattern as
        # generator_params.py. Storage doesn't burn fuel, and the raw data's own variable-cost
        # param (VOM) is too sparse to build a real per-region value from (present for only a
        # handful of plants in a couple of regions, absent everywhere else), so 0 is the
        # honest default rather than a fabricated non-zero figure.
        row_mc = {"storage": storage.storage, "cost_type": "marginal_cost"}
        for y in years:
            row_mc[y] = 0
        rows.append(row_mc)

    storage_costs = pd.DataFrame(rows, columns=["storage", "cost_type"] + years)

    # reorder to match schema
    storage_constants = storage_constants[[
        "storage", "zone", "tech", "duration_hours", "capacity_mw",
        "efficiency", "life", "start_year", "retirement_year",
        "unit_size_mwh", "annual_build_limit_mwh", "expandable",
    ]]

    return storage_constants, storage_costs
