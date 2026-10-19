"""Build generator_constants.csv and generator_costs.csv from EPM input data."""
import uuid
import warnings
from pathlib import Path

import pandas as pd

from .standard_names import standardize_tech, standardize_fuel, default_fuel_for_tech, coarsen_cost_tech
from .data_checks import check_generators, check_capex_trajectory, check_profile_time_coverage
from .re_potentials import build_re_potentials
from .utils import init_df, pivot_wide, pivot_years, align

# sentinel for "no known retirement year" in the raw data -- converted to NEVER_RETIRES_LIFE
NO_RETIREMENT_YEAR = 9999
# the schema has start_year + life, never blank (see seer/schema/README.md). Fill-ins:
DEFAULT_START_YEAR = 2010   # no start year: online / buildable before any modeled horizon
NEVER_RETIRES_LIFE = 100    # existing unit with no retirement year: outlives any horizon
DEFAULT_BUILD_LIFE = 30     # build option with no life (needed to annualize capex)

# generator_constants.csv column order
GENERATOR_COLUMNS = [
    "generator", "zone", "tech", "fuel", "capacity", "isre", "expandable", "unit_size",
    "annual_build_limit", "total_build_limit", "ramp_up_rate", "ramp_down_rate",
    "start_year", "life", "profile_id",
]


def to_start_year_and_life(df, name_col, source=""):
    """Replace retirement_year with start_year + life, both always filled, so a unit is active
    in years start_year .. start_year + life - 1. Existing units: life = retirement_year -
    start_year + 1 (retirement_year is the last year online); no retirement year means
    NEVER_RETIRES_LIFE. Build options keep their own life (it's per build), DEFAULT_BUILD_LIFE
    if missing. A missing start_year is DEFAULT_START_YEAR. Every fill-in is warned about."""
    df = df.copy()
    existing = ~df["expandable"].astype(bool)
    start = df["start_year"].astype(float)
    retire = df["retirement_year"].astype(float)
    life = df["life"].astype(float)

    no_start = start.isna()
    if no_start.any():
        warnings.warn(f"{source}: no start_year, set to {DEFAULT_START_YEAR}: {sorted(df.loc[no_start, name_col])}")
    start = start.fillna(DEFAULT_START_YEAR)

    never = existing & (retire.isna() | (retire >= NO_RETIREMENT_YEAR))
    life[existing] = retire[existing] - start[existing] + 1
    life[never] = NEVER_RETIRES_LIFE
    bad = existing & ~never & (life < 1)
    if bad.any():
        # retires before it starts -- never online; keep it a 1-year window so it stays valid
        warnings.warn(f"{source}: retirement_year before start_year, life set to 1: {sorted(df.loc[bad, name_col])}")
        life[bad] = 1

    no_life = ~existing & life.isna()
    if no_life.any():
        warnings.warn(f"{source}: build option(s) with no life, set to {DEFAULT_BUILD_LIFE}: {sorted(df.loc[no_life, name_col])}")
    life[no_life] = DEFAULT_BUILD_LIFE

    df["start_year"], df["life"] = start.round().astype("Int64"), life.round().astype("Int64")
    return df.drop(columns="retirement_year")


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
    generator_cost = build_costs(
        generator_constants, default_gen_capex, custom_gen_capex, fuel_prices, model_years, input_dir
    )
    # capex/heat_rate were only needed to build generator_cost and can now be dropped
    generator_constants = generator_constants.drop(columns=["capex", "heat_rate"])

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

    # some regions' renewable tech is finer-grained than any real PyPSA technology (resource-
    # quality tiers like Georgia's Wind1/2/3) -- that granularity had to survive the matching
    # step just above (it's what keeps their distinct profiles from colliding into one), but
    # once profile_id is assigned, nothing downstream reads tech for profile lookup any more,
    # so it's now safe to coarsen generator_constants.tech to its canonical costing name.
    # renewable_profiles.tech is untouched -- it stays at its original granularity (metadata).
    generator_constants["tech"] = coarsen_cost_tech(generator_constants["tech"])

    # "hydrogen_CCGT" isn't a real standalone PyPSA technology (no separate cost data), and
    # it's never just CCGT either -- the raw EPM data gives combined-cycle and open-cycle
    # hydrogen units the exact same tech=Hydrogen/fuel=Hydrogen (e.g. Mexico's "Generic CCGT
    # H2" vs "Generic OCGT H2", Turkiye's "..._CCGT_H2" vs "..._OCGT_H2") -- only the
    # generator's own name says which cycle it is. Split on that name pattern and approximate
    # each with its matching gas-turbine cost basis; anything not explicitly named OCGT
    # defaults to CCGT (the more common case across regions).
    is_h2 = generator_constants["tech"] == "hydrogen_CCGT"
    is_ocgt_named = generator_constants["generator"].str.contains("OCGT", case=False, na=False)
    generator_constants.loc[is_h2, "tech"] = "CCGT"
    generator_constants.loc[is_h2 & is_ocgt_named, "tech"] = "OCGT"

    # import is a placeholder for externally-sourced power, not a real buildable generation
    # asset -- never a build option, regardless of what BuildLimitperYear/Capex happen to say
    # for it (several regions pair an existing import link with a real annual uprate allowance
    # but no cost data at all -- see the drop below for exactly that case).
    is_import = generator_constants["tech"] == "import"
    generator_constants.loc[is_import, "expandable"] = False
    generator_constants.loc[is_import, "annual_build_limit"] = pd.NA
    generator_constants.loc[is_import, "total_build_limit"] = pd.NA

    # forcing expandable=False above can leave a pure candidate import/interconnection option
    # (one that never had its own existing capacity -- e.g. a "<name>_expansion" split-off, or
    # Eswatini's SAR/EDM/SAPP_Connection_New) with neither capacity nor annual_build_limit --
    # completely inert, contributing nothing. Drop those; an import row with a real existing
    # capacity (even 0, a currently-unused interconnector) is still meaningful and stays.
    inert_import = is_import & generator_constants["capacity"].isna()
    if inert_import.any():
        dropped_import = set(generator_constants.loc[inert_import, "generator"])
        warnings.warn(
            f"Dropping import-tech generator(s) with no existing capacity (pure candidates, "
            f"which can't be build options either): {sorted(dropped_import)}"
        )
        generator_constants = generator_constants[~inert_import].reset_index(drop=True)
        generator_cost = generator_cost[~generator_cost["generator"].isin(dropped_import)].reset_index(drop=True)

    # an expandable generator with literally no cost data anywhere -- no trajectory (even after
    # the flat-capex fallback in build_costs above) and no base capex scalar at all -- has
    # nothing to fall back to and no sensible interpretation as a buildable option. Drop it
    # entirely rather than leave a free-to-build (PyPSA silently treats NaN capital_cost as 0)
    # or crashing (pyomo's Param domain rejects NaN outright) option in the schema.
    capex_by_gen = generator_cost[generator_cost["cost_type"] == "capital"].set_index("generator")
    year_cols = [c for c in capex_by_gen.columns if c != "cost_type"]
    unpriced_names = set(capex_by_gen.index[capex_by_gen[year_cols].isna().all(axis=1)])
    truly_unpriced = (
        generator_constants["expandable"]
        & ~is_import
        & generator_constants["generator"].isin(unpriced_names)
    )
    if truly_unpriced.any():
        dropped = set(generator_constants.loc[truly_unpriced, "generator"])
        warnings.warn(
            f"Dropping expandable generator(s) with no cost data anywhere (no trajectory, no "
            f"base capex): {sorted(dropped)}"
        )
        generator_constants = generator_constants[~truly_unpriced].reset_index(drop=True)
        generator_cost = generator_cost[~generator_cost["generator"].isin(dropped)].reset_index(drop=True)

    # an existing (non-expandable) generator with no capacity value at all -- e.g. Rwanda's
    # Beloluane_TH_MOZ/Aggreko_IPP_ICE_MWI: raw Status=Existing, but no Capacity,
    # BuildLimitperYear, or MaxTotalBuild anywhere. No number anywhere to recover -- same
    # "nothing to fall back to" situation as the unpriced build options above, just on the
    # existing side.
    empty_existing = ~generator_constants["expandable"] & generator_constants["capacity"].isna()
    if empty_existing.any():
        dropped_empty = set(generator_constants.loc[empty_existing, "generator"])
        warnings.warn(
            f"Dropping existing generator(s) with no capacity value anywhere in the raw data: "
            f"{sorted(dropped_empty)}"
        )
        generator_constants = generator_constants[~empty_existing].reset_index(drop=True)
        generator_cost = generator_cost[~generator_cost["generator"].isin(dropped_empty)].reset_index(drop=True)

    # reorder for readability
    generator_constants = generator_constants.sort_values(["expandable", "fuel", "tech"]).reset_index(drop=True)

    generator_constants = to_start_year_and_life(generator_constants, "generator", input_dir)
    generator_constants = generator_constants[GENERATOR_COLUMNS]

    # generator_costs.csv rows should follow the same generator order as generator_constants.csv
    generator_cost = align(generator_cost, generator_constants, "generator")

    return generator_constants, generator_cost, renewable_profiles


def _split_existing_and_expansion(gc):
    """Some regions' raw EPM data gives a single generator BOTH a real existing Capacity AND a
    real nonzero BuildLimitperYear on the same row -- e.g. Nigeria's Egbin (1320 MW), Jebba
    (578.4 MW), Kainji (760 MW), Shiroro (600 MW): all real plants with real nameplate
    capacities that also carry a small annual rehab/uprate build limit. Checked across every
    region: 2552 generators do this. The seer schema's `expandable` flag is a strict either/or
    (existing fixed capacity, or a pure build option) -- it can't represent "existing, and also
    separately expandable" on one row. Split any such generator into two: the original name
    keeps its real existing capacity (its BuildLimitperYear/MaxTotalBuild move to the new row,
    since that allowance isn't part of the fixed existing unit); a new "<generator>_expansion"
    row keeps the annual build limit and shares the same zone/tech/fuel/cost basis as the
    original (same underlying technology), with Capacity nulled so it resolves to a pure build
    option once `expandable` is computed below. The new row records its parent's name in
    `split_from`, so it also resolves to the parent's generator-specific renewable profile (and
    so the same isre) in match_re_profiles, rather than looking up its own "_expansion" name."""
    both = gc["Capacity"].notna() & gc["BuildLimitperYear"].notna() & (gc["BuildLimitperYear"] != 0)
    if not both.any():
        return gc

    existing = gc.copy()
    existing.loc[both, ["BuildLimitperYear", "MaxTotalBuild"]] = pd.NA

    expansion = gc[both].copy()
    expansion["generator"] = expansion["generator"] + "_expansion"
    expansion["Capacity"] = pd.NA
    expansion["split_from"] = gc.loc[both, "generator"]

    collisions = set(expansion["generator"]) & set(existing["generator"])
    if collisions:
        raise ValueError(
            f"Generated expansion-option generator name(s) collide with an existing generator: "
            f"{sorted(collisions)} -- pick a different naming scheme in _split_existing_and_expansion."
        )
    return pd.concat([existing, expansion], ignore_index=True)


def _derive_flags(gc, data_path):
    """Derive expandable/isre/retirement_year/life, and null out columns only meaningful
    for one of expandable/existing."""
    gc = _split_existing_and_expansion(gc)

    # expandable: a real nonzero annual or total build limit with no existing capacity (see
    # data/processed_gams/README.md re: this vs. the Status-based fixes), a real nonzero annual
    # build limit alongside existing capacity (split off above), or no capacity but real cost
    # data to build from
    has_annual = gc["BuildLimitperYear"].notna() & (gc["BuildLimitperYear"] != 0)
    has_total = gc["MaxTotalBuild"].notna() & (gc["MaxTotalBuild"] != 0)
    gc["expandable"] = (
        has_annual
        | (gc["Capacity"].isna() & has_total)
        | (gc["Capacity"].isna() & gc["Capex"].notna())
    )
    # annual_build_limit / total_build_limit are only kept where explicitly given -- absent
    # means unconstrained (see seer/schema/README.md; the model backends read blank as inf).
    # EPM only applies its build-limit constraints where the value is nonzero ($-conditioned),
    # so 0 means unconstrained too, not "never build". Only meaningful for build options.
    gc.loc[~has_annual | ~gc["expandable"], "BuildLimitperYear"] = pd.NA
    gc.loc[~has_total | ~gc["expandable"], "MaxTotalBuild"] = pd.NA
    gc.loc[gc["expandable"], "Capacity"] = pd.NA   # only existing generators have a capacity
    gc.loc[~gc["expandable"], "UnitSize"] = pd.NA  # only expandable generators have a unit_size
    # ramp rates are a fraction of capacity per hour, both positive (the down rate is a magnitude,
    # not a signed change). EPM applies them the same way (per time step, x capacity). Missing
    # means fully flexible (1). Some regions' data has values above 1 (e.g. Bosnia: 5.0 for
    # hydro/gas, up to 3.0 for coal) -- a fraction above 1 can't bind (a unit can already go
    # from zero to full in one hour), so cap at 1 rather than carry a meaningless value. A 0
    # (e.g. Burkina Faso's hydro, imports and existing PV) is EPM's blank, not "can never change
    # output" -- also fully flexible.
    for col in ["RampUpRate", "RampDnRate"]:
        gc[col] = gc[col].abs().replace(0, pd.NA).fillna(1).clip(upper=1)

    gc.rename(columns={
        "BuildLimitperYear": "annual_build_limit",
        "MaxTotalBuild": "total_build_limit",
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

    # Life / retirement_year are turned into the schema's start_year + life (always filled) by
    # to_start_year_and_life, once the columns are final

    gc.columns = gc.columns.str.lower()
    return gc


def build_costs(generator_constants, default_gen_capex, custom_gen_capex, fuel_prices, model_years, input_dir):
    """generator_costs.csv: two rows per generator (capex and marginal cost), years as
    columns. capex = capex trajectory x capex scalar. marginal_cost = heat_rate x fuel price
    (+ VOM where the region sets includeVOM)."""

    ztf = ["zone", "tech", "fuel"]
    gc = generator_constants.set_index("generator")
    # EPM's Capex is total (overnight) capital cost in $M/MW -- seer's capex is $/MW, incurred
    # once in the build year (see seer/schema/README.md)
    gc["capex"] = gc["capex"] * 1e6

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

    # a region can have real trajectory data covering every modeled year overall, yet still be
    # missing it for one particular generator -- check_capex_trajectory only validates year
    # coverage of whichever source has any data, not that every generator is represented in it
    # (no custom trajectory row of its own, and its own (zone,tech,fuel) absent from the default
    # table too). That slips through silently as NaN. A build option's capex must never be NaN
    # (pyomo's Param domain rejects it outright at model-build time; PyPSA silently treats a NaN
    # capital_cost as free-to-build, which is worse). Fall back to flat (no decline) for just
    # these generators, the same treatment already used when a whole region has no trajectory
    # data at all -- but only where there's a real base capex to hold flat; a generator with
    # neither a trajectory nor a base capex has no cost signal to fall back to at all.
    still_missing = gc["expandable"] & capex_cost.isna().all(axis=1) & gc["capex"].notna()
    if still_missing.any():
        warnings.warn(
            f"No capex trajectory found for expandable generator(s) with a real base capex -- "
            f"holding capex flat (no cost decline) for: {sorted(gc.index[still_missing])}"
        )
        capex_cost.loc[still_missing] = pd.DataFrame(
            {y: gc.loc[still_missing, "capex"] for y in model_years}
        )

    # existing units have no build cost -- explicit 0, never blank (see seer/schema/README.md).
    # A build option with no capex is left NaN and fails validation rather than becoming free.
    existing = ~gc["expandable"].astype(bool)
    capex_cost.loc[existing] = capex_cost.loc[existing].fillna(0.0)

    """build marginal cost"""
    # match fuel/zone case/whitespace-insensitively -- raw spellings disagree even within the
    # SAME workbook (e.g. Botswana's Generator Data calls it "Diesel"/"FuelOil", but its own
    # Fuel Prices sheet calls the exact same fuels "diesel"/"Fueloil"); an exact-string merge
    # silently drops a real, present price over nothing but a casing difference.
    #
    # PRICE_FUEL_ALIASES folds a few known same-commodity spelling variants that differ
    # between a region's own Generator Data and its own Fuel Prices sheet -- "HFO" (Heavy
    # Fuel Oil) and "FuelOil" are the same commodity under different abbreviations (one
    # region's generators say "HFO" while its price sheet says "FuelOil", another region has
    # it the other way around); WAPP's price sheet has a literal typo, "Biomas" (missing the
    # final "s"). Checked every region's own Fuel Prices sheet first: none of them price both
    # spellings as distinct entries, so folding them here can never silently prefer the wrong
    # one. This is intentionally narrower than pypsa_fuel_names.csv's "oil"/"solid biomass"
    # buckets, which also group in e.g. Diesel -- a genuinely different, separately-priced
    # commodity in every region that has both (Malawi's Diesel is ~20% more expensive than its
    # FuelOil) and must stay its own distinct price match.
    PRICE_FUEL_ALIASES = {"hfo": "fueloil", "biomas": "biomass"}

    def _norm(s):
        return s.astype(str).str.strip().str.lower().replace(PRICE_FUEL_ALIASES)

    gc_reset = gc.reset_index()
    # "CustomFuel" is a GAMS placeholder meaning "no real fuel commodity -- this generator is
    # priced some other way" (e.g. every nuclear plant in this dataset: Malawi's Uranium_NU_ZAF,
    # Turkiye's AKKUYU_1-4). It never matches any real price row, but the generator's own raw
    # TECH value (still raw/un-standardized at this point, same as fuel) usually already IS a
    # real commodity name in disguise -- Turkiye's price sheet literally has a "Nuclear" entry,
    # matching AKKUYU's own raw tech="Nuclear" directly once case-folded. Use tech as the
    # price-match key for just these rows.
    fuel_for_price = gc_reset["fuel"].where(gc_reset["fuel"].str.lower() != "customfuel", gc_reset["tech"])
    fp_norm = fuel_prices.assign(fuel=_norm(fuel_prices["fuel"]), zone=_norm(fuel_prices["zone"]))

    # a region can price imported gas only as LNG, never a bare "Gas" commodity at all (e.g.
    # Morocco: Coal/Diesel/FuelOil/Hydrogen/LNG/LNGCCS, no plain "Gas" row anywhere) --
    # unlike HFO/FuelOil, LNG and Gas are genuinely, separately priced wherever a region has
    # both (confirmed across every region checked), so this substitution only fires when
    # "gas" has ZERO price rows in this region at all -- there's no real "Gas" entry to
    # conflict with, and gas is what Morocco's own LNG import price actually is for a plain
    # gas-fueled generator here.
    region_fuels = set(fp_norm["fuel"].unique())
    if "gas" not in region_fuels and "lng" in region_fuels:
        is_gas = fuel_for_price.str.lower() == "gas"
        fuel_for_price = fuel_for_price.where(~is_gas, "lng")

    gc_reset["_fuel_norm"] = _norm(fuel_for_price)
    gc_reset["_zone_norm"] = _norm(gc_reset["zone"])

    exact = gc_reset.merge(
        pivot_years(fp_norm, ["fuel", "zone"], model_years),
        left_on=["_fuel_norm", "_zone_norm"], right_on=["fuel", "zone"], how="left",
        suffixes=("", "_pricezone"),
    ).set_index("generator")[model_years]

    # several regions (Turkiye, Mexico, and the multi-zone regional pools CAPP/EAPP/Malawi/
    # Rwanda/WAPP/Serbia) price fuel at the country level, but model generators at a finer
    # sub-national zone (Turkiye's 9 grid regions, Mexico's CEN/NES/NOR/...) -- an exact
    # (fuel, zone) match never succeeds for any of them. zcmap.csv already carries the zone ->
    # country mapping used to build zones.csv itself; fall back to matching the generator's
    # own country against fuel_prices' own "zone" column wherever the exact zone match comes
    # up empty.
    # some regions' zcmap.csv is a zone-adjacency list rather than a clean zone->country table
    # (e.g. Serbia's lists every bordering zone as a separate "country" row for the same zone)
    # -- keep just the first row per zone, same resolution epm2seer.data_checks.
    # check_zones already applies when building zones.csv's own "country" column, so this
    # fallback can never be more wrong than that column already is.
    zcmap = pd.read_csv(f"{input_dir}zcmap.csv")[["zone", "country"]].drop_duplicates(subset="zone", keep="first")
    zone_to_country = zcmap.set_index("zone")["country"]
    gc_reset["_country_norm"] = _norm(gc_reset["zone"].map(zone_to_country))
    fp_country_wide = pivot_years(fp_norm, ["fuel", "zone"], model_years).reset_index()
    by_country = gc_reset.merge(
        fp_country_wide, left_on=["_fuel_norm", "_country_norm"], right_on=["fuel", "zone"], how="left",
        suffixes=("", "_pricezone2"),
    ).set_index("generator")[model_years]

    # some regions genuinely never price a given fuel for one particular zone even though they
    # do for others in the SAME region (e.g. EAPP's Burundi only prices Diesel/Peat/HFO/Coal --
    # never Gas/Biomass/Waste -- while other EAPP zones do). There's no real zone-specific
    # number to recover here, so fall back to this fuel's average price across every zone that
    # DOES have one, region-wide, rather than leave it unpriced -- a reasonable regional proxy,
    # not a guess from nothing.
    fp_region_avg = fp_norm.groupby(["fuel", "year"])["value"].mean().reset_index()
    fp_region_wide = pivot_years(fp_region_avg, ["fuel"], model_years)
    by_region = gc_reset.merge(
        fp_region_wide, left_on="_fuel_norm", right_index=True, how="left"
    ).set_index("generator")[model_years]

    fuel_price_traj = exact.where(exact.notna(), by_country)
    fuel_price_traj = fuel_price_traj.where(fuel_price_traj.notna(), by_region)
    # a fuel's own raw price data can genuinely start (or end) partway through the modeled
    # horizon -- e.g. Iraq's Gas price rows exist for 2018-2020 but are blank, with real values
    # only from 2021 on; Egypt's/Lao's/Liberia's price data simply doesn't have rows for their
    # region's earliest/latest modeled years at all. Hold flat at the nearest year with a real
    # price rather than leave those years priced at NaN -- same reasoning as the capex fallback
    # above. model_years is already ascending, so this only reaches across a real gap, never
    # invents a trend. A fuel with NO real price anywhere (not even one year) has nothing to
    # extrapolate from and is correctly left NaN -- caught by SeerInstance validation.
    fuel_price_traj = fuel_price_traj.ffill(axis=1).bfill(axis=1)

    # geothermal, water (hydro/ror), wind, and solar are never priced in ANY region's Fuel
    # Prices sheet -- by GAMS modeling convention they're treated as free, naturally-occurring
    # energy, not a purchased commodity like coal/gas/oil/biomass/uranium. Their generators
    # still often carry a real (if arbitrary -- e.g. Botswana's wind AND solar both share the
    # exact placeholder HeatRate=9.42) heat_rate, so they don't hit the heat_rate==0 fallback
    # below and would otherwise end up with a NaN marginal cost despite having nothing wrong
    # with their data. Treat these fuels' genuinely-unpriced rows as a real $0, not a gap.
    # hydrogen and municipal waste are the same situation by explicit user call: hydrogen is
    # never priced anywhere in this dataset (Mexico/Serbia/Turkiye/Poland's hydrogen-fired
    # units all have a real heat_rate but no fuel price), and municipal solid waste combustion
    # is conventionally treated as a free/near-zero-cost disposal fuel, distinct from purchased
    # biomass (Poland's Incineration_plants/Konin_ZPTOK/Olsztyn_3's raw fuel1 is literally
    # "MunicipalWaste"; EAPP's Burundi plant is bare "Waste" -- neither spelling folds to
    # plain "Biomass" until standardize_fuel runs, after this). Checked every region's own
    # Fuel Prices sheet: none of them price "Waste" at all, so this can't silently override a
    # real number either.
    # gc["fuel"] here is still the RAW, pre-standardization spelling (standardize_fuel runs
    # later in build_generator_params) -- e.g. "Geothermal"/"Water"/"Wind"/"PV"/"Solar"/
    # "Hydrogen"/"MunicipalWaste"/"Waste", not the canonical lowercase names used elsewhere.
    # Match case-insensitively.
    # "import" generators (e.g. Botswana's Moz_imports, HeatRate=9.42 -- the exact same
    # placeholder value Botswana gives its wind/solar) are a mix: some have a real negotiated
    # import price (Namibia's ESKOMimport/ZESCOImport/ZPCImport all do, and keep it -- this
    # only fires where NOTHING was found at all, never overriding a real price), others have
    # none anywhere to find. Zero is the right default for the latter: an import link with no
    # known tariff isn't free in reality, but there's no commodity to price it against either.
    free_fuels = gc["fuel"].str.lower().isin(
        ["geothermal", "water", "wind", "solar", "pv", "hydrogen", "municipalwaste", "waste", "import"]
    )
    fuel_price_traj = fuel_price_traj.where(~(free_fuels & fuel_price_traj.isna().all(axis=1)), 0.0)

    marginal_cost = fuel_price_traj.mul(gc["heat_rate"], axis=0)
    # marginal cost is 0 for generators with no heat_rate
    marginal_cost = marginal_cost.where(gc["heat_rate"].fillna(0) != 0, 0.0)

    # VOM ($/MWh) on top of fuel, only for regions that opt in (pSettings.csv includeVOM=1) --
    # e.g. Burkina Faso, whose imports and PPA solar are priced only through VOM
    settings = pd.read_csv(f"{input_dir}pSettings.csv")
    include_vom = settings.loc[settings["parameters"] == "includeVOM", "value"]
    if not include_vom.empty and float(include_vom.iloc[0]):
        custom = pd.read_csv(f"{input_dir}pGenDataInputCustom.csv")
        vom = custom[custom["parameters"] == "VOM"].set_index("generator")["value"].astype(float)
        marginal_cost = marginal_cost.add(vom.reindex(gc.index).fillna(0.0), axis=0)

    # marginal cost given directly ($/MWh tariff), replacing heat rate x fuel price -- e.g.
    # Botswana's Moz_imports (see scripts/convert_epm_open.py's marginal_cost_from_tariff)
    override_path = f"{input_dir}pMarginalCostOverride.csv"
    if Path(override_path).exists():
        override = pivot_years(pd.read_csv(override_path), ["generator"], model_years)
        override = override.reindex(gc.index).ffill(axis=1).bfill(axis=1)
        marginal_cost = override.where(override.notna(), marginal_cost)

    """concatenate capex and marginal cost"""
    # row order is finalized later, to match generator_constants.csv
    generator_cost = pd.concat([
        capex_cost.assign(cost_type="capital"),
        marginal_cost.assign(cost_type="marginal"),
    ]).reset_index()
    generator_cost = generator_cost[["generator", "cost_type"] + model_years]
    return generator_cost


def match_re_profiles(gc, profiles_raw, times):
    """Resolve each generator to at most one renewable profile, by priority: generator-
    specific, then zone+tech generic, then tech-only fallback (see seer/schema/README.md).
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
    # zone+tech generic, then tech-only fallback, keeping the first match found. A
    # "<name>_expansion" split-off (see _split_existing_and_expansion) is the same plant, so
    # its generator-specific lookup uses its parent's name.
    gc = gc.copy()
    if "split_from" in gc:
        gc["generator"], own_name = gc["split_from"].fillna(gc["generator"]), gc["generator"]
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
    if "split_from" in gc:
        gc["generator"] = own_name
        gc = gc.drop(columns="split_from")

    claimed_but_unmatched = gc["isre"] & profile_id.isna()
    for _, gen in gc[claimed_but_unmatched].iterrows():
        warnings.warn(
            f"Generator {gen.generator!r} (zone={gen.zone}, tech={gen.tech}) is flagged "
            "renewable but has no matching profile -- marking as non-renewable."
        )

    gc["profile_id"] = profile_id
    gc["isre"] = profile_id.notna()
    renewable_profiles = profiles_raw[["profile_id", "zone", "tech", "generator", "time", "value"]]
    return gc, renewable_profiles


