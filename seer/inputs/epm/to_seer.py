"""Convert csvs specifying EPM input data into the seer schema."""

import warnings
from pathlib import Path
import pandas as pd

from .utils import convert_season_day_hour
from .generator_params import build_generator_params
from .re_potentials import build_re_potentials
from .storage_params import build_storage_params
from .data_checks import *


def epm_to_seer(input_path, output_path):
    """Run all conversion steps to turn EPM model specification to seer schema
    seer schema files are written to the output path.
    returns a dict of all the dataframes, also written to csvs.
    """

    # set input and output directories
    input_dir = f"{Path(input_path)}/"
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    """modelled years"""
    years = pd.read_csv(f"{input_dir}y.csv")[["year"]]

    """demand forecast & profile"""
    demand_forecast = check_demand_forecast(pd.read_csv(f"{input_dir}pDemandForecast.csv"))
    demand_profile = pd.read_csv(f"{input_dir}pDemandProfile.csv")
    demand_profile = convert_season_day_hour(demand_profile)[["zone", "time", "value"]]

    """zones"""
    zones = check_zones(pd.read_csv(f"{input_dir}zcmap.csv"))
    # default name for a zone is the zone itself
    zones["name"] = zones["zone"]

    """network"""
    network_path = Path(f"{input_dir}pTransferLimit.csv")
    if network_path.exists():
        network = check_network(pd.read_csv(network_path), zones["zone"])
        # original data has time varying capacities, but seer has static -> use average
        network = network.groupby(["from", "to"])["value"].mean().reset_index().rename(columns={"value": "capacity"})
    else:
        # if there is no network, create a placeholder empty network
        network = pd.DataFrame(columns=["from", "to", "capacity"])
    
    """settings"""
    settings = check_settings(pd.read_csv(f"{input_dir}pSettings.csv"))



    demand_forecast = _drop_unknown_zones(demand_forecast, valid_zones, "demand_forecast.csv")
    demand_profile = _drop_unknown_zones(demand_profile, valid_zones, "demand_profile.csv")

    generator_constants, generator_costs = build_generator_params(input_dir, model_years=years["year"].tolist())
    generator_constants, generator_costs = _drop_unknown_zones(
        generator_constants, valid_zones, "generator_constants.csv", generator_costs, "generator"
    )
    renewable_profiles_raw = build_re_potentials(input_dir)
    # times.csv: the canonical dispatch time grid, taken directly from demand_profile's own
    # `time` values (EPM data is often already a handful of representative days per month, not
    # a full year -- see seer/inputs/README.md's times.csv entry)
    times = pd.DataFrame({"time": sorted(demand_profile["time"].unique())})
    generator_constants, renewable_profiles = _match_renewable_profiles(
        generator_constants, renewable_profiles_raw, times["time"].tolist()
    )
    # storage_costs.csv must carry a column for every modeled year (years.csv) -- build.py's
    # per-vintage capex lookup indexes storage_costs by those exact values (see
    # storage_params.py's build_storage_params docstring)
    storage_constants, storage_costs = build_storage_params(input_dir, years=years["year"].tolist())
    storage_constants, storage_costs = _drop_unknown_zones(
        storage_constants, valid_zones, "storage_constants.csv", storage_costs, "storage"
    )

    outputs = {
        "generator_constants": generator_constants,
        "generator_costs": generator_costs,
        "renewable_profiles": renewable_profiles,
        "storage_constants": storage_constants,
        "storage_costs": storage_costs,
        "years": years,
        "times": times,
        "zones": zones,
        "network": network,
        "settings": settings,
        "demand_forecast": demand_forecast,
        "demand_profile": demand_profile,
    }
    # save seer schema csv files to output directory
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    return outputs


"""Helper functions"""
def _drop_incomplete_profiles(df, key_cols, times, label):
    """Drop (with a warning) any key whose profile doesn't cover every one of `times` -- e.g.
    a raw source that only has data for a subset of representative days/hours. A partial
    profile can't safely drive dispatch for the times it's missing, and is far better caught
    here (with a clear warning naming the culprit) than left to crash deep inside model
    construction (a bare "some modeled time has no value" from seer.models.time_utils)."""
    if df.empty:
        return df
    n_times = len(times)
    counts = df.groupby(key_cols).size().reset_index(name="n")
    incomplete = counts[counts["n"] != n_times]
    if incomplete.empty:
        return df
    warnings.warn(
        f"{label} profile(s) {list(incomplete[key_cols].itertuples(index=False, name=None))} "
        f"cover only some of the {n_times} modeled times (times.csv), not all -- dropping "
        "(treating as unmatched) rather than risk an incomplete dispatch profile."
    )
    return df.merge(incomplete[key_cols].assign(_incomplete=True), on=key_cols, how="left").pipe(
        lambda d: d[d["_incomplete"].isna()].drop(columns="_incomplete")
    )


def _drop_unknown_zones(df, valid_zones, label, costs_df=None, id_col=None):
    """Drop rows referencing a zone not in zones.csv -- with a warning naming what was
    dropped, since no Bus is ever created for such a zone (e.g. cross-region contamination
    in the raw gams csvs: Mexico's pDemandProfile.csv has real rows for "Mogadishu", CAPP's
    generator data references "Burundi"/"Rwanda"/"DRC_East", none of which are CAPP zones).

    If `costs_df`/`id_col` are given (generator_costs.csv/storage_costs.csv, joined by
    generator/storage name), matching rows are dropped there too so the two files stay
    consistent.
    """
    bad = ~df["zone"].isin(valid_zones)
    if not bad.any():
        return (df, costs_df) if costs_df is not None else df
    bad_zones = sorted(df.loc[bad, "zone"].unique())
    warnings.warn(
        f"{label}: dropping {bad.sum()} row(s) referencing zone(s) not in zones.csv "
        f"(not modeled -- likely cross-region data contamination): {bad_zones}"
    )
    df = df[~bad]
    if costs_df is not None:
        costs_df = costs_df[costs_df[id_col].isin(df[id_col])]
        return df, costs_df
    return df



def _match_renewable_profiles(generator_constants, profiles_raw, times):
    """Keep only renewable_profiles rows that actually match something real, checked in this
    priority order per generator (see seer/inputs/README.md):
    1. generator-specific -- kept only if its identifier is an actual generator name (some
       regions' pVREgenProfile.csv variant has no real "generator" column at all, so its
       identifiers are really sub-zone resource labels with nothing to attach to -- see
       re_potentials.py's docstring).
    2. zone+tech generic -- kept only if some generator actually has that (zone, tech).
    3. tech-only fallback (zone-agnostic) -- kept only if some generator has that tech at
       all. Never synthesized here (e.g. by averaging other zones) -- only present if the
       source data already had a zone-agnostic row, or one was added by hand during manual
       data cleanup.
    Matched profiles are further required to cover every modeled time (see
    _drop_incomplete_profiles) -- a partial match is treated as no match.

    isre is (re)computed here from scratch, straight from profile existence via any of the
    three tiers above -- regardless of what any raw RETech/VRETech flag said, since profile
    existence is the actual ground truth for whether a generator's dispatch should be
    profile-driven. A generator whose raw flag claimed renewable but has no matching profile
    at all gets a warning and isre=False (can't be simulated as a variable renewable with no
    profile to mask its output).

    `times` is the region's full times.csv time list, used to validate profile completeness.

    Returns (generator_constants, renewable_profiles) with isre updated and unmatched profile
    rows dropped.
    """
    real_zone_tech = generator_constants[["zone", "tech"]].drop_duplicates()
    real_generators = set(generator_constants["generator"])
    real_techs = set(generator_constants["tech"])

    has_tech = profiles_raw["tech"].notna()
    has_zone = profiles_raw["zone"].notna()
    zone_tech_matched = profiles_raw[has_tech & has_zone].merge(real_zone_tech, on=["zone", "tech"], how="inner")
    tech_only_matched = profiles_raw[has_tech & ~has_zone]
    tech_only_matched = tech_only_matched[tech_only_matched["tech"].isin(real_techs)]
    specific_matched = profiles_raw[~has_tech & profiles_raw["generator"].isin(real_generators)]

    zone_tech_matched = _drop_incomplete_profiles(zone_tech_matched, ["zone", "tech"], times, "generic")
    tech_only_matched = _drop_incomplete_profiles(tech_only_matched, ["tech"], times, "tech-only fallback")
    specific_matched = _drop_incomplete_profiles(specific_matched, ["generator"], times, "generator-specific")

    has_specific = generator_constants["generator"].isin(set(specific_matched["generator"]))
    matched_zone_tech = set(zip(zone_tech_matched["zone"], zone_tech_matched["tech"]))
    has_zone_tech = generator_constants.apply(lambda g: (g.zone, g.tech) in matched_zone_tech, axis=1)
    has_tech_only = generator_constants["tech"].isin(set(tech_only_matched["tech"]))
    has_any = has_specific | has_zone_tech | has_tech_only

    claimed_but_unmatched = generator_constants["isre"] & ~has_any
    for _, gen in generator_constants[claimed_but_unmatched].iterrows():
        warnings.warn(
            f"Generator {gen.generator!r} (zone={gen.zone}, tech={gen.tech}) is flagged "
            "renewable but has no matching renewable_profiles.csv data -- marking as non-renewable."
        )

    generator_constants = generator_constants.copy()
    generator_constants["isre"] = has_any
    matched = pd.concat([zone_tech_matched, tech_only_matched, specific_matched], ignore_index=True)
    return generator_constants, matched

