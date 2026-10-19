"""Structural/referential validation of a SeerInstance against seer/schema/README.md."""

from __future__ import annotations


class SchemaValidationError(ValueError):
    """Raised by SeerInstance.validate() with every violation found, newline-separated."""


def validate_seer_instance(instance) -> None:
    errors: list[str] = []

    zones = set(instance.zones.get("zone", []))
    years = set(instance.years.get("year", []))
    times = set(instance.times.get("time", []))

    _check_zones(instance, errors)
    _check_years(instance, errors)
    _check_times(instance, errors)
    _check_settings(instance, errors)
    _check_network(instance, zones, errors)
    _check_demand_forecast(instance, zones, years, errors)
    _check_demand_profile(instance, zones, times, years, errors)
    _check_renewable_profiles(instance, times, years, errors)
    _check_generator_constants(instance, zones, errors)
    _check_generator_costs(instance, years, errors)
    _check_storage_constants(instance, zones, errors)
    _check_storage_costs(instance, years, errors)

    if errors:
        raise SchemaValidationError("\n".join(errors))


def _require_columns(df, file_name, required, errors):
    missing = set(required) - set(df.columns)
    if missing:
        errors.append(f"{file_name}.csv: missing required column(s) {sorted(missing)}")
        return False
    return True


def _check_zones(instance, errors):
    # every zone id must be unique
    df = instance.zones
    if not _require_columns(df, "zones", ["zone", "country", "name"], errors):
        return
    if df["zone"].duplicated().any():
        errors.append(f"zones.csv: duplicate zone id(s) {sorted(df.loc[df['zone'].duplicated(), 'zone'].unique())}")


def _check_years(instance, errors):
    # every modeled year must be unique
    df = instance.years
    if not _require_columns(df, "years", ["year"], errors):
        return
    if df["year"].duplicated().any():
        errors.append(f"years.csv: duplicate year(s) {sorted(df.loc[df['year'].duplicated(), 'year'].unique())}")


def _check_times(instance, errors):
    # times.csv needs a time column; an optional weight column must be positive where given
    if not _require_columns(instance.times, "times", ["time"], errors):
        return
    if "weight" in instance.times.columns:
        weight = instance.times["weight"].astype(float)
        bad = weight.notna() & (weight <= 0)
        if bad.any():
            errors.append(f"times.csv: non-positive weight(s) at time(s): {sorted(instance.times.loc[bad, 'time'])}")


def _check_settings(instance, errors):
    # a discount rate (dr) parameter must be present
    df = instance.settings
    if not _require_columns(df, "settings", ["parameters", "value"], errors):
        return
    if not (df["parameters"].astype(str).str.upper() == "DR").any():
        errors.append("settings.csv: missing required 'dr' (discount rate) parameter")


def _check_network(instance, zones, errors):
    # every line's from/to zone must be a real zone id
    df = instance.network
    if df.empty:
        return
    if not _require_columns(df, "network", ["from", "to", "capacity"], errors):
        return
    bad = ~(df["from"].isin(zones) & df["to"].isin(zones))
    if bad.any():
        bad_pairs = sorted(set(df.loc[bad, ["from", "to"]].itertuples(index=False, name=None)))
        errors.append(f"network.csv: line(s) reference zone(s) not in zones.csv: {bad_pairs}")
    # lines are undirected: each zone pair appears once (A->B and B->A would double its capacity)
    pairs = df.apply(lambda row: tuple(sorted((row["from"], row["to"]))), axis=1)
    if pairs.duplicated().any():
        errors.append(f"network.csv: zone pair(s) listed more than once: {sorted(set(pairs[pairs.duplicated()]))}")
    # an expandable line must have a capex (start_year/unit_size/total_build_limit are optional)
    if "expandable" in df.columns:
        expandable = df["expandable"].fillna(False).astype(bool)
        if expandable.any():
            missing = expandable & (df["capital"].isna() if "capital" in df.columns else True)
            if missing.any():
                errors.append(f"network.csv: expandable line(s) missing 'capital': {sorted(set(pairs[missing]))}")


def _check_demand_forecast(instance, zones, years, errors):
    df = instance.demand_forecast
    if not _require_columns(df, "demand_forecast", ["zone", "type", "year", "value"], errors):
        return
    # zone must be a real zone
    bad_zone = ~df["zone"].isin(zones)
    if bad_zone.any():
        errors.append(f"demand_forecast.csv: zone(s) not in zones.csv: {sorted(df.loc[bad_zone, 'zone'].unique())}")
    # type must be peak (energy was dropped from the schema)
    bad_type = df["type"] != "peak"
    if bad_type.any():
        errors.append(f"demand_forecast.csv: 'type' must be 'peak', found {sorted(df.loc[bad_type, 'type'].unique())}")
    # a zone's peak forecast (if given) must cover every modeled year
    expected_years = set(years)
    peak = df[(df["type"] == "peak") & df["value"].notna()]
    coverage = peak.groupby("zone")["year"].apply(set)
    incomplete = coverage[~coverage.apply(expected_years.issubset)]
    if not incomplete.empty:
        errors.append(f"demand_forecast.csv: zone(s) with incomplete 'peak' year coverage: {sorted(incomplete.index)}")


def _check_demand_profile(instance, zones, times, years, errors):
    df = instance.demand_profile
    if not _require_columns(df, "demand_profile", ["zone", "time", "value"], errors):
        return
    # zone must be a real zone
    bad_zone = ~df["zone"].isin(zones)
    if bad_zone.any():
        errors.append(f"demand_profile.csv: zone(s) not in zones.csv: {sorted(df.loc[bad_zone, 'zone'].unique())}")
    # time must be a real time
    bad_time = ~df["time"].isin(times)
    if bad_time.any():
        errors.append(f"demand_profile.csv: time(s) not in times.csv: {sorted(df.loc[bad_time, 'time'].unique())}")
    if "year" in df.columns and df["year"].notna().any():
        # year, if given per-row, must be a modeled year
        bad_year = df["year"].notna() & ~df["year"].isin(years)
        if bad_year.any():
            errors.append(f"demand_profile.csv: year(s) not in years.csv: {sorted(df.loc[bad_year, 'year'].unique())}")
        return  # per-year profiles: completeness checked well enough by the above

    # a zone's demand profile (if given) must cover every modeled time
    counts = df[df["value"].notna()].groupby("zone")["time"].nunique()
    incomplete = counts[counts != len(times)]
    if not incomplete.empty:
        errors.append(f"demand_profile.csv: zone(s) with incomplete time coverage: {sorted(incomplete.index)}")


def _check_renewable_profiles(instance, times, years, errors):
    df = instance.renewable_profiles
    if not _require_columns(df, "renewable_profiles", ["profile_id", "zone", "tech", "generator", "time", "value"], errors):
        return
    # every row must belong to a profile
    if df["profile_id"].isna().any():
        errors.append("renewable_profiles.csv: row(s) with no profile_id")
    # time must be a real time
    bad_time = ~df["time"].isin(times)
    if bad_time.any():
        errors.append(f"renewable_profiles.csv: time(s) not in times.csv: {sorted(df.loc[bad_time, 'time'].unique())}")
    if "year" in df.columns and df["year"].notna().any():
        # year, if given per-row, must be a modeled year
        bad_year = df["year"].notna() & ~df["year"].isin(years)
        if bad_year.any():
            errors.append(f"renewable_profiles.csv: year(s) not in years.csv: {sorted(df.loc[bad_year, 'year'].unique())}")
    else:
        # template (no year) case: every profile needs a value for every modeled time, or a
        # generator could get derated to 0 output at some hours without anyone noticing
        counts = df[df["value"].notna()].groupby("profile_id")["time"].nunique()
        incomplete = counts[counts != len(times)]
        if not incomplete.empty:
            errors.append(f"renewable_profiles.csv: profile_id(s) with incomplete time coverage: {sorted(incomplete.index)}")


def _check_start_year_and_life(df, file_name, id_col, errors):
    """start_year and life are required on every row (never blank) and life > 0. A unit is
    active in start_year .. start_year + life - 1; retirement_year is no longer in the schema."""
    if "retirement_year" in df.columns:
        errors.append(f"{file_name}: 'retirement_year' is no longer in the schema -- use start_year + life")
    for col in ["start_year", "life"]:
        if col not in df.columns:
            errors.append(f"{file_name}: missing '{col}'")
            continue
        missing = df[col].isna()
        if missing.any():
            errors.append(f"{file_name}: missing '{col}' for: {sorted(df.loc[missing, id_col])}")
    if "life" in df.columns:
        bad = df["life"].astype(float) <= 0
        if bad.any():
            errors.append(f"{file_name}: 'life' must be > 0 for: {sorted(df.loc[bad, id_col])}")


def _check_generator_constants(instance, zones, errors):
    df = instance.generator_constants
    required = ["generator", "zone", "tech", "fuel", "capacity", "expandable", "isre"]
    if not _require_columns(df, "generator_constants", required, errors):
        return
    # every generator id must be unique
    if df["generator"].duplicated().any():
        errors.append(f"generator_constants.csv: duplicate generator id(s) {sorted(df.loc[df['generator'].duplicated(), 'generator'].unique())}")
    # zone must be a real zone
    bad_zone = ~df["zone"].isin(zones)
    if bad_zone.any():
        errors.append(f"generator_constants.csv: zone(s) not in zones.csv: {sorted(df.loc[bad_zone, 'zone'].unique())}")

    # every generator has a start_year and a life (annual/total_build_limit are optional,
    # defaulting to unconstrained (inf)), see seer/schema/README.md
    expandable = df["expandable"].astype(bool)
    _check_start_year_and_life(df, "generator_constants.csv", "generator", errors)

    # build limits, where given, must be non-negative (absent means unconstrained)
    for col in ["annual_build_limit", "total_build_limit"]:
        if col in df.columns:
            negative = df[col].astype(float) < 0
            if negative.any():
                errors.append(f"generator_constants.csv: negative '{col}' for generator(s): {sorted(df.loc[negative, 'generator'])}")

    # ramp rates, where given, are a fraction of capacity per hour: 0 < rate <= 1
    for col in ["ramp_up_rate", "ramp_down_rate"]:
        if col in df.columns:
            rate = df[col].astype(float)
            bad = rate.notna() & ((rate <= 0) | (rate > 1))
            if bad.any():
                errors.append(f"generator_constants.csv: '{col}' outside (0, 1] for generator(s): {sorted(df.loc[bad, 'generator'])}")

    # an existing unit (expandable=False) must have a capacity
    missing_capacity = ~expandable & df["capacity"].isna()
    if missing_capacity.any():
        errors.append(f"generator_constants.csv: existing generator(s) missing 'capacity': {sorted(df.loc[missing_capacity, 'generator'])}")

    # a renewable generator (isre=True) must have a profile_id, and it must point to a real profile
    isre = df["isre"].astype(bool)
    if "profile_id" not in df.columns:
        if isre.any():
            errors.append("generator_constants.csv: missing 'profile_id' (required when isre=True)")
    else:
        missing_profile = isre & df["profile_id"].isna()
        if missing_profile.any():
            errors.append(f"generator_constants.csv: isre generator(s) missing 'profile_id': {sorted(df.loc[missing_profile, 'generator'])}")
        valid_profile_ids = set(instance.renewable_profiles.get("profile_id", []))
        has_profile = df["profile_id"].notna()
        bad_profile = has_profile & ~df["profile_id"].isin(valid_profile_ids)
        if bad_profile.any():
            errors.append(f"generator_constants.csv: generator(s) with profile_id not in renewable_profiles.csv: {sorted(df.loc[bad_profile, 'generator'])}")


def _check_cost_table(df, file_name, id_col, constants_df, years, errors):
    if not _require_columns(df, file_name, [id_col, "cost_type"], errors):
        return
    # cost_type must be capital or marginal
    bad_cost_type = ~df["cost_type"].isin(["capital", "marginal"])
    if bad_cost_type.any():
        errors.append(f"{file_name}.csv: 'cost_type' must be 'capital' or 'marginal', found {sorted(df.loc[bad_cost_type, 'cost_type'].unique())}")

    # every modeled year must have its own column
    year_cols = [str(y) for y in years]
    missing_year_cols = set(year_cols) - set(df.columns)
    if missing_year_cols:
        errors.append(f"{file_name}.csv: missing column(s) for modeled year(s) {sorted(missing_year_cols)}")
    present_year_cols = [c for c in year_cols if c in df.columns]

    # every unit must have both a capex row and a marginal_cost row
    cost_types_by_id = df.groupby(id_col)["cost_type"].apply(set)
    incomplete = cost_types_by_id[cost_types_by_id != {"capital", "marginal"}]
    if not incomplete.empty:
        errors.append(f"{file_name}.csv: {id_col}(s) missing capex and/or marginal_cost: {sorted(incomplete.index)}")

    if not present_year_cols or constants_df is None or id_col not in constants_df.columns:
        return

    # every cost must be a real number -- no blanks. Existing units' capital is never read but
    # is still written as an explicit 0 (seer/schema/README.md).
    marginal = df[df["cost_type"] == "marginal"].set_index(id_col)
    bad_marginal = marginal[present_year_cols].isna().any(axis=1)
    if bad_marginal.any():
        errors.append(f"{file_name}.csv: marginal_cost has NaN value(s) for {id_col}(s): {sorted(bad_marginal[bad_marginal].index)}")

    capex = df[df["cost_type"] == "capital"].set_index(id_col)
    bad_capex = capex[present_year_cols].isna().any(axis=1)
    if bad_capex.any():
        expandable_ids = set(constants_df.loc[constants_df["expandable"].astype(bool), id_col])
        bad_ids = bad_capex[bad_capex].index
        bad_build = sorted(i for i in bad_ids if i in expandable_ids)
        bad_existing = sorted(i for i in bad_ids if i not in expandable_ids)
        if bad_build:
            errors.append(f"{file_name}.csv: expandable {id_col}(s) have NaN capex value(s): {bad_build}")
        if bad_existing:
            errors.append(f"{file_name}.csv: existing {id_col}(s) have blank capex -- must be 0: {bad_existing}")


def _check_generator_costs(instance, years, errors):
    _check_cost_table(instance.generator_costs, "generator_costs", "generator", instance.generator_constants, years, errors)
    # generator must be a real generator
    known_generators = set(instance.generator_constants.get("generator", []))
    bad = ~instance.generator_costs.get("generator", []).isin(known_generators) if "generator" in instance.generator_costs.columns else None
    if bad is not None and bad.any():
        errors.append(f"generator_costs.csv: generator(s) not in generator_constants.csv: {sorted(instance.generator_costs.loc[bad, 'generator'].unique())}")


def _check_storage_constants(instance, zones, errors):
    df = instance.storage_constants
    required = ["storage", "zone", "tech", "capacity_mw", "duration_hours", "expandable"]
    if not _require_columns(df, "storage_constants", required, errors):
        return
    # every storage id must be unique
    if df["storage"].duplicated().any():
        errors.append(f"storage_constants.csv: duplicate storage id(s) {sorted(df.loc[df['storage'].duplicated(), 'storage'].unique())}")
    # zone must be a real zone
    bad_zone = ~df["zone"].isin(zones)
    if bad_zone.any():
        errors.append(f"storage_constants.csv: zone(s) not in zones.csv: {sorted(df.loc[bad_zone, 'zone'].unique())}")

    # every storage unit has a start_year and a life -- the annual/total build limits and
    # unit_size_mwh are optional (absent means unconstrained / continuous), see
    # seer/schema/README.md
    expandable = df["expandable"].astype(bool)
    _check_start_year_and_life(df, "storage_constants.csv", "storage", errors)
    for col in ["annual_build_limit_mwh", "total_build_limit_mwh"]:
        if col in df.columns:
            negative = df[col].astype(float) < 0
            if negative.any():
                errors.append(f"storage_constants.csv: negative '{col}' for storage unit(s): {sorted(df.loc[negative, 'storage'])}")

    # an existing unit (expandable=False) must have a capacity_mw and duration_hours
    for col in ["capacity_mw", "duration_hours"]:
        missing = ~expandable & df[col].isna()
        if missing.any():
            errors.append(f"storage_constants.csv: existing storage unit(s) missing '{col}': {sorted(df.loc[missing, 'storage'])}")


def _check_storage_costs(instance, years, errors):
    _check_cost_table(instance.storage_costs, "storage_costs", "storage", instance.storage_constants, years, errors)
    # storage must be a real storage unit
    known_storage = set(instance.storage_constants.get("storage", []))
    bad = ~instance.storage_costs.get("storage", []).isin(known_storage) if "storage" in instance.storage_costs.columns else None
    if bad is not None and bad.any():
        errors.append(f"storage_costs.csv: storage id(s) not in storage_constants.csv: {sorted(instance.storage_costs.loc[bad, 'storage'].unique())}")
