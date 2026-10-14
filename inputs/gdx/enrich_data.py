"""Fuel/tech domain lookups and generator & VRE profile enrichment.

Standardizing raw tech/fuel *values* (e.g. "CCGTEx" -> a canonical PyPSA technology name) is
deliberately NOT done here -- it happens once, centrally, in epm2seer.standard_names, so every raw
region-specific spelling is seen and mapped in one place rather than split across gdx's own
ad hoc vocabulary and a second seer-side pass. This module still normalizes raw *parameter
names* (STORAGE_PARAM_ALIAS, GEN_PARAM_ALIAS -- e.g. "Capacity" vs "CapacityMWh") and does the
structural work of reconstructing zone/fuel/tech for generators when a region's raw gdx has
no gmap symbol at all -- both of those need GAMS-specific domain knowledge epm2seer has no
way to reconstruct on its own, so they stay here.
"""

# Import relevant packages
import re
import numpy as np
import pandas as pd
from .standardize_profiles import standardize_all_profiles

# Aliases for pStorageDataInput parameter names (the "parameters" column values) that vary
# across regions/vintages of the raw pStorDataExcel symbol -- e.g. some regions use plain
# "Capacity"/"Capex" instead of "CapacityMWh"/"CapexMWh" for the same storage-energy fields.
# Canonical names (right-hand side) are what epm2seer.storage_params expects.
STORAGE_PARAM_ALIAS = {
    "Capacity": "CapacityMWh",
    "Capex": "CapexMWh",
}

# Alias for pGenDataInputCustom/pGenDataInputDefault parameter names that vary across regions.
# "VRETech" is the synonym for "RETech" (epm2seer.generator_params expects "RETech";
# renamed to "isre" in the seer schema): confirmed by checking actual values across regions --
# VRETech=1 only for variable-renewable techs (PV/Wind), matching RETech's meaning exactly.
# NOT the same as "RE_Flag", which is also 1 for storage/hydro in most regions (a broader
# "RE-related" eligibility flag, not "output driven by a weather profile") and would
# mis-tag non-profile-driven generators as renewable if aliased here.
GEN_PARAM_ALIAS = {
    "VRETech": "RETech",
}


def _strip_ordinal_prefix(name) -> str:
    """Strip a leading GAMS ordinal prefix, e.g. "18.WindNew" -> "WindNew"."""

    return re.sub(r"^\d+\.", "", str(name).strip())


def build_fuel_id_lookup(ftf):
    """
    Build a {fuel_id: (fuel_name, tech_name)} dict from a country's ftfindex: the raw,
    un-standardized names associated with each numeric fuel_id in pGenDataExcel.

    ftfindex's declared column order (see inputs/gdx/symbol_mapping.csv) is (fuel, tech,
    value) -- a broader category (e.g. "Hydro") and a more specific label (e.g. "Water").
    This is called on the *raw* gamspy-loaded frame, before symbol-mapping has renamed its
    columns to those real names (they're still generic, e.g. "uni_0"/"uni_1"/"value") -- so
    columns are read positionally when they aren't already literally named "fuel"/"tech".

    Both fuel_name and tech_name are kept rather than collapsed to one, since either can be
    blank for a given row (some regions only populate one of the two) and losing whichever
    one IS populated would silently drop real information; when one side is blank, the
    other's value is used for both placeholders. Standardizing these raw names to a
    canonical tech/fuel vocabulary happens downstream, in epm2seer.standard_names (see this
    module's docstring for why).

    Returns an empty dict if the structure is not recognized.
    """

    ftf = ftf.copy()
    ftf.columns = ftf.columns.str.strip().str.lower()

    def _clean(value):
        stripped = _strip_ordinal_prefix(value) if pd.notna(value) else ""
        return stripped if stripped.strip() else None

    if "fuel" in ftf.columns and "tech" in ftf.columns:
        fuel_col, tech_col = "fuel", "tech"
        id_col = next((c for c in ftf.columns if c not in ("fuel", "tech")), None)
    elif len(ftf.columns) >= 3:
        fuel_col, tech_col, id_col = ftf.columns[0], ftf.columns[1], ftf.columns[2]
    elif len(ftf.columns) == 2:
        fuel_col = tech_col = ftf.columns[0]
        id_col = ftf.columns[1]
    else:
        return {}

    if id_col is None:
        return {}

    lookup = {}
    for _, row in ftf.iterrows():
        fuel_id = row[id_col]
        fuel_name = _clean(row[fuel_col])
        tech_name = _clean(row[tech_col])
        if fuel_name or tech_name:
            lookup[fuel_id] = (fuel_name or tech_name, tech_name or fuel_name)

    return lookup


def enrich_gen_data(region, country_data, country_data_raw):
    """
    Populate zone/fuel/tech columns in pGenDataInputCustom. 
    Helper for enrich_gen_data function.
    """

    target = country_data["pGenDataInputCustom"].copy()
    gen_raw = country_data_raw["pGenDataExcel"].copy()
    zones = country_data_raw["pZoneIndex"].copy()
    ftf = country_data_raw["ftfindex"].copy()

    target.columns = target.columns.str.strip().str.lower()
    gen_raw.columns = gen_raw.columns.str.strip().str.lower()
    zones.columns = zones.columns.str.strip().str.lower()

    # ── Zone mapping ─────────────────────────────────────────────────────────
    zone_id_col = next((c for c in zones.columns if pd.to_numeric(zones[c], errors="coerce").notna().all()), None)
    zone_name_col = next((c for c in zones.columns if c != zone_id_col), None)

    if zone_id_col and zone_name_col:
        zone_map = (
            zones.assign(**{zone_id_col: pd.to_numeric(zones[zone_id_col], errors="coerce")})
            .dropna(subset=[zone_id_col])
            .set_index(zone_id_col)[zone_name_col]
            .to_dict()
        )
    else:
        zone_map = {}
        print(f"  [{region}] warn — could not build zone map")

    # ── Pivot pGenDataExcel to wide format ───────────────────────────────────
    param_col = next(
        (c for c in gen_raw.columns if c != "value"
         and gen_raw[c].dropna().astype(str).str.lower().isin(["zone", "fuel1", "fuel2"]).any()),
        None
    )
    gen_col = next((c for c in gen_raw.columns if c not in (param_col, "value")), None)

    if not param_col or not gen_col:
        print(f"  [{region}] warn — could not identify columns in pGenDataExcel")
        return target

    gen_wide = (
        gen_raw[[gen_col, param_col, "value"]]
        .dropna(subset=[param_col])
        .pivot_table(index=gen_col, columns=param_col, values="value", aggfunc="first", observed=False)
        .reset_index()
    )
    gen_wide.columns = [str(c).strip().lower() for c in gen_wide.columns]
    gen_wide = gen_wide.rename(columns={gen_col.lower(): "generator"})

    fuel_id_lookup = build_fuel_id_lookup(ftf)

    # ── Resolve zone, fuel, tech per generator ───────────────────────────────
    if "zone" in gen_wide.columns:
        gen_wide["_zone"] = pd.to_numeric(gen_wide["zone"], errors="coerce").map(zone_map)
    else:
        gen_wide["_zone"] = np.nan

    fuel_id_series = None
    if "fuel1" in gen_wide.columns:
        fuel_id_series = pd.to_numeric(gen_wide["fuel1"], errors="coerce")
    if "fuel2" in gen_wide.columns:
        fuel2 = pd.to_numeric(gen_wide["fuel2"], errors="coerce")
        fuel_id_series = fuel_id_series.fillna(fuel2) if fuel_id_series is not None else fuel2

    if fuel_id_series is not None and fuel_id_lookup:
        # fuel_id_lookup gives (fuel_name, tech_name) raw placeholders per fuel_id --
        # standardizing them into canonical tech/fuel values happens downstream, in
        # epm2seer.standard_names
        gen_wide["_fuel"] = fuel_id_series.map({k: v[0] for k, v in fuel_id_lookup.items()})
        gen_wide["_tech"] = fuel_id_series.map({k: v[1] for k, v in fuel_id_lookup.items()})
    else:
        gen_wide["_fuel"] = np.nan
        gen_wide["_tech"] = np.nan
        print(f"  [{region}] warn — fuel/tech mapping could not be applied")

    skip_cols = {"zone", "fuel", "tech", "technology", "value", "parameters"}
    gen_id_col = next((c for c in target.columns if c not in skip_cols), None)

    if not gen_id_col:
        print(f"  [{region}] warn — could not identify generator id column in pGenDataInputCustom")
        return target

    enrichment = gen_wide[["generator", "_zone", "_fuel", "_tech"]].drop_duplicates("generator")
    target = target.merge(enrichment, how="left", left_on=gen_id_col, right_on="generator")

    for target_col, enrich_col in [("zone", "_zone"), ("fuel", "_fuel"), ("tech", "_tech")]:
        if target_col in target.columns and enrich_col in target.columns:
            target[target_col] = target[target_col].fillna(target[enrich_col])

    # "generator" from the merge is only a redundant artifact column when gen_id_col is a
    # differently-named column holding the same identity (e.g. "plant"); when gen_id_col IS
    # "generator", the merge collapses both sides into that single column, and it's target's
    # own real per-generator identity — must not be dropped.
    drop_cols = ["_zone", "_fuel", "_tech"]
    if gen_id_col != "generator":
        drop_cols.append("generator")

    return target.drop(columns=drop_cols, errors="ignore")


def map_vre_tech(region, symbol_name, df):
    """Strip any leading GAMS UEL ordinal prefix (e.g. "18.WindNew" -> "WindNew") from
    df['tech'] -- a raw GDX artifact, not a naming choice. Standardizing the resulting raw
    tech spelling to a canonical vocabulary happens downstream, in epm2seer.standard_names (see
    this module's docstring for why).
    """

    if "tech" not in df.columns:
        return df
    df = df.copy()
    df["tech"] = df["tech"].map(_strip_ordinal_prefix)
    return df


def _resolve_gmap_generator(generator: str, lookup: pd.DataFrame):
    """
    Resolve a single pVREgenProfile generator name to a (zone, tech) pair from
    gmap, tolerating naming drift between the GDX generator set and gmap's
    generator labels:
      1. Exact match.
      2. Word-boundary prefix match — gmap generators equal to `generator`
         plus a trailing " <qualifier>" (e.g. "Rukarara 5" matches
         "Rukarara 5 Mushishito", but not an unrelated "Rukarara 2", and
         "Boali" matches "Boali 1 Rebuild"/"Boali 2"/"Boali 3" but not the
         unrelated "Boaliiii"). If every such candidate agrees on (zone,
         tech), that shared value is used; if they disagree, the match is
         treated as unresolved.

    Returns (zone, tech, candidates) where candidates is the list of gmap
    generator names considered ambiguous (empty unless there was a conflict).
    """

    if generator in lookup.index:
        row = lookup.loc[generator]
        return row["zone"], row["tech"], []

    prefix = f"{generator} "
    candidates = [name for name in lookup.index if name.startswith(prefix)]
    if not candidates:
        return np.nan, np.nan, []

    values = {(lookup.loc[name, "zone"], lookup.loc[name, "tech"]) for name in candidates}
    if len(values) == 1:
        zone, tech = next(iter(values))
        return zone, tech, []

    return np.nan, np.nan, candidates


def enrich_vre_zone_tech(region: str, country_data: dict) -> pd.DataFrame | None:
    """
    Populate zone/tech in pVREgenProfile via gmap (generator -> zone, tech) for
    the generator-indexed variant (no zone/tech dimension in the raw symbol).
    No-op if pVREgenProfile already has zone/tech (the 5-dim variant) or is
    missing/empty.
    """

    vre = country_data.get("pVREgenProfile")
    if vre is None or vre.empty or "generator" not in vre.columns:
        return vre

    gmap = country_data.get("gmap")
    if gmap is None or gmap.empty:
        print(f"  [{region}] [pVREgenProfile] warn — gmap missing, cannot populate zone/tech")
        return vre

    lookup = gmap.drop_duplicates("generator").set_index("generator")[["zone", "tech"]]

    vre = vre.copy()
    resolved = {}
    ambiguous = {}
    for generator in vre["generator"].unique():
        zone, tech, candidates = _resolve_gmap_generator(generator, lookup)
        resolved[generator] = (zone, tech)
        if candidates:
            ambiguous[generator] = candidates

    vre["zone"] = vre["generator"].map(lambda g: resolved[g][0])
    vre["tech"] = vre["generator"].map(lambda g: resolved[g][1])

    if ambiguous:
        print(f"  [{region}] [pVREgenProfile] warn — ambiguous gmap matches (conflicting zone/tech): {ambiguous}")

    missing = sorted(vre.loc[vre["tech"].isna(), "generator"].unique())
    if missing:
        print(f"  [{region}] [pVREgenProfile] warn — no gmap entry for generators: {missing}")

    return vre


def enrich_gen_vre_data(region: str, country_data: dict, country_data_raw: dict) -> dict:
    """
    Populate missing zone, fuel, and tech columns in pGenDataInputCustom using
    pZoneIndex and ftfindex (raw, un-standardized fuel names -- see this module's
    docstring). Also populate zone/tech in pVREgenProfile via gmap, and strip GAMS UEL
    ordinal prefixes from tech values in pVREProfile / pVREgenProfile.

    Returns
    -------
    dict
        Mapping of symbol_name -> enriched dataframe, containing only the
        symbols that were actually processed.
    """

    updates = {}

    # ── pGenDataInputCustom enrichment ────────────────────────────────────────
    required_raw = ["pGenDataExcel", "pZoneIndex", "ftfindex"]
    missing_raw = [
        sym for sym in required_raw
        if sym not in country_data_raw or country_data_raw[sym] is None or country_data_raw[sym].empty
    ]

    if "pGenDataInputCustom" not in country_data or country_data["pGenDataInputCustom"] is None or country_data["pGenDataInputCustom"].empty:
        print(f"  [{region}] skipping pGenDataInputCustom — missing pGenDataInputCustom")
    elif missing_raw:
        print(f"  [{region}] skipping pGenDataInputCustom — missing {missing_raw} in raw gdx symbols")
    else:
        updates["pGenDataInputCustom"] = enrich_gen_data(region, country_data, country_data_raw)

    # ── pVREProfile / pVREgenProfile zone/tech population + normalization ────
    for vre_symbol in ("pVREProfile", "pVREgenProfile"):
        vre_df = country_data.get(vre_symbol)
        if vre_df is None or vre_df.empty:
            continue
        if vre_symbol == "pVREgenProfile":
            vre_df = enrich_vre_zone_tech(region, country_data)
        if vre_df is not None and "tech" in vre_df.columns:
            updates[vre_symbol] = map_vre_tech(region, vre_symbol, vre_df)

    return updates


def normalize_storage_params(df):
    """Normalize pStorageDataInput parameter-name variants (STORAGE_PARAM_ALIAS) to the
    canonical names epm2seer.storage_params expects."""
    df = df.copy()
    df["parameters"] = df["parameters"].replace(STORAGE_PARAM_ALIAS)
    return df


def normalize_gen_params(df):
    """Normalize pGenDataInputCustom/pGenDataInputDefault parameter-name variants
    (GEN_PARAM_ALIAS) to the canonical names epm2seer.generator_params expects."""
    df = df.copy()
    df["parameters"] = df["parameters"].replace(GEN_PARAM_ALIAS)
    return df


def reconcile_data(all_data, all_data_raw):
    """
    Apply enrich_gen_vre_data() to every country/region — enriching pGenDataInputCustom and
    (for pVREProfile/pVREgenProfile) stripping GAMS UEL ordinal prefixes from tech values,
    then merging the results back in. Tech/fuel *value* standardization (raw spelling ->
    canonical name) is not done here -- see this module's docstring.

    Apply standardize_all_profiles to ensure demand and VRE generation
    profiles are all standardized for ingestion into SEER schema
    """

    reconciled = {}
    for region, country_data in all_data.items():
        print(f"Enriching: {region}")

        country_data_raw = all_data_raw[region]
        updated = dict(country_data)
        updates = enrich_gen_vre_data(region, country_data, country_data_raw)
        updated.update(updates)

        storage_df = updated.get("pStorageDataInput")
        if storage_df is not None and not storage_df.empty:
            updated["pStorageDataInput"] = normalize_storage_params(storage_df)

        for gen_symbol in ("pGenDataInputCustom", "pGenDataInputDefault"):
            gen_df = updated.get(gen_symbol)
            if gen_df is not None and not gen_df.empty:
                updated[gen_symbol] = normalize_gen_params(gen_df)

        updated.update(standardize_all_profiles(region, updated))
        reconciled[region] = updated

    return reconciled
