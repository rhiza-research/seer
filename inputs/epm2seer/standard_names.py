"""Standardize tech/fuel names to a common vocabulary aligned with PyPSA's technology-data
cost database (https://github.com/PyPSA/technology-data/blob/master/outputs/costs_2020.csv),
so cost data can eventually be looked up or cross-checked against that database directly.

Two synonym tables live in data/: pypsa_tech_names.csv and pypsa_fuel_names.csv, each mapping
every raw spelling variant we've seen across regions/pipelines to one canonical name. This is
a single, central place for that standardization -- used by every input-processing pipeline
(EPM/gdx today, others later), rather than each one maintaining its own separate alias list.
Raw-data cleanup that ISN'T tech/fuel naming (parameter-name variants, reconstructing
zone/tech/fuel when a region's raw gdx has no gmap symbol) stays in gdx/enrich_data.py, since
that needs GAMS-specific domain knowledge this module doesn't have. GAMS ordinal-prefix
stripping (e.g. "1.CoalEx" -> "CoalEx") is mostly done there too, but not every raw symbol
column goes through it (e.g. pFuelPrice.csv's own "fuel" column doesn't) -- normalization here
strips it too, as cheap defense-in-depth against whichever raw column was missed.

Matching is case/whitespace-insensitive. The tech table additionally supports compound
"<tech>-<fuel>" synonyms (checked before the plain tech-only synonym) for genuinely
tech-ambiguous raw values -- e.g. "ICE"/"ST"/"Steam" are all bare prime-mover labels that by
themselves don't say whether the plant burns coal, oil, or something else; the true PyPSA
technology can only be resolved once you also know the fuel.

A third table, pypsa_cost_tech_names.csv, is deliberately separate: some regions' raw renewable
tech values are genuinely finer-grained than a PyPSA tech (e.g. Georgia's WIND1/2/3, distinct
resource-quality tiers sharing identical hardware) and must stay that way through
`epm2seer.generator_params.match_re_profiles`, which joins a generator to its
renewable_profiles.csv row on (zone, tech) -- folding them into "onwind" there would collapse
three genuinely different capacity-factor curves into one. `coarsen_cost_tech` is meant to run
on generator_constants.tech only, strictly after that matching step has already happened (once
profile_id is assigned, downstream consumers key off profile_id, not tech) -- never on
renewable_profiles.csv's own tech column, which stays at its original granularity as
documented metadata.
"""

import re
import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

_ORDINAL_PREFIX = re.compile(r"^\d+\.")


def _normalize(value):
    """Strip a leading GAMS UEL ordinal prefix (e.g. "1.CoalEx" -> "CoalEx"), then lowercase
    + strip whitespace; None for NaN/missing input."""
    if pd.isna(value):
        return None
    stripped = _ORDINAL_PREFIX.sub("", str(value).strip())
    return stripped.strip().lower()


def _normalize_compound(tech, fuel):
    tech_n, fuel_n = _normalize(tech), _normalize(fuel)
    if tech_n is None or fuel_n is None:
        return None
    return f"{tech_n}-{fuel_n}"


def _load_synonyms(csv_path, has_fuel_col=False):
    """Build a {normalized_synonym: standard_name} dict from a synonym csv, plus (if
    has_fuel_col) a {standard_name: default_fuel} dict from its "fuel" column. Every
    standard_name implicitly maps to itself, even without an explicit self-referencing row,
    so already-canonical values pass straight through unchanged."""
    df = pd.read_csv(csv_path)
    lookup = {}
    default_fuel = {}
    for _, row in df.iterrows():
        standard_name = str(row["standard_name"]).strip()
        lookup[_normalize(standard_name)] = standard_name
        lookup[_normalize(row["synonym"])] = standard_name
        if has_fuel_col:
            fuel_value = row.get("fuel")
            if pd.notna(fuel_value) and str(fuel_value).strip():
                fuel_value = str(fuel_value).strip()
                existing = default_fuel.get(standard_name)
                if existing is not None and existing != fuel_value:
                    warnings.warn(
                        f"{csv_path.name}: {standard_name!r} has conflicting default fuel "
                        f"values ({existing!r} vs {fuel_value!r}) across its synonym rows -- "
                        f"keeping {existing!r}."
                    )
                else:
                    default_fuel[standard_name] = fuel_value
    return lookup, default_fuel


_TECH_LOOKUP, _TECH_DEFAULT_FUEL = _load_synonyms(DATA_DIR / "pypsa_tech_names.csv", has_fuel_col=True)
_FUEL_LOOKUP, _ = _load_synonyms(DATA_DIR / "pypsa_fuel_names.csv")
_COST_TECH_LOOKUP, _ = _load_synonyms(DATA_DIR / "pypsa_cost_tech_names.csv")


def _resolve_tech(tech, fuel):
    """Returns (standard_name_or_original, matched: bool)."""
    if fuel is not None and pd.notna(fuel):
        compound = _normalize_compound(tech, fuel)
        if compound in _TECH_LOOKUP:
            return _TECH_LOOKUP[compound], True
    bare = _normalize(tech)
    if bare in _TECH_LOOKUP:
        return _TECH_LOOKUP[bare], True
    return tech, False


def _resolve_fuel(fuel):
    bare = _normalize(fuel)
    if bare in _FUEL_LOOKUP:
        return _FUEL_LOOKUP[bare], True
    return fuel, False


def standardize_tech(tech, fuel=None):
    """Map a tech value (or pandas Series of them) to its canonical PyPSA-aligned name.

    When `fuel` is given (a matching scalar or Series), a compound "<tech>-<fuel>" synonym is
    tried first -- needed for bare prime-mover labels ("ICE", "ST", "Steam") that can't be
    resolved from tech alone. Falls back to a plain tech-only match. Warns (once per distinct
    unmapped (tech, fuel) pair) and leaves the original value unchanged for anything neither
    matches.
    """
    if isinstance(tech, pd.Series):
        fuel_series = fuel if isinstance(fuel, pd.Series) else pd.Series([fuel] * len(tech), index=tech.index)
        cache = {}
        unmapped = set()
        result = []
        for t, f in zip(tech, fuel_series):
            if pd.isna(t):
                result.append(t)
                continue
            key = (t, f if pd.notna(f) else None)
            if key not in cache:
                value, matched = _resolve_tech(t, f)
                cache[key] = value
                if not matched:
                    unmapped.add(key)
            result.append(cache[key])
        if unmapped:
            warnings.warn(
                f"No tech standardization mapping for (tech, fuel) pair(s): "
                f"{sorted(unmapped, key=str)} -- leaving as-is."
            )
        return pd.Series(result, index=tech.index, name=tech.name)

    value, matched = _resolve_tech(tech, fuel)
    if not matched:
        warnings.warn(f"No tech standardization mapping for tech={tech!r}, fuel={fuel!r} -- leaving as-is.")
    return value


def standardize_fuel(fuel):
    """Map a fuel value (or pandas Series of them) to its canonical PyPSA-aligned name. Warns
    (once per distinct unmapped value) and leaves the original value unchanged for anything
    unmapped."""
    if isinstance(fuel, pd.Series):
        cache = {}
        unmapped = set()
        result = []
        for f in fuel:
            if pd.isna(f):
                result.append(f)
                continue
            if f not in cache:
                value, matched = _resolve_fuel(f)
                cache[f] = value
                if not matched:
                    unmapped.add(f)
            result.append(cache[f])
        if unmapped:
            warnings.warn(f"No fuel standardization mapping for: {sorted(unmapped, key=str)} -- leaving as-is.")
        return pd.Series(result, index=fuel.index, name=fuel.name)

    value, matched = _resolve_fuel(fuel)
    if not matched:
        warnings.warn(f"No fuel standardization mapping for fuel={fuel!r} -- leaving as-is.")
    return value


def default_fuel_for_tech(standard_tech):
    """The default fuel for a standardized tech name (from pypsa_tech_names.csv's "fuel"
    column), or None if that tech has no associated fuel (renewables, storage, import)."""
    return _TECH_DEFAULT_FUEL.get(standard_tech)


def coarsen_cost_tech(tech):
    """Map a tech value (or pandas Series) that's finer-grained than any real PyPSA technology
    (resource-quality tiers like Georgia's Wind1/2/3) to its canonical costing tech, via
    pypsa_cost_tech_names.csv. Anything not in that table (already-canonical names, or any
    other unmapped value) passes through unchanged -- unlike standardize_tech, this never
    warns, since most tech values are expected not to need this second pass."""
    if isinstance(tech, pd.Series):
        return tech.map(lambda t: _COST_TECH_LOOKUP.get(_normalize(t), t) if pd.notna(t) else t)
    return _COST_TECH_LOOKUP.get(_normalize(tech), tech) if pd.notna(tech) else tech
