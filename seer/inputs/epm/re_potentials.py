"""Build renewable_profiles.csv from EPM input data.

renewable_profiles.csv can carry two different kinds of rows, distinguished by which of
`tech`/`generator` is populated (see seer/inputs/README.md):
- generic (tech set, generator NaN): one profile shared by every generator of that
  (zone, tech), sourced from pVREProfile.csv. tech names are standardized (e.g. "Solar" ->
  "PV") to match generator_constants.tech's canonical vocabulary (see gdx.enrich_data's
  FUEL_NAME_MAP tech_label conventions).
- generator-specific (generator set, tech NaN): a profile for one individual generator,
  sourced from pVREgenProfile.csv. That symbol sometimes has a real "generator" column (e.g.
  hydro plants, resolved via gdx.enrich_data.enrich_vre_zone_tech) -- other regions' variant
  has no such column, and its own "tech" values are really per-site resource labels, not
  necessarily real generator names.

This module returns every candidate profile found, not yet filtered to only the ones that
actually match a real generator/tech combination -- see to_seer.epm_to_seer's matching step,
which needs the generator list this module doesn't have access to.
"""
from pathlib import Path

import pandas as pd

from ._time_convert import convert_season_day_hour

# the month column's name varies across sources
MONTH_COL_OPTIONS = ["month", "Month", "Month of the Year", "season"]

# pVREProfile.csv's raw tech names -> generator_constants.tech's canonical vocabulary
RENEWABLE_TECH_ALIAS = {
    "Solar": "PV",
    "SolarTracker": "PV",
}


def build_re_potentials(data_path="data/Gambia/"):
    """Process renewable potentials into a seer schema file: renewable_profiles.csv (see
    module docstring for the generic/generator-specific row shapes)."""
    generic = _read_zone_tech_profile(data_path + "pVREProfile.csv")
    specific = _read_gen_profile(data_path + "pVREgenProfile.csv")

    if generic is None and specific is None:
        raise ValueError("No renewable profile data found (neither pVREProfile.csv nor pVREgenProfile.csv).")

    frames = [df for df in (generic, specific) if df is not None]
    return pd.concat(frames, ignore_index=True)


def _read_zone_tech_profile(path):
    """pVREProfile.csv: zone/tech-level (generic) profiles."""
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    col = next((c for c in MONTH_COL_OPTIONS if c in df.columns), None)
    if col is None:
        raise ValueError(f"No valid month column found in {path}.")
    df = df.rename(columns={col: "season"})

    df = convert_season_day_hour(df)
    df["tech"] = df["tech"].replace(RENEWABLE_TECH_ALIAS)
    df["generator"] = pd.NA
    return df[["zone", "tech", "generator", "time", "value"]]


def _read_gen_profile(path):
    """pVREgenProfile.csv: per-individual-identifier profiles. Uses the "generator" column
    when present (a genuine per-plant symbol); otherwise its own "tech" column values ARE the
    per-row identifier (some regions' variant has no separate "generator" column at all --
    to_seer's matching step drops whichever of these don't correspond to a real generator)."""
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if "zone" not in df.columns:
        return None  # couldn't be resolved to zone/tech upstream (no gmap) -- unusable

    col = next((c for c in MONTH_COL_OPTIONS if c in df.columns), None)
    if col is None:
        raise ValueError(f"No valid month column found in {path}.")
    df = df.rename(columns={col: "season"})
    df = convert_season_day_hour(df)

    identifier_col = "generator" if "generator" in df.columns else "tech"
    df["generator"] = df[identifier_col]
    df["tech"] = pd.NA
    return df[["zone", "tech", "generator", "time", "value"]]
