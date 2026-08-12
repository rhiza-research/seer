"""Generate synthetic seer-schema input data.
Data generation is structured to enable easy scaling of the model (zones, years, generator/storage
mix) for solver performance testing, independent of which build backend (pyomo or PyPSA) consumes it.

Per-parameter-group generation (generator/storage/demand/renewable) lives in its own module,
each reading its own seed CSV from seed_data/. This file just orchestrates them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .demand_params import build_demand
from .generator_params import build_generators
from .re_potentials import build_renewable_profiles
from .storage_params import build_storage

SEED_DATA_DIR = Path(__file__).parent / "seed_data"

# ---- discounting ----
ANNUAL_DISCOUNT_RATE = 0.05  # 5%/year, applied to costs relative to the first modeled year

# ---- starting generation/storage mix: existing capacity is sized off each zone's own peak load ----
# (per-type parameters - ramp rates, costs, etc. - live in generator_params.py/storage_params.py)
STARTING_GENERATOR_MIX = {"gas": 0.5, "coal": 0.3, "PV": 0.2}  # shares of starting capacity by tech; must sum to 1
STARTING_STORAGE_MIX = {"battery": 1.0}
STARTING_RESERVE_MARGIN = 0.9  # existing generator capacity = max load over the horizon * this margin
STARTING_STORAGE_CAPACITY_MWH = 50.0  # existing storage capacity per zone

# default peak demand (MW), used for every zone unless zone_peak_demand_mw overrides it
DEFAULT_ZONE_PEAK_DEMAND_MW = 500.0

# ---- demand: randomized peak-demand growth rate, shared across zones ----
DEMAND_GROWTH_RATE_MEAN = 0.02
DEMAND_GROWTH_RATE_STD = 0.005

# ---- renewables: +/- perturbation applied per (zone, tech) to the seed capacity-factor curve ----
RE_PERTURBATION_PCT = 0.05

# ---- cost stochasticity: one +/- draw per generator/storage unit per cost type, held constant
# across years (so escalation trajectories stay smooth) ----
GENERATOR_COST_VARIANCE_PCT = {"capex": 0.10, "marginal_cost": 0.20}
STORAGE_COST_VARIANCE_PCT = {"capex": 0.10}  # marginal_cost is fixed at 0 for storage


# years - list of years
def generate_years(years):
    return pd.DataFrame({"year": years})


# settings - single-valued model settings, one row per setting (parameters, value)
def generate_settings(discount_rate):
    return pd.DataFrame({"parameters": ["dr"], "value": [discount_rate]})


# zones - list of zones
def generate_zones(zones, country_name):
    rows = [{"zone": zone, "country": country_name, "name": zone} for zone in zones]
    return pd.DataFrame(rows)


def generate_all_data(n_zones, years, output_path, country_name="Seerland", zone_peak_demand_mw=None):
    """Generate every seer-schema input CSV for a given problem scale (number of zones, years),
    writing it to `output_path`.

    zone_peak_demand_mw is either a single value (used for every zone) or a list with exactly
    one entry per zone; defaults to DEFAULT_ZONE_PEAK_DEMAND_MW for every zone.

    output_path may be a local path or a remote URL (e.g. gs://bucket/prefix); writes go through
    pandas/fsspec either way, so no filesystem-specific handling is needed here.
    """
    output_path = str(output_path).rstrip("/")
    is_remote = "://" in output_path
    if not is_remote:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    ZONES = [f"zone_{i}" for i in range(n_zones)]

    if zone_peak_demand_mw is None:
        zone_peak_demand_mw = [DEFAULT_ZONE_PEAK_DEMAND_MW] * n_zones
    elif np.isscalar(zone_peak_demand_mw):
        zone_peak_demand_mw = [zone_peak_demand_mw] * n_zones
    else:
        assert len(zone_peak_demand_mw) == n_zones, "zone_peak_demand_mw must be a single value or one per zone"

    demandforecast_df, demandprofile_df = build_demand(
        ZONES, zone_peak_demand_mw, years, DEMAND_GROWTH_RATE_MEAN, DEMAND_GROWTH_RATE_STD,
        seed_path=SEED_DATA_DIR / "demand_seed.csv",
    )

    # generator/storage starting capacity is sized off each zone's own peak load
    max_load = demandforecast_df.groupby("zone")["value"].max().to_dict()
    generator_initial_capacity = [max_load[zone] * STARTING_RESERVE_MARGIN for zone in ZONES]
    generator_constants_df, generator_costs_df = build_generators(
        ZONES, STARTING_GENERATOR_MIX, generator_initial_capacity, years, GENERATOR_COST_VARIANCE_PCT,
        seed_path=SEED_DATA_DIR / "generator_seed.csv",
    )
    storage_constants_df, storage_costs_df = build_storage(
        ZONES, STARTING_STORAGE_MIX, STARTING_STORAGE_CAPACITY_MWH, years, STORAGE_COST_VARIANCE_PCT,
        seed_path=SEED_DATA_DIR / "storage_seed.csv",
    )

    outputs = {
        "years": generate_years(years),
        "settings": generate_settings(ANNUAL_DISCOUNT_RATE),
        "zones": generate_zones(ZONES, country_name),
        "demandforecast": demandforecast_df,
        "demandprofile": demandprofile_df,
        "generator_constants": generator_constants_df,
        "generator_costs": generator_costs_df,
        "renewable_profiles": build_renewable_profiles(
            ZONES, RE_PERTURBATION_PCT, seed_path=SEED_DATA_DIR / "re_seed.csv"
        ),
        "storage_constants": storage_constants_df,
        "storage_costs": storage_costs_df,
    }

    for name, df in outputs.items():
        df.to_csv(f"{output_path}/{name}.csv", index=False)

    return output_path


def main():
    # ---- problem size: adjust these to scale the model ----
    repo_root = Path(__file__).resolve().parents[3]
    output_path = repo_root / "data" / "seer" / "synthetic" / "Seerland"
    generate_all_data(
        n_zones=1, years=[2025, 2030, 2035], output_path=output_path,
        country_name="Seerland", zone_peak_demand_mw=[500.0],
    )


if __name__ == "__main__":
    main()
