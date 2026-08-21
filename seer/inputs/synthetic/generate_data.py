"""Generate synthetic seer-schema input data.
Data generation is structured to enable easy scaling of the model (zones, years, generator/storage
mix) for solver performance testing, independent of which build backend (pyomo or PyPSA) consumes it.

Per-parameter-group generation (generator/storage/demand/renewable) lives in its own module,
each reading its own seed CSV from seed_data/, and each drawing its own noise (via noise.py) for
the quantities it owns. The one exception is demand's peak-growth rate: that's randomized here
(not inside demand_params.py) since generator/storage/network sizing all depend on its result
(via zonal capacity/max_load), so it needs to be resolved before those calls, not inside them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .demand_params import build_demand_forecast, build_demand_profile
from .generator_params import build_generators
from .network_params import build_network
from .noise import noise_multiplier
from .re_potentials import build_renewable_profiles
from .storage_params import build_storage

SEED_DATA_DIR = Path(__file__).parent / "seed_data"

# generation shares by tech
default_generator_mix = {"gas": 0.5, "coal": 0.3, "PV": 0.2}


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


def generate_all_data(
    n_zones, years, output_path, country_name="Seerland", zone_peak_demand_mw=500.0,
    annual_discount_rate=0.05,  # 5%/year, applied to costs relative to the first modeled year
    starting_generator_mix=None,  # shares of starting generator capacity by tech; must sum to 1
    starting_reserve_margin=1.5,  # existing generator capacity = max load over the horizon * this margin
    storage_capacity_pct=0.1,  # existing storage capacity = zonal capacity (max load) * this fraction
    demand_growth_rate_mean=0.02,  # root (unnoised) demand growth rate, shared across zones
    noise_pct=0.10,  # shared noise level for every randomized quantity (see noise.py)
    mesh_level=0.2,  # probability that each possible line beyond the spanning path also exists
    congestion_level=0.3,  # fraction by which line capacity is derated below the zones' average peak demand
):
    """Generate every seer-schema input CSV for a given problem scale (number of zones, years),
    writing it to `output_path`.

    zone_peak_demand_mw is either a single value (used for every zone) or a list with exactly
    one entry per zone.

    noise_pct is the single shared noise level passed to every parameter-group module (demand
    profile, renewable profiles, generator/storage costs, network line capacity), each of which
    draws its own noise ~ N(1, noise_pct) (clamped to be non-negative, see noise.py) for the
    quantities it owns. The one quantity randomized here instead is the demand growth rate,
    since generator/storage/network sizing all depend on its realized value.

    output_path may be a local path or a remote URL (e.g. gs://bucket/prefix); writes go through
    pandas/fsspec either way, so no filesystem-specific handling is needed here.
    """
    if starting_generator_mix is None:
        starting_generator_mix = default_generator_mix

    output_path = str(output_path).rstrip("/")
    is_remote = "://" in output_path
    if not is_remote:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    ZONES = [f"zone_{i}" for i in range(n_zones)]

    if np.isscalar(zone_peak_demand_mw):
        zone_peak_demand_mw = [zone_peak_demand_mw] * n_zones
    else:
        assert len(zone_peak_demand_mw) == n_zones, "zone_peak_demand_mw must be a single value or one per zone"

    # demand growth rate, perturbed by noise per zone
    growth_rate = demand_growth_rate_mean * noise_multiplier(noise_pct, size=n_zones)
    demandforecast_df = build_demand_forecast(ZONES, zone_peak_demand_mw, years, growth_rate)
    demandprofile_df = build_demand_profile(ZONES, noise_pct, seed_path=SEED_DATA_DIR / "demand_seed.csv")

    # zonal capacity - each zone's peak load over the horizon - drives generator, storage, and network sizing
    max_load = demandforecast_df.groupby("zone")["value"].max().to_dict()
    generator_initial_capacity = [max_load[zone] * starting_reserve_margin for zone in ZONES]
    generator_constants_df, generator_costs_df = build_generators(
        ZONES, starting_generator_mix, generator_initial_capacity, years, noise_pct,
        seed_path=SEED_DATA_DIR / "generator_seed.csv",
    )
    storage_initial_capacity = [max_load[zone] * storage_capacity_pct for zone in ZONES]
    storage_constants_df, storage_costs_df = build_storage(
        ZONES, storage_initial_capacity, years, noise_pct,
        seed_path=SEED_DATA_DIR / "storage_seed.csv",
    )
    network_df = build_network(ZONES, max_load, mesh_level, congestion_level, noise_pct)

    outputs = {
        "years": generate_years(years),
        "settings": generate_settings(annual_discount_rate),
        "zones": generate_zones(ZONES, country_name),
        "demandforecast": demandforecast_df,
        "demandprofile": demandprofile_df,
        "generator_constants": generator_constants_df,
        "generator_costs": generator_costs_df,
        "renewable_profiles": build_renewable_profiles(
            ZONES, noise_pct, seed_path=SEED_DATA_DIR / "re_seed.csv"
        ),
        "storage_constants": storage_constants_df,
        "storage_costs": storage_costs_df,
        "network": network_df,
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
