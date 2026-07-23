"""Generate synthetic input CSVs for the seer capacity expansion model.

Follows the seer schema (see synthseer/data/<region> for the reference
layout): one CSV per input file, each produced by its own generate_*
function below.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "input_data"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---- demand profile shape: this defines how many hourly timepoints exist per year ----
MONTHS = [f"m{i}" for i in range(1, 13)]  # seasons = calendar months
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # standard (non-leap) year, sums to 365
HOUR_NAMES = [f"t{i}" for i in range(1, 25)]

HOURLY_LOAD_SHAPE = [
    0.62, 0.58, 0.55, 0.54, 0.56, 0.62,
    0.72, 0.85, 0.92, 0.95, 0.97, 0.98,
    0.96, 0.94, 0.93, 0.92, 0.94, 0.99,
    1.00, 0.97, 0.90, 0.80, 0.72, 0.66,
]

# ---- demand forecast assumptions ----
ZONE_PEAK_DEMAND_MEAN_MW = 500.0
ZONE_PEAK_DEMAND_STD_MW = 100.0
ZONE_GROWTH_RATE_MEAN = 0.02
ZONE_GROWTH_RATE_STD = 0.005

# ---- renewable profile assumptions ----
RENEWABLE_TECHS = ["PV"]
PV_DAILY_SHAPE = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.02, 0.15, 0.40, 0.65, 0.85, 0.95,
    1.00, 0.95, 0.85, 0.65, 0.40, 0.15,
    0.02, 0.0, 0.0, 0.0, 0.0, 0.0,
]
# scales the nominal daily shape between 0 and 1 by month, to reflect seasonal irradiance
PV_MONTHLY_SCALE = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75]


"""Write model data files"""

# years.csv - list of years
def generate_years(years):
    df = pd.DataFrame({"year": years})
    df.to_csv(DATA_DIR / "years.csv", index=False)
    return df


# zones.csv - list of zones
def generate_zones(zones, country_name):
    rows = [{"zone": zone, "country": country_name, "name": zone} for zone in zones]
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "zones.csv", index=False)
    return df


# generator_constants.csv - list of generators and their fixed parameters
def generate_generator_constants(years, zones, generator_types, generator_type_params):
    rows = []
    for zone in zones:
        for gen_type in generator_types:
            params = generator_type_params[gen_type]
            common = {
                "zone": zone,
                "tech": gen_type,
                "fuel": gen_type,
                "ramp_down_rate": params["ramp_down_rate"],
                "ramp_up_rate": params["ramp_up_rate"],
                "life": params["life"],
                "RETech": params["RETech"],
            }

            # existing generator: present from the start of the horizon, retires after it
            rows.append({
                "generator": f"{zone}_{gen_type}_existing",
                **common,
                "capacity": params["existing_capacity"],
                "retirement_year": years[-1] + 1,
                "start_year": years[0],
                "unit_size": params["existing_capacity"],
                "annual_build_limit": None,
                "expandable": False,
            })

            # new-build option: no fixed start/retirement, capacity added subject to a per-year limit
            rows.append({
                "generator": f"{zone}_{gen_type}_new",
                **common,
                "capacity": params["new_unit_size"],
                "retirement_year": None,
                "start_year": None,
                "unit_size": params["new_unit_size"],
                "annual_build_limit": params["annual_build_limit"],
                "expandable": True,
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "generator_constants.csv", index=False)
    return df


# generator_costs.csv - costs for each generator and cost type
def generate_generator_costs(years, zones, generator_types, generator_type_params):
    base_year = years[0]
    rows = []
    for zone in zones:
        for gen_type in generator_types:
            params = generator_type_params[gen_type]
            for suffix in ("existing", "new"):
                generator = f"{zone}_{gen_type}_{suffix}"
                for cost_type, base_value, escalation in (
                    ("capex", params["capex"], params["capex_escalation"]),
                    ("marginal_cost", params["marginal_cost"], params["marginal_cost_escalation"]),
                ):
                    row = {"generator": generator, "cost_type": cost_type}
                    for year in years:
                        row[year] = round(base_value * (1 + escalation) ** (year - base_year), 4)
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "generator_costs.csv", index=False)
    return df

# demandforecast.csv - annual peak demand for each zone
def generate_demandforecast(years, zones):
    base_year = years[0]
    starting_demand = rng.normal(ZONE_PEAK_DEMAND_MEAN_MW, ZONE_PEAK_DEMAND_STD_MW, size=len(zones))
    growth_rate = rng.normal(ZONE_GROWTH_RATE_MEAN, ZONE_GROWTH_RATE_STD, size=len(zones))

    rows = [
        {"zone": zone, "type": "peak", "year": year, "value": round(start * (1 + growth) ** (year - base_year), 3)}
        for zone, start, growth in zip(zones, starting_demand, growth_rate)
        for year in years
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "demandforecast.csv", index=False)
    return df


# demandprofile.csv - hourly demand profile for each zone
def generate_demandprofile(zones):
    rows = [
        {"zone": zone, "season": month, "day": f"d{day}", "hour": hour, "value": value}
        for zone in zones
        for month, days_in_month in zip(MONTHS, DAYS_IN_MONTH)
        for day in range(1, days_in_month + 1)
        for hour, value in zip(HOUR_NAMES, HOURLY_LOAD_SHAPE)
    ]

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "demandprofile.csv", index=False)
    return df


# renewable_profiles.csv - hourly renewable availability profile for each zone and tech
def generate_renewable_profiles(zones):
    rows = [
        {"zone": zone, "tech": tech, "month": month, "day": day, "hour": hour, "value": round(scale * shape, 6)}
        for zone in zones
        for tech in RENEWABLE_TECHS
        for month, days_in_month, scale in zip(range(1, 13), DAYS_IN_MONTH, PV_MONTHLY_SCALE)
        for day in range(1, days_in_month + 1)
        for hour, shape in zip(range(1, 25), PV_DAILY_SHAPE)
    ]

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "renewable_profiles.csv", index=False)
    return df


def generate_storage_constants():
    pass


def generate_storage_costs():
    pass


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # ---- problem size: adjust these to scale the model ----
    YEARS = [2025, 2030, 2035]
    N_ZONES = 2
    ZONES = [f"zone_{i}" for i in range(N_ZONES)]
    GENERATOR_TYPES = ["gas", "coal", "PV"]
    COUNTRY_NAME = "Seerland"  # all zones belong to this single synthetic country
    # ---- generator constants, one set of parameters per generator type ----
    GENERATOR_TYPE_PARAMS = {
        #           ramp_down  ramp_up  life(yr)  RETech  existing_MW  new_unit_MW  annual_build_limit_MW  capex($M/MW)  capex_escalation  marginal_cost($/MWh)  marginal_cost_escalation
        "gas":  {"ramp_down_rate": 1.0, "ramp_up_rate": 1.0, "life": 30.0, "RETech": False, "existing_capacity": 100.0, "new_unit_size": 50.0, "annual_build_limit": 50.0, "capex": 0.80, "capex_escalation": 0.005, "marginal_cost": 45.0, "marginal_cost_escalation": 0.01},
        "coal": {"ramp_down_rate": 0.5, "ramp_up_rate": 0.5, "life": 40.0, "RETech": False, "existing_capacity": 80.0, "new_unit_size": 100.0, "annual_build_limit": 50.0, "capex": 2.00, "capex_escalation": 0.005, "marginal_cost": 35.0, "marginal_cost_escalation": 0.01},
        "PV":   {"ramp_down_rate": 1.0, "ramp_up_rate": 1.0, "life": 25.0, "RETech": True, "existing_capacity": 20.0, "new_unit_size": 20.0, "annual_build_limit": 20.0, "capex": 0.90, "capex_escalation": -0.03, "marginal_cost": 3.0, "marginal_cost_escalation": 0.0},
    }

    generate_years(YEARS)
    generate_zones(ZONES, COUNTRY_NAME)
    generate_generator_constants(YEARS, ZONES, GENERATOR_TYPES, GENERATOR_TYPE_PARAMS)
    generate_generator_costs(YEARS, ZONES, GENERATOR_TYPES, GENERATOR_TYPE_PARAMS)
    generate_demandforecast(YEARS, ZONES)
    generate_demandprofile(ZONES)
    generate_renewable_profiles(ZONES)
    generate_storage_constants()
    generate_storage_costs()


if __name__ == "__main__":
    main()
