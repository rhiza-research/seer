import pandas as pd

from .abstract_model import seer_base_model


def load_data(data_path):
    """Read the seer-schema input CSVs at `data_path` and build a Pyomo AbstractModel data dict.
    data_path may be a local path or a remote URL (e.g. gs://bucket/prefix); reads go through
    pandas/fsspec either way."""
    data_path = str(data_path).rstrip("/")

    years_df = pd.read_csv(f"{data_path}/years.csv")
    settings_df = pd.read_csv(f"{data_path}/settings.csv")
    zones_df = pd.read_csv(f"{data_path}/zones.csv")
    generator_constants_df = pd.read_csv(f"{data_path}/generator_constants.csv")
    generator_costs_df = pd.read_csv(f"{data_path}/generator_costs.csv")
    demandforecast_df = pd.read_csv(f"{data_path}/demandforecast.csv")
    demandprofile_df = pd.read_csv(f"{data_path}/demandprofile.csv")
    renewable_profiles_df = pd.read_csv(f"{data_path}/renewable_profiles.csv")
    storage_constants_df = pd.read_csv(f"{data_path}/storage_constants.csv")
    storage_costs_df = pd.read_csv(f"{data_path}/storage_costs.csv")

    years = years_df["year"].tolist()
    zones = zones_df["zone"].tolist()
    generators = generator_constants_df["generator"].tolist()  # local convenience list, not passed to Pyomo directly

    # discount rate is a single scalar setting; the per-year factor is computed here
    discount_rate = settings_df.set_index("parameters")["value"]["dr"]
    base_year = years[0]
    year_discount_factor = {year: 1 / (1 + discount_rate) ** (year - base_year) for year in years}

    # hourly index: derived from the demand profile's per-zone row count (e.g. 365 days * 24 hours)
    hours_per_zone = len(demandprofile_df) // len(zones)
    hours = list(range(1, hours_per_zone + 1))

    # ---- demand: annual peak (demandforecast) x normalized hourly shape (demandprofile) ----
    demand_shape = {
        (zone, hour): value
        for zone, group in demandprofile_df.groupby("zone", sort=False)
        for hour, value in zip(hours, group["value"])
    }
    peak_demand = {
        (row.zone, int(row.year)): row.value
        for row in demandforecast_df.itertuples()
    }
    demand = {
        (zone, year, hour): peak_demand[(zone, year)] * demand_shape[(zone, hour)]
        for zone in zones
        for year in years
        for hour in hours
    }

    # ---- renewable availability, keyed by (zone, tech) ----
    renewable_availability = {
        (zone, tech, hour): value
        for (zone, tech), group in renewable_profiles_df.groupby(["zone", "tech"], sort=False)
        for hour, value in zip(hours, group["value"])
    }

    # ---- generator attributes ----
    constants = generator_constants_df.set_index("generator")
    existing_generators = constants[~constants["expandable"]].index.tolist()
    new_generators = constants[constants["expandable"]].index.tolist()

    generator_zone = constants["zone"].to_dict()
    generator_tech = constants["tech"].to_dict()
    generator_retech = constants["RETech"].astype(bool).to_dict()

    # only meaningful (and only loaded) for existing generators
    generator_capacity = constants.loc[existing_generators, "capacity"].to_dict()
    generator_start_year = constants.loc[existing_generators, "start_year"].to_dict()
    generator_retirement_year = constants.loc[existing_generators, "retirement_year"].to_dict()
    # only meaningful (and only loaded) for new-build generators
    generator_annual_build_limit = constants.loc[new_generators, "annual_build_limit"].to_dict()
    generator_unit_size = constants.loc[new_generators, "unit_size"].to_dict()

    # dispatchable generators have a capacity factor of 1.0 every hour; renewables are derated by their zone/tech profile
    generator_capacity_factor = {
        (gen, hour): (
            renewable_availability[(generator_zone[gen], generator_tech[gen], hour)]
            if generator_retech[gen] else 1.0
        )
        for gen in generators
        for hour in hours
    }

    # ---- costs ----
    costs = generator_costs_df.set_index(["generator", "cost_type"])
    generator_capex_cost = {
        (gen, year): costs.loc[(gen, "capex"), str(year)]
        for gen in new_generators
        for year in years
    }
    generator_marginal_cost = {
        (gen, year): costs.loc[(gen, "marginal_cost"), str(year)]
        for gen in generators
        for year in years
    }

    # ---- storage attributes ----
    storage_constants = storage_constants_df.set_index("storage")
    storage = storage_constants.index.tolist()  # local convenience list, not passed to Pyomo directly
    existing_storage = storage_constants[~storage_constants["expandable"]].index.tolist()
    new_storage = storage_constants[storage_constants["expandable"]].index.tolist()

    storage_zone = storage_constants["zone"].to_dict()
    storage_carrier = storage_constants["carrier"].to_dict()
    storage_duration_hours = storage_constants["duration_hours"].to_dict()

    # only meaningful (and only loaded) for existing storage
    storage_capacity_mwh = storage_constants.loc[existing_storage, "capacity_mwh"].to_dict()
    storage_start_year = storage_constants.loc[existing_storage, "start_year"].to_dict()
    storage_retirement_year = storage_constants.loc[existing_storage, "retirement_year"].to_dict()
    # only meaningful (and only loaded) for new-build storage
    storage_annual_build_limit_mwh = storage_constants.loc[new_storage, "annual_build_limit_mwh"].to_dict()

    # ---- storage costs ----
    storage_costs = storage_costs_df.set_index(["storage", "cost_type"])
    storage_capex_cost = {
        (s, year): storage_costs.loc[(s, "capex"), str(year)]
        for s in new_storage
        for year in years
    }
    storage_marginal_cost = {
        (s, year): storage_costs.loc[(s, "marginal_cost"), str(year)]
        for s in storage
        for year in years
    }

    return {
        None: {
            "zones": {None: zones},
            "years": {None: years},
            "hours": {None: hours},
            "year_discount_factor": year_discount_factor,
            "existing_generators": {None: existing_generators},
            "new_generators": {None: new_generators},
            "generator_zone": generator_zone,
            "generator_tech": generator_tech,
            "generator_capacity": generator_capacity,
            "generator_start_year": generator_start_year,
            "generator_retirement_year": generator_retirement_year,
            "generator_annual_build_limit": generator_annual_build_limit,
            "generator_unit_size": generator_unit_size,
            "generator_capex_cost": generator_capex_cost,
            "generator_marginal_cost": generator_marginal_cost,
            "generator_capacity_factor": generator_capacity_factor,
            "existing_storage": {None: existing_storage},
            "new_storage": {None: new_storage},
            "storage_zone": storage_zone,
            "storage_carrier": storage_carrier,
            "storage_duration_hours": storage_duration_hours,
            "storage_capacity_mwh": storage_capacity_mwh,
            "storage_start_year": storage_start_year,
            "storage_retirement_year": storage_retirement_year,
            "storage_annual_build_limit_mwh": storage_annual_build_limit_mwh,
            "storage_capex_cost": storage_capex_cost,
            "storage_marginal_cost": storage_marginal_cost,
            "demand": demand,
        }
    }


def build(data_path, discrete_additions=False, storage_daily=False, storage_reset_hour=0, storage_reset_soc=0.5):
    """Build a concrete Pyomo model instance from seer-schema CSVs at `data_path`."""
    model = seer_base_model(
        discrete_additions=discrete_additions,
        storage_daily=storage_daily,
        storage_reset_hour=storage_reset_hour,
        storage_reset_soc=storage_reset_soc,
    )
    return model.create_instance(load_data(data_path))
