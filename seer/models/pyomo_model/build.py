import pandas as pd

from ..time_utils import build_snapshot_calendar, load_times, resolve_profile
from .abstract_model import seer_base_model


def load_data(data_path, discrete_additions=False):
    """Read the seer-schema input CSVs at `data_path` and build a Pyomo AbstractModel data dict.
    data_path may be a local path or a remote URL (e.g. gs://bucket/prefix); reads go through
    pandas/fsspec either way.

    `discrete_additions` must match the value passed to seer_base_model() -- generator_unit_size
    is only declared (and only needs data) in that mode; most raw source data has no UnitSize
    value at all for most generators, which is fine as long as this mode isn't requested."""
    data_path = str(data_path).rstrip("/")

    years_df = pd.read_csv(f"{data_path}/years.csv")
    settings_df = pd.read_csv(f"{data_path}/settings.csv")
    zones_df = pd.read_csv(f"{data_path}/zones.csv")
    generator_constants_df = pd.read_csv(f"{data_path}/generator_constants.csv")
    generator_costs_df = pd.read_csv(f"{data_path}/generator_costs.csv")
    demand_forecast_df = pd.read_csv(f"{data_path}/demand_forecast.csv")
    demand_profile_df = pd.read_csv(f"{data_path}/demand_profile.csv")
    renewable_profiles_df = pd.read_csv(f"{data_path}/renewable_profiles.csv")
    storage_constants_df = pd.read_csv(f"{data_path}/storage_constants.csv")
    storage_costs_df = pd.read_csv(f"{data_path}/storage_costs.csv")
    # network.csv is optional - datasets with no inter-zone lines (e.g. single-zone) can omit it
    try:
        network_df = pd.read_csv(f"{data_path}/network.csv")
    except FileNotFoundError:
        network_df = pd.DataFrame(columns=["from", "to", "capacity"])

    years = years_df["year"].tolist()
    zones = zones_df["zone"].tolist()
    generators = generator_constants_df["generator"].tolist()  # local convenience list, not passed to Pyomo directly

    # discount rate is a single scalar setting; the per-year factor is computed here
    discount_rate = settings_df.set_index("parameters")["value"]["dr"]
    base_year = years[0]
    year_discount_factor = {year: 1 / (1 + discount_rate) ** (year - base_year) for year in years}

    # times.csv x years.csv, projected onto each real year (skipping e.g. Feb 29 in a
    # non-leap year) -- see seer/schema/README.md and seer.models.time_utils
    calendar = build_snapshot_calendar(years, load_times(data_path))
    hours = sorted(calendar["hour"].unique())
    times = list(zip(calendar["year"], calendar["hour"]))
    snapshot_datetime = {
        (row.year, row.hour): row.datetime for row in calendar.itertuples()
    }

    # ---- demand: annual peak (demand_forecast) x normalized profile shape (demand_profile) ----
    peak = demand_forecast_df[demand_forecast_df["type"] == "peak"][["zone", "year", "value"]].rename(
        columns={"value": "peak"}
    )
    demand_long = resolve_profile(demand_profile_df, calendar, key_cols=["zone"])
    demand_long = demand_long.merge(peak, on=["zone", "year"], how="left")
    demand = {
        (row.zone, row.year, row.hour): row.value * row.peak
        for row in demand_long.itertuples()
    }

    availability = {
        (row.profile_id, row.year, row.hour): row.value
        for row in resolve_profile(renewable_profiles_df, calendar, key_cols=["profile_id"]).itertuples()
    } if not renewable_profiles_df.empty else {}

    # ---- generator attributes ----
    constants = generator_constants_df.set_index("generator")
    existing_generators = constants[~constants["expandable"]].index.tolist()
    new_generators = constants[constants["expandable"]].index.tolist()

    generator_zone = constants["zone"].to_dict()
    generator_tech = constants["tech"].to_dict()
    generator_isre = constants["isre"].astype(bool).to_dict()
    generator_profile_id = constants["profile_id"].to_dict()

    # only meaningful (and only loaded) for existing generators
    generator_capacity = constants.loc[existing_generators, "capacity"].to_dict()
    generator_start_year = constants.loc[existing_generators, "start_year"].to_dict()
    generator_retirement_year = constants.loc[existing_generators, "retirement_year"].to_dict()
    # only meaningful (and only loaded) for new-build generators
    generator_annual_build_limit = constants.loc[new_generators, "annual_build_limit"].to_dict()

    # dispatchable generators have a capacity factor of 1.0 every (year, hour); renewables are
    # derated by their own profile_id's availability (each may itself vary by year). isre
    # guarantees profile_id is set -- see epm2seer.generator_params.match_re_profiles.
    def _capacity_factor(gen, year, hour):
        return availability[(generator_profile_id[gen], year, hour)]

    generator_capacity_factor = {
        (gen, year, hour): _capacity_factor(gen, year, hour) if generator_isre[gen] else 1.0
        for gen in generators
        for year, hour in times
    }

    # ---- costs ----
    costs = generator_costs_df.set_index(["generator", "cost_type"])
    generator_capex_cost = {
        (gen, year): costs.loc[(gen, "capex"), str(year)]
        for gen in new_generators
        for year in years
    }
    # 0 if a generator's marginal_cost row (or a specific year within it) is missing --
    # should always be fully present per seer/schema/README.md, but fall back defensively
    # rather than crash or pass NaN into a NonNegativeReals param
    def _marginal_cost(gen, year):
        if (gen, "marginal_cost") not in costs.index:
            return 0.0
        value = costs.loc[(gen, "marginal_cost"), str(year)]
        return 0.0 if pd.isna(value) else value

    generator_marginal_cost = {(gen, year): _marginal_cost(gen, year) for gen in generators for year in years}

    # ---- storage attributes ----
    storage_constants = storage_constants_df.set_index("storage")
    storage = storage_constants.index.tolist()  # local convenience list, not passed to Pyomo directly
    existing_storage = storage_constants[~storage_constants["expandable"]].index.tolist()
    new_storage = storage_constants[storage_constants["expandable"]].index.tolist()

    storage_zone = storage_constants["zone"].to_dict()
    storage_tech = storage_constants["tech"].to_dict()
    storage_duration_hours = storage_constants["duration_hours"].to_dict()

    # only meaningful (and only loaded) for existing storage. capacity_mwh isn't a stored
    # column -- it's derived from capacity_mw * duration_hours wherever it's needed.
    storage_capacity_mwh = (
        storage_constants["capacity_mw"] * storage_constants["duration_hours"]
    ).loc[existing_storage].to_dict()
    storage_start_year = storage_constants.loc[existing_storage, "start_year"].to_dict()
    storage_retirement_year = storage_constants.loc[existing_storage, "retirement_year"].to_dict()
    # only meaningful (and only loaded) for new-build storage
    storage_annual_build_limit_mwh = storage_constants.loc[new_storage, "annual_build_limit_mwh"].to_dict()

    # ---- storage costs ----
    storage_costs = storage_costs_df.set_index(["storage", "cost_type"])

    # 0 if a storage unit's capex/marginal_cost row (or a specific year within it) is
    # missing -- should always be fully present per seer/schema/README.md, but fall back
    # defensively rather than crash or pass NaN into a NonNegativeReals param
    def _storage_cost(s, cost_type, year):
        if (s, cost_type) not in storage_costs.index:
            return 0.0
        value = storage_costs.loc[(s, cost_type), str(year)]
        return 0.0 if pd.isna(value) else value

    storage_capex_cost = {
        (s, year): _storage_cost(s, "capex", year)
        for s in new_storage
        for year in years
    }
    storage_marginal_cost = {
        (s, year): _storage_cost(s, "marginal_cost", year)
        for s in storage
        for year in years
    }

    # ---- network attributes - lines identified by row position, since (from, to) isn't guaranteed unique ----
    lines = network_df.index.tolist()
    line_from = network_df["from"].to_dict()
    line_to = network_df["to"].to_dict()
    line_capacity = network_df["capacity"].to_dict()

    data = {
        None: {
            "zones": {None: zones},
            "years": {None: years},
            "hours": {None: hours},
            "times": {None: times},
            "snapshot_datetime": snapshot_datetime,
            "year_discount_factor": year_discount_factor,
            "existing_generators": {None: existing_generators},
            "new_generators": {None: new_generators},
            "generator_zone": generator_zone,
            "generator_tech": generator_tech,
            "generator_capacity": generator_capacity,
            "generator_start_year": generator_start_year,
            "generator_retirement_year": generator_retirement_year,
            "generator_annual_build_limit": generator_annual_build_limit,
            "generator_capex_cost": generator_capex_cost,
            "generator_marginal_cost": generator_marginal_cost,
            "generator_capacity_factor": generator_capacity_factor,
            "existing_storage": {None: existing_storage},
            "new_storage": {None: new_storage},
            "storage_zone": storage_zone,
            "storage_tech": storage_tech,
            "storage_duration_hours": storage_duration_hours,
            "storage_capacity_mwh": storage_capacity_mwh,
            "storage_start_year": storage_start_year,
            "storage_retirement_year": storage_retirement_year,
            "storage_annual_build_limit_mwh": storage_annual_build_limit_mwh,
            "storage_capex_cost": storage_capex_cost,
            "storage_marginal_cost": storage_marginal_cost,
            "lines": {None: lines},
            "line_from": line_from,
            "line_to": line_to,
            "line_capacity": line_capacity,
            "demand": demand,
        }
    }
    if discrete_additions:
        data[None]["generator_unit_size"] = constants.loc[new_generators, "unit_size"].to_dict()
    return data


def build(data_path, discrete_additions=False, battery_reset="annual", storage_reset_hour=0, storage_reset_soc=0.5):
    """Build a concrete Pyomo model instance from seer-schema CSVs at `data_path`."""
    model = seer_base_model(
        discrete_additions=discrete_additions,
        battery_reset=battery_reset,
        storage_reset_hour=storage_reset_hour,
        storage_reset_soc=storage_reset_soc,
    )
    return model.create_instance(load_data(data_path, discrete_additions=discrete_additions))
