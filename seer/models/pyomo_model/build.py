import pandas as pd

from ..utils.time_utils import build_snapshot_calendar, compute_time_weights, load_time_weights, load_times, resolve_profile
from .abstract_model import seer_base_model


def annualized_capex_coefficient(capex, build_year, life, years, discount_rate, year_discount_factor):
    """Capex coefficient (per unit added in `build_year`) that makes the objective's
    `year_discount_factor[build_year] * coefficient` equal to annualized payments instead of a
    one-off total: capex x CRF(discount_rate, life) charged in every modeled year the vintage is
    active (build_year <= year < build_year + life), each discounted to the first modeled year.
    Payments past the horizon are never charged -- the end-of-horizon treatment EPM/PyPSA use."""
    r = discount_rate
    crf = 1 / life if r == 0 else r / (1 - (1 + r) ** -life)
    active = [y for y in years if build_year <= y < build_year + life]
    return capex * crf * sum(year_discount_factor[y] for y in active) / year_discount_factor[build_year]


def load_data(data_path, discrete_additions=False, annualize_capex=False):
    """Read the seer-schema input CSVs at `data_path` and build a Pyomo AbstractModel data dict.
    data_path may be a local path or a remote URL (e.g. gs://bucket/prefix); reads go through
    pandas/fsspec either way.

    `discrete_additions` must match the value passed to seer_base_model() -- generator_unit_size
    is only declared (and only needs data) in that mode; most raw source data has no UnitSize
    value at all for most generators, which is fine as long as this mode isn't requested.

    `annualize_capex`: the schema's capex is a total cost incurred once in the build year (the
    default here). True instead charges it as annual payments over the asset's `life`, only for
    modeled years inside the horizon (see annualized_capex_coefficient) -- removes the
    end-of-horizon bias against late builds. Requires `life` on every build option (already
    required by the schema)."""
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
    # non-leap year) -- see seer/schema/README.md and seer.models.utils.time_utils
    calendar = build_snapshot_calendar(years, load_times(data_path), load_time_weights(data_path))
    hours = sorted(calendar["hour"].unique())
    times = list(zip(calendar["year"], calendar["hour"]))
    snapshot_datetime = {
        (row.year, row.hour): row.datetime for row in calendar.itertuples()
    }
    time_weight = compute_time_weights(calendar)

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
    generator_life = constants.loc[existing_generators, "life"].to_dict()
    # only meaningful (and only loaded) for new-build generators
    # build limits are optional in the schema -- absent (blank, or no column at all) means unconstrained
    def _build_limit(col):
        values = constants[col] if col in constants else pd.Series(float("nan"), index=constants.index)
        return values.loc[new_generators].astype(float).fillna(float("inf")).to_dict()
    generator_annual_build_limit = _build_limit("annual_build_limit")
    generator_total_build_limit = _build_limit("total_build_limit")
    # earliest build year for build options
    generator_build_start_year = constants.loc[new_generators, "start_year"].to_dict()
    generator_build_life = constants.loc[new_generators, "life"].to_dict()

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
        (gen, year): costs.loc[(gen, "capital"), str(year)]
        for gen in new_generators
        for year in years
    }
    if annualize_capex:
        generator_capex_cost = {
            (gen, year): annualized_capex_coefficient(
                value, year, constants.loc[gen, "life"], years, discount_rate, year_discount_factor
            )
            for (gen, year), value in generator_capex_cost.items()
        }
    # 0 if a generator's marginal_cost row (or a specific year within it) is missing --
    # should always be fully present per seer/schema/README.md, but fall back defensively
    # rather than crash or pass NaN into a NonNegativeReals param
    def _marginal_cost(gen, year):
        if (gen, "marginal") not in costs.index:
            return 0.0
        value = costs.loc[(gen, "marginal"), str(year)]
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
    storage_life = storage_constants.loc[existing_storage, "life"].to_dict()
    # only meaningful (and only loaded) for new-build storage
    # build limits are optional -- absent (blank, or no column at all) means unconstrained
    def _storage_build_limit(col):
        values = storage_constants[col] if col in storage_constants else pd.Series(float("nan"), index=storage_constants.index)
        return values.loc[new_storage].astype(float).fillna(float("inf")).to_dict()
    storage_annual_build_limit_mwh = _storage_build_limit("annual_build_limit_mwh")
    storage_total_build_limit_mwh = _storage_build_limit("total_build_limit_mwh")
    storage_build_start_year = storage_constants.loc[new_storage, "start_year"].to_dict()
    storage_build_life = storage_constants.loc[new_storage, "life"].to_dict()

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
        (s, year): _storage_cost(s, "capital", year)
        for s in new_storage
        for year in years
    }
    if annualize_capex:
        storage_capex_cost = {
            (s, year): annualized_capex_coefficient(
                value, year, storage_constants.loc[s, "life"], years, discount_rate, year_discount_factor
            )
            for (s, year), value in storage_capex_cost.items()
        }
    storage_marginal_cost = {
        (s, year): _storage_cost(s, "marginal", year)
        for s in storage
        for year in years
    }

    # ---- network attributes - lines identified by row position ----
    lines = network_df.index.tolist()
    line_from = network_df["from"].to_dict()
    line_to = network_df["to"].to_dict()
    line_capacity = network_df["capacity"].fillna(0.0).to_dict()
    # expansion columns are optional -- a network.csv without them has no expandable lines
    def _network_col(col, default):
        return network_df[col] if col in network_df else pd.Series(default, index=network_df.index)
    line_expandable = _network_col("expandable", False).fillna(False).astype(bool)
    new_lines = network_df.index[line_expandable].tolist()
    line_build_start_year = _network_col("start_year", float("nan")).loc[new_lines].astype(float).fillna(0).to_dict()
    line_total_build_limit = (
        _network_col("total_build_limit", float("nan")).loc[new_lines].astype(float).fillna(float("inf")).to_dict()
    )
    line_capex = _network_col("capital", 0.0).loc[new_lines].astype(float).fillna(0.0)
    # lines have no life in the schema -- under annualize_capex they're treated as never
    # retiring: a perpetual annual payment of capex x discount_rate (CRF as life -> inf)
    line_capex_cost = {
        (line, year): (
            annualized_capex_coefficient(line_capex[line], year, float("inf"), years, discount_rate, year_discount_factor)
            if annualize_capex and discount_rate > 0 else line_capex[line]
        )
        for line in new_lines
        for year in years
    }

    data = {
        None: {
            "zones": {None: zones},
            "years": {None: years},
            "hours": {None: hours},
            "times": {None: times},
            "snapshot_datetime": snapshot_datetime,
            "time_weight": time_weight,
            "year_discount_factor": year_discount_factor,
            "existing_generators": {None: existing_generators},
            "new_generators": {None: new_generators},
            "generator_zone": generator_zone,
            "generator_tech": generator_tech,
            "generator_capacity": generator_capacity,
            "generator_start_year": generator_start_year,
            "generator_life": generator_life,
            "generator_annual_build_limit": generator_annual_build_limit,
            "generator_total_build_limit": generator_total_build_limit,
            "generator_build_start_year": generator_build_start_year,
            "generator_build_life": generator_build_life,
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
            "storage_life": storage_life,
            "storage_annual_build_limit_mwh": storage_annual_build_limit_mwh,
            "storage_total_build_limit_mwh": storage_total_build_limit_mwh,
            "storage_build_start_year": storage_build_start_year,
            "storage_build_life": storage_build_life,
            "storage_capex_cost": storage_capex_cost,
            "storage_marginal_cost": storage_marginal_cost,
            "lines": {None: lines},
            "line_from": line_from,
            "line_to": line_to,
            "line_capacity": line_capacity,
            "new_lines": {None: new_lines},
            "line_build_start_year": line_build_start_year,
            "line_total_build_limit": line_total_build_limit,
            "line_capex_cost": line_capex_cost,
            "demand": demand,
        }
    }
    if discrete_additions:
        data[None]["generator_unit_size"] = constants.loc[new_generators, "unit_size"].to_dict()
        data[None]["line_unit_size"] = _network_col("unit_size", float("nan")).loc[new_lines].to_dict()
    return data


def build(data_path, discrete_additions=False, battery_reset="daily", annualize_capex=False):
    """Build a concrete Pyomo model instance from seer-schema CSVs at `data_path`.
    battery_reset: see seer_base_model. annualize_capex: see load_data."""
    model = seer_base_model(discrete_additions=discrete_additions, battery_reset=battery_reset)
    return model.create_instance(
        load_data(data_path, discrete_additions=discrete_additions, annualize_capex=annualize_capex)
    )
