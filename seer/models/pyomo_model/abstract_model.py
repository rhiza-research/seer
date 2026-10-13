import pyomo.environ as pyo

"""Specifies a capacity planning model. Features of the model are:
- hourly resolution dispatch; annual build decisions.
"""

def seer_base_model(discrete_additions=False, battery_reset="annual", storage_reset_hour=0, storage_reset_soc=0.5):
    # None = no periodic reset within a year, i.e. only at year start ("annual")
    reset_periods_hours = {"annual": None, "daily": 24, "weekly": 24 * 7}
    if battery_reset not in reset_periods_hours:
        raise ValueError(f"battery_reset must be 'annual', 'daily', or 'weekly', got {battery_reset!r}")
    reset_period_hours = reset_periods_hours[battery_reset]
    assert 0 <= storage_reset_hour <= 23, "storage_reset_hour must be between 0 and 23"
    assert 0 <= storage_reset_soc <= 1, "storage_reset_soc must be between 0 and 1"

    # define a capacity planning model
    model = pyo.AbstractModel()

    """Indices"""
    # spatial
    model.zones = pyo.Set()
    # temporal. hours is a 1-indexed within-year position (see seer/schema/README.md's
    # times.csv), ranging over the longest modeled year -- times is NOT a strict years x hours
    # cross product, since a time can be skipped in a given year (e.g. Feb 29 in a non-leap
    # year), so it's supplied explicitly as its own (year, hour) pairs rather than derived.
    model.years = pyo.Set(ordered=True)
    model.hours = pyo.Set(ordered=True)
    model.times = pyo.Set(ordered=True, dimen=2)
    # generators - existing and new build generators together comprise all generators
    model.existing_generators = pyo.Set()
    model.new_generators = pyo.Set()
    model.generators = model.existing_generators | model.new_generators
    # storage - existing and new build storage together comprise all storage
    model.existing_storage = pyo.Set()
    model.new_storage = pyo.Set()
    model.storage = model.existing_storage | model.new_storage
    # network - transmission lines connecting zones
    model.lines = pyo.Set()

    """Parameters"""
    # generator attributes
    model.generator_zone = pyo.Param(model.generators, within=pyo.Any)
    model.generator_tech = pyo.Param(model.generators, within=pyo.Any)

    # existing generators have a fixed capacity, between start and retirement year
    model.generator_capacity = pyo.Param(model.existing_generators, within=pyo.NonNegativeReals)
    model.generator_start_year = pyo.Param(model.existing_generators, within=pyo.Reals)
    model.generator_retirement_year = pyo.Param(model.existing_generators, within=pyo.Reals)
    # new build generators have an annual build limit
    model.generator_annual_build_limit = pyo.Param(model.new_generators, within=pyo.NonNegativeReals)

    # generator costs
    model.generator_marginal_cost = pyo.Param(model.generators, model.years, within=pyo.NonNegativeReals)
    model.generator_capex_cost = pyo.Param(model.new_generators, model.years, within=pyo.NonNegativeReals)

    # capacity factor: 1.0 for dispatchable generators, per-(year, hour) factor for renewables
    # (renewable_profiles.csv may be a template reused every year, or vary by year -- see
    # seer.models.time_utils.resolve_profile)
    model.generator_capacity_factor = pyo.Param(model.generators, model.times, within=pyo.NonNegativeReals)

    # storage attributes
    model.storage_zone = pyo.Param(model.storage, within=pyo.Any)
    model.storage_tech = pyo.Param(model.storage, within=pyo.Any)
    # duration (energy/power ratio, hours) is fixed by technology; power capacity is derived from it, never expanded independently
    model.storage_duration_hours = pyo.Param(model.storage, within=pyo.PositiveReals)

    # existing storage has a fixed energy capacity, between start and retirement year
    model.storage_capacity_mwh = pyo.Param(model.existing_storage, within=pyo.NonNegativeReals)
    model.storage_start_year = pyo.Param(model.existing_storage, within=pyo.Reals)
    model.storage_retirement_year = pyo.Param(model.existing_storage, within=pyo.Reals)
    # new build storage has an annual (energy) build limit
    model.storage_annual_build_limit_mwh = pyo.Param(model.new_storage, within=pyo.NonNegativeReals)

    # storage costs
    model.storage_marginal_cost = pyo.Param(model.storage, model.years, within=pyo.NonNegativeReals)
    model.storage_capex_cost = pyo.Param(model.new_storage, model.years, within=pyo.NonNegativeReals)

    # network attributes - fixed capacity lines between two zones; no line expansion (yet)
    model.line_from = pyo.Param(model.lines, within=pyo.Any)
    model.line_to = pyo.Param(model.lines, within=pyo.Any)
    model.line_capacity = pyo.Param(model.lines, within=pyo.NonNegativeReals)

    # demand - defined at each time in each zone
    model.demand = pyo.Param(model.zones, model.times, within=pyo.NonNegativeReals)
    # the real timestamp each (year, hour) represents (see seer.models.time_utils) - carried
    # only for output tables to report a real datetime, not used by any constraint/objective
    model.snapshot_datetime = pyo.Param(model.times, within=pyo.Any)
    # discount factor relative to the first modeled year, applied to every cost term in the objective
    model.year_discount_factor = pyo.Param(model.years, within=pyo.NonNegativeReals)
    # value of lost load ($/MWh); kept high so lost load is only used when truly necessary
    # TODO: why is this intialized
    model.voll = pyo.Param(within=pyo.NonNegativeReals, initialize=1_000_000.0, mutable=True)

    """Variables"""
    model.generation_output = pyo.Var(model.generators, model.times, within=pyo.NonNegativeReals)
    # capacity added in a given year (persists into all subsequent years); only new-build generators can add capacity
    if discrete_additions:
        # additions must come in whole unit_size increments: units is the decision variable,
        # generation_added_capacity is just units * unit_size, used everywhere else unchanged.
        # unit_size is only declared (and only needs data) here -- most raw source data has no
        # UnitSize value at all for most generators, which is fine as long as this mode isn't
        # requested (see seer.models.pyomo_model.build.load_data).
        model.generator_unit_size = pyo.Param(model.new_generators, within=pyo.PositiveReals)
        model.generation_added_units = pyo.Var(model.new_generators, model.years, within=pyo.NonNegativeIntegers)

        def added_capacity_rule(model, gen, year):
            return model.generation_added_units[gen, year] * model.generator_unit_size[gen]

        model.generation_added_capacity = pyo.Expression(model.new_generators, model.years, rule=added_capacity_rule)
    else:
        model.generation_added_capacity = pyo.Var(model.new_generators, model.years, within=pyo.NonNegativeReals)
    model.total_capacity = pyo.Var(model.generators, model.years, within=pyo.NonNegativeReals)
    # unserved demand; keeps the model feasible even when capacity/build limits can't otherwise meet demand
    model.lost_load = pyo.Var(model.zones, model.times, within=pyo.NonNegativeReals)

    # storage: signed net change (positive = charging/drawing from the grid, negative = discharging/injecting to it)
    model.storage_change = pyo.Var(model.storage, model.times, within=pyo.Reals)
    model.soc = pyo.Var(model.storage, model.times, within=pyo.NonNegativeReals)
    # energy capacity added in a given year (persists into all subsequent years); only new-build storage can add capacity
    model.storage_added_capacity = pyo.Var(model.new_storage, model.years, within=pyo.NonNegativeReals)
    model.storage_total_energy_capacity = pyo.Var(model.storage, model.years, within=pyo.NonNegativeReals)

    # signed line flow (positive = flowing from line_from to line_to); bounded by capacity below
    model.line_flow = pyo.Var(model.lines, model.times, within=pyo.Reals)

    # power capacity is derived from energy capacity via the fixed duration ratio, never an independent decision
    def storage_power_capacity_rule(model, s, year):
        return model.storage_total_energy_capacity[s, year] / model.storage_duration_hours[s]

    model.storage_total_power_capacity = pyo.Expression(model.storage, model.years, rule=storage_power_capacity_rule)

    """Objective"""
    def objective_rule(model):
        # total cost objective, discounted to the first modeled year
        opex = sum(
            model.year_discount_factor[year]
            * model.generator_marginal_cost[gen, year] * model.generation_output[gen, year, hour]
            for gen in model.generators
            for year, hour in model.times
        )
        capex = sum(
            model.year_discount_factor[year]
            * model.generator_capex_cost[gen, year] * model.generation_added_capacity[gen, year]
            for gen in model.new_generators
            for year in model.years
        )
        lost_load_cost = sum(
            model.year_discount_factor[year] * model.voll * model.lost_load[zone, year, hour]
            for zone in model.zones
            for year, hour in model.times
        )
        storage_capex = sum(
            model.year_discount_factor[year]
            * model.storage_capex_cost[s, year] * model.storage_added_capacity[s, year]
            for s in model.new_storage
            for year in model.years
        )
        # storage_marginal_cost is 0 for now; not yet wired to a throughput term (see storage_change's signed
        # definition) - left out of the objective until that's decided, since it would currently be a no-op anyway
        return opex + capex + lost_load_cost + storage_capex

    # least cost objective
    model.objective = pyo.Objective(
        rule=objective_rule,
        sense=pyo.minimize
    )


    """Constraints"""

    # 1. supply + storage discharge + net line imports = demand + storage charge + lost load, in each zone
    def supply_equals_demand_rule(model, zone, year, hour):
        generation = sum(
            model.generation_output[gen, year, hour]
            for gen in model.generators
            if model.generator_zone[gen] == zone
        )
        storage_net_supply = sum(
            -model.storage_change[s, year, hour]
            for s in model.storage
            if model.storage_zone[s] == zone
        )
        # a line's flow is positive from line_from -> line_to: that's an import for line_to,
        # an export (negative import) for line_from
        net_line_imports = sum(
            model.line_flow[line, year, hour]
            for line in model.lines
            if model.line_to[line] == zone
        ) - sum(
            model.line_flow[line, year, hour]
            for line in model.lines
            if model.line_from[line] == zone
        )
        return (
            generation + storage_net_supply + net_line_imports + model.lost_load[zone, year, hour]
            == model.demand[zone, year, hour]
        )

    model.supply_equals_demand = pyo.Constraint(
        model.zones,
        model.times,
        rule=supply_equals_demand_rule
    )

    # 2. capacity added in a year is bounded by the annual build limit (new-build generators only)
    def build_limit_rule(model, gen, year):
        return model.generation_added_capacity[gen, year] <= model.generator_annual_build_limit[gen]

    model.build_limit = pyo.Constraint(
        model.new_generators,
        model.years,
        rule=build_limit_rule
    )

    # 3. total capacity of new generators is the sum of all capacity added up to and including this year
    def capacity_accumulation_rule(model, gen, year):
        prior_and_current_years = [
            y for y in model.years if model.years.ord(y) <= model.years.ord(year)
        ]
        return model.total_capacity[gen, year] == sum(
            model.generation_added_capacity[gen, y] for y in prior_and_current_years
        )

    model.capacity_accumulation_constraint = pyo.Constraint(
        model.new_generators,
        model.years,
        rule=capacity_accumulation_rule
    )

    # 4. capacity of generators depends on whether they are active
    def existing_capacity_rule(model, gen, year):
        is_active = pyo.value(model.generator_start_year[gen]) <= year <= pyo.value(model.generator_retirement_year[gen])
        return model.total_capacity[gen, year] == (model.generator_capacity[gen] if is_active else 0)

    model.existing_capacity_constraint = pyo.Constraint(
        model.existing_generators,
        model.years,
        rule=existing_capacity_rule
    )

    # 5. generator output is limited by capacity
    def generator_output_limit_rule(model, gen, year, hour):
        return model.generation_output[gen, year, hour] <= (
            model.total_capacity[gen, year] * model.generator_capacity_factor[gen, year, hour]
        )

    model.generator_output_limit = pyo.Constraint(
        model.generators,
        model.times,
        rule=generator_output_limit_rule
    )

    # 6. energy capacity added in a year is bounded by the annual build limit (new-build storage only)
    def storage_build_limit_rule(model, s, year):
        return model.storage_added_capacity[s, year] <= model.storage_annual_build_limit_mwh[s]

    model.storage_build_limit = pyo.Constraint(
        model.new_storage,
        model.years,
        rule=storage_build_limit_rule
    )

    # 7. total energy capacity of new storage is the sum of all capacity added up to and including this year
    def storage_capacity_accumulation_rule(model, s, year):
        prior_and_current_years = [
            y for y in model.years if model.years.ord(y) <= model.years.ord(year)
        ]
        return model.storage_total_energy_capacity[s, year] == sum(
            model.storage_added_capacity[s, y] for y in prior_and_current_years
        )

    model.storage_capacity_accumulation_constraint = pyo.Constraint(
        model.new_storage,
        model.years,
        rule=storage_capacity_accumulation_rule
    )

    # 8. energy capacity of existing storage depends on whether it is active
    def storage_existing_capacity_rule(model, s, year):
        is_active = pyo.value(model.storage_start_year[s]) <= year <= pyo.value(model.storage_retirement_year[s])
        return model.storage_total_energy_capacity[s, year] == (model.storage_capacity_mwh[s] if is_active else 0)

    model.storage_existing_capacity_constraint = pyo.Constraint(
        model.existing_storage,
        model.years,
        rule=storage_existing_capacity_rule
    )

    # 9. state of charge (MWh) can never exceed the storage's total energy capacity
    def storage_soc_limit_rule(model, s, year, hour):
        return model.soc[s, year, hour] <= model.storage_total_energy_capacity[s, year]

    model.storage_soc_limit = pyo.Constraint(
        model.storage,
        model.times,
        rule=storage_soc_limit_rule
    )

    # 10. charge/discharge rate (MW) is bounded by the storage's total power capacity. Two one-sided
    # constraints rather than a single ranged one, since the bound itself (power capacity) depends on a
    # variable (energy capacity) and Pyomo's ranged/chained inequalities require constant bounds.
    def storage_charge_limit_rule(model, s, year, hour):
        return model.storage_change[s, year, hour] <= model.storage_total_power_capacity[s, year]

    model.storage_charge_limit = pyo.Constraint(
        model.storage,
        model.times,
        rule=storage_charge_limit_rule
    )

    def storage_discharge_limit_rule(model, s, year, hour):
        return model.storage_change[s, year, hour] >= -model.storage_total_power_capacity[s, year]

    model.storage_discharge_limit = pyo.Constraint(
        model.storage,
        model.times,
        rule=storage_discharge_limit_rule
    )

    # 11. state of charge evolution. Years are non-contiguous representative periods, so SOC is always
    # reset/anchored at the start of every year; battery_reset="daily"/"weekly" additionally resets
    # every 24/168 hours, which also breaks the year into independent daily/weekly windows (the same
    # rule does both jobs).
    def soc_rule(model, s, year, hour):
        hour_index = model.hours.ord(hour) - 1  # 0-based position within the year
        is_year_start = hour == model.hours.first()
        is_periodic_reset = reset_period_hours is not None and (hour_index % reset_period_hours == storage_reset_hour)

        if is_year_start or is_periodic_reset:
            return model.soc[s, year, hour] == storage_reset_soc * model.storage_total_energy_capacity[s, year]

        previous_hour = model.hours.prev(hour)
        return model.soc[s, year, hour] == model.soc[s, year, previous_hour] + model.storage_change[s, year, hour]

    model.soc_constraint = pyo.Constraint(
        model.storage,
        model.times,
        rule=soc_rule
    )

    # 12. line flow (either direction) is bounded by the line's fixed capacity. Two one-sided
    # constraints rather than a ranged one, for consistency with the storage charge/discharge limits above.
    def line_flow_upper_limit_rule(model, line, year, hour):
        return model.line_flow[line, year, hour] <= model.line_capacity[line]

    model.line_flow_upper_limit = pyo.Constraint(
        model.lines,
        model.times,
        rule=line_flow_upper_limit_rule
    )

    def line_flow_lower_limit_rule(model, line, year, hour):
        return model.line_flow[line, year, hour] >= -model.line_capacity[line]

    model.line_flow_lower_limit = pyo.Constraint(
        model.lines,
        model.times,
        rule=line_flow_lower_limit_rule
    )

    return model
