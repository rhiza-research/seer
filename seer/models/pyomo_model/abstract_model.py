import pandas as pd
import pyomo.environ as pyo

"""Specifies a capacity planning model. Features of the model are:
- hourly resolution dispatch; annual build decisions.
"""

def seer_base_model(discrete_additions=False, battery_reset="daily"):
    """battery_reset: the window storage state of charge must balance over -- "daily" (default:
    each day ends at the level it started, which the model chooses; needed for weighted
    representative days, see times.csv `weight`) or "annual" (once per modeled year)."""
    if battery_reset not in ("daily", "annual"):
        raise ValueError(f"battery_reset must be 'daily' or 'annual', got {battery_reset!r}")

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
    # lines whose capacity can be expanded (a subset of lines)
    model.new_lines = pyo.Set(within=model.lines)

    """Parameters"""
    # generator attributes
    model.generator_zone = pyo.Param(model.generators, within=pyo.Any)
    model.generator_tech = pyo.Param(model.generators, within=pyo.Any)

    # existing generators have a fixed capacity, active from start_year for life years
    model.generator_capacity = pyo.Param(model.existing_generators, within=pyo.NonNegativeReals)
    model.generator_start_year = pyo.Param(model.existing_generators, within=pyo.Reals)
    model.generator_life = pyo.Param(model.existing_generators, within=pyo.PositiveReals)
    # new build generators have an annual and a whole-horizon total build limit (inf = unconstrained)
    model.generator_annual_build_limit = pyo.Param(model.new_generators, within=pyo.NonNegativeReals)
    model.generator_total_build_limit = pyo.Param(model.new_generators, within=pyo.NonNegativeReals)
    # ...and can't be built before their start_year (default 0: no restriction)
    model.generator_build_start_year = pyo.Param(model.new_generators, within=pyo.Reals, default=0)
    # ...and each year's additions retire after life years
    model.generator_build_life = pyo.Param(model.new_generators, within=pyo.PositiveReals)

    # generator costs
    model.generator_marginal_cost = pyo.Param(model.generators, model.years, within=pyo.NonNegativeReals)
    model.generator_capex_cost = pyo.Param(model.new_generators, model.years, within=pyo.NonNegativeReals)

    # capacity factor: 1.0 for dispatchable generators, per-(year, hour) factor for renewables
    # (renewable_profiles.csv may be a template reused every year, or vary by year -- see
    # seer.models.utils.time_utils.resolve_profile)
    model.generator_capacity_factor = pyo.Param(model.generators, model.times, within=pyo.NonNegativeReals)

    # storage attributes
    model.storage_zone = pyo.Param(model.storage, within=pyo.Any)
    model.storage_tech = pyo.Param(model.storage, within=pyo.Any)
    # duration (energy/power ratio, hours) is fixed by technology; power capacity is derived from it, never expanded independently
    model.storage_duration_hours = pyo.Param(model.storage, within=pyo.PositiveReals)

    # existing storage has a fixed energy capacity, active from start_year for life years
    model.storage_capacity_mwh = pyo.Param(model.existing_storage, within=pyo.NonNegativeReals)
    model.storage_start_year = pyo.Param(model.existing_storage, within=pyo.Reals)
    model.storage_life = pyo.Param(model.existing_storage, within=pyo.PositiveReals)
    # new build storage has an annual (energy) build limit
    model.storage_annual_build_limit_mwh = pyo.Param(model.new_storage, within=pyo.NonNegativeReals)
    model.storage_total_build_limit_mwh = pyo.Param(model.new_storage, within=pyo.NonNegativeReals)
    model.storage_build_start_year = pyo.Param(model.new_storage, within=pyo.Reals, default=0)
    model.storage_build_life = pyo.Param(model.new_storage, within=pyo.PositiveReals)

    # storage costs
    model.storage_marginal_cost = pyo.Param(model.storage, model.years, within=pyo.NonNegativeReals)
    model.storage_capex_cost = pyo.Param(model.new_storage, model.years, within=pyo.NonNegativeReals)

    # network attributes - undirected lines between two zones: existing capacity, plus optional
    # expansion (start year, total limit, capex per MW added -- see seer/schema/README.md)
    model.line_from = pyo.Param(model.lines, within=pyo.Any)
    model.line_to = pyo.Param(model.lines, within=pyo.Any)
    model.line_capacity = pyo.Param(model.lines, within=pyo.NonNegativeReals)
    model.line_build_start_year = pyo.Param(model.new_lines, within=pyo.Reals, default=0)
    model.line_total_build_limit = pyo.Param(model.new_lines, within=pyo.NonNegativeReals)
    model.line_capex_cost = pyo.Param(model.new_lines, model.years, within=pyo.NonNegativeReals)

    # demand - defined at each time in each zone; a zone with no demand data (e.g. a pure
    # interconnection zone) has zero demand, same as PyPSA (a bus with no load)
    model.demand = pyo.Param(model.zones, model.times, within=pyo.NonNegativeReals, default=0)

    # how many real hours each modeled snapshot stands for -- 1.0 for a full 8760/8784-hour year,
    # greater than 1 for a representative-days grid (see time_utils.compute_time_weights)
    model.time_weight = pyo.Param(model.times, within=pyo.PositiveReals)
    # the real timestamp each (year, hour) represents (see seer.models.utils.time_utils) - carried
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
    # line capacity added in a given year (persists for the rest of the horizon -- lines have no
    # life in the schema); whole unit_size increments under discrete_additions, like generators
    if discrete_additions:
        model.line_unit_size = pyo.Param(model.new_lines, within=pyo.PositiveReals)
        model.line_added_units = pyo.Var(model.new_lines, model.years, within=pyo.NonNegativeIntegers)
        model.line_added_capacity = pyo.Expression(
            model.new_lines, model.years,
            rule=lambda model, line, year: model.line_added_units[line, year] * model.line_unit_size[line],
        )
    else:
        model.line_added_capacity = pyo.Var(model.new_lines, model.years, within=pyo.NonNegativeReals)

    def line_total_capacity_rule(model, line, year):
        added = 0
        if line in model.new_lines:
            added = sum(model.line_added_capacity[line, y] for y in model.years if y <= year)
        return model.line_capacity[line] + added

    model.line_total_capacity = pyo.Expression(model.lines, model.years, rule=line_total_capacity_rule)

    # power capacity is derived from energy capacity via the fixed duration ratio, never an independent decision
    def storage_power_capacity_rule(model, s, year):
        return model.storage_total_energy_capacity[s, year] / model.storage_duration_hours[s]

    model.storage_total_power_capacity = pyo.Expression(model.storage, model.years, rule=storage_power_capacity_rule)

    """Objective"""
    def objective_rule(model):
        # total cost objective, discounted to the first modeled year
        opex = sum(
            model.year_discount_factor[year] * model.time_weight[year, hour]
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
            model.year_discount_factor[year] * model.time_weight[year, hour] * model.voll * model.lost_load[zone, year, hour]
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
        line_capex = sum(
            model.year_discount_factor[year]
            * model.line_capex_cost[line, year] * model.line_added_capacity[line, year]
            for line in model.new_lines
            for year in model.years
        )
        return opex + capex + lost_load_cost + storage_capex + line_capex

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

    # 2b. capacity added over the whole horizon is bounded by the total build limit (e.g. a
    # single build-once project) -- skipped where unconstrained (inf)
    def total_build_limit_rule(model, gen):
        limit = pyo.value(model.generator_total_build_limit[gen])
        if limit == float("inf"):
            return pyo.Constraint.Skip
        return sum(model.generation_added_capacity[gen, y] for y in model.years) <= limit

    model.total_build_limit = pyo.Constraint(
        model.new_generators,
        rule=total_build_limit_rule
    )

    # 2c. nothing is added before a build option's start_year (same as PyPSA, which only
    # creates build-year vintages from start_year on)
    def build_start_year_rule(model, gen, year):
        if year >= pyo.value(model.generator_build_start_year[gen]):
            return pyo.Constraint.Skip
        return model.generation_added_capacity[gen, year] == 0

    model.build_start_year = pyo.Constraint(
        model.new_generators,
        model.years,
        rule=build_start_year_rule
    )

    # 3. total capacity of new generators is the capacity added up to and including this year
    # that hasn't reached the end of its life (a build in y is active y <= year < y + life,
    # as in PyPSA)
    def capacity_accumulation_rule(model, gen, year):
        life = pyo.value(model.generator_build_life[gen])
        active_builds = [y for y in model.years if y <= year < y + life]
        return model.total_capacity[gen, year] == sum(
            model.generation_added_capacity[gen, y] for y in active_builds
        )

    model.capacity_accumulation_constraint = pyo.Constraint(
        model.new_generators,
        model.years,
        rule=capacity_accumulation_rule
    )

    # 4. capacity of generators depends on whether they are active
    def existing_capacity_rule(model, gen, year):
        start = pyo.value(model.generator_start_year[gen])
        is_active = start <= year < start + pyo.value(model.generator_life[gen])
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

    # 6a. energy capacity added over the whole horizon is bounded by the total build limit
    # (e.g. a single build-once project) -- skipped where unconstrained (inf)
    def storage_total_build_limit_rule(model, s):
        limit = pyo.value(model.storage_total_build_limit_mwh[s])
        if limit == float("inf"):
            return pyo.Constraint.Skip
        return sum(model.storage_added_capacity[s, y] for y in model.years) <= limit

    model.storage_total_build_limit = pyo.Constraint(
        model.new_storage,
        rule=storage_total_build_limit_rule
    )

    # 6b. nothing is added before a storage build option's start_year (see 2c)
    def storage_build_start_year_rule(model, s, year):
        if year >= pyo.value(model.storage_build_start_year[s]):
            return pyo.Constraint.Skip
        return model.storage_added_capacity[s, year] == 0

    model.storage_build_start_year_constraint = pyo.Constraint(
        model.new_storage,
        model.years,
        rule=storage_build_start_year_rule
    )

    # 7. total energy capacity of new storage is the capacity added up to and including this
    # year that hasn't reached the end of its life (as for generators)
    def storage_capacity_accumulation_rule(model, s, year):
        life = pyo.value(model.storage_build_life[s])
        active_builds = [y for y in model.years if y <= year < y + life]
        return model.storage_total_energy_capacity[s, year] == sum(
            model.storage_added_capacity[s, y] for y in active_builds
        )

    model.storage_capacity_accumulation_constraint = pyo.Constraint(
        model.new_storage,
        model.years,
        rule=storage_capacity_accumulation_rule
    )

    # 8. energy capacity of existing storage depends on whether it is active
    def storage_existing_capacity_rule(model, s, year):
        start = pyo.value(model.storage_start_year[s])
        is_active = start <= year < start + pyo.value(model.storage_life[s])
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

    # 11. state of charge evolution, cyclic within each window (a day, or a whole year -- see
    # battery_reset): each window's first hour follows on from that same window's last hour, so
    # storage ends every window where it started (a level the model chooses) and can't carry
    # energy between windows. With weighted representative days (times.csv `weight`), carrying
    # energy between days would let storage charge on a lightly weighted day and discharge on a
    # heavily weighted one -- creating energy in the weighted totals.
    ring_previous = {}  # (year, hour) -> previous (year, hour) in its window's ring

    def _ring_previous(model, year):
        hours = [h for (y, h) in model.times if y == year]
        if battery_reset == "annual":
            windows = [hours]
        else:
            windows, current, current_date = [], [], None
            for h in hours:
                date = pd.Timestamp(model.snapshot_datetime[year, h]).date()
                if date != current_date and current:
                    windows.append(current)
                    current = []
                current_date = date
                current.append(h)
            windows.append(current)
        for window in windows:
            for i, h in enumerate(window):
                ring_previous[(year, h)] = (year, window[i - 1])  # i=0 wraps to the window's last hour

    def soc_rule(model, s, year, hour):
        if (year, hour) not in ring_previous:
            _ring_previous(model, year)
        previous = ring_previous[(year, hour)]
        return model.soc[s, year, hour] == model.soc[s, previous[0], previous[1]] + model.storage_change[s, year, hour]

    model.soc_constraint = pyo.Constraint(
        model.storage,
        model.times,
        rule=soc_rule
    )

    # 12. line flow (either direction) is bounded by the line's fixed capacity. Two one-sided
    # constraints rather than a ranged one, for consistency with the storage charge/discharge limits above.
    def line_flow_upper_limit_rule(model, line, year, hour):
        return model.line_flow[line, year, hour] <= model.line_total_capacity[line, year]

    model.line_flow_upper_limit = pyo.Constraint(
        model.lines,
        model.times,
        rule=line_flow_upper_limit_rule
    )

    def line_flow_lower_limit_rule(model, line, year, hour):
        return model.line_flow[line, year, hour] >= -model.line_total_capacity[line, year]

    model.line_flow_lower_limit = pyo.Constraint(
        model.lines,
        model.times,
        rule=line_flow_lower_limit_rule
    )

    # 13. line expansion: nothing added before the line's start_year, and no more than its
    # total_build_limit over the whole horizon (skipped where unconstrained, inf)
    def line_build_start_year_rule(model, line, year):
        if year >= pyo.value(model.line_build_start_year[line]):
            return pyo.Constraint.Skip
        return model.line_added_capacity[line, year] == 0

    model.line_build_start_year_constraint = pyo.Constraint(
        model.new_lines,
        model.years,
        rule=line_build_start_year_rule
    )

    def line_total_build_limit_rule(model, line):
        limit = pyo.value(model.line_total_build_limit[line])
        if limit == float("inf"):
            return pyo.Constraint.Skip
        return sum(model.line_added_capacity[line, y] for y in model.years) <= limit

    model.line_total_build_limit_constraint = pyo.Constraint(
        model.new_lines,
        rule=line_total_build_limit_rule
    )

    return model
