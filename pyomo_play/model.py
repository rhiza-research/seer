import pyomo.environ as pyo

def seer_base_model(discrete_additions=False):
    # define a capacity planning model
    model = pyo.AbstractModel()

    """Indices"""
    # spatial
    model.zones = pyo.Set()
    # temporal
    model.years = pyo.Set(ordered=True)
    model.hours = pyo.Set(ordered=True)
    model.times = model.years * model.hours
    # generators - existing and new build generators together comprise all generators
    model.existing_generators = pyo.Set()
    model.new_generators = pyo.Set()
    model.generators = model.existing_generators | model.new_generators

    """Parameters"""
    # generator attributes
    model.generator_zone = pyo.Param(model.generators, within=pyo.Any)
    model.generator_tech = pyo.Param(model.generators, within=pyo.Any)

    # existing generators have a fixed capacity, between start and retirement year
    model.generator_capacity = pyo.Param(model.existing_generators, within=pyo.NonNegativeReals)
    model.generator_start_year = pyo.Param(model.existing_generators, within=pyo.Reals)
    model.generator_retirement_year = pyo.Param(model.existing_generators, within=pyo.Reals)
    # new build generators have an annual build limit and a discrete capacity increment
    model.generator_annual_build_limit = pyo.Param(model.new_generators, within=pyo.NonNegativeReals)
    model.generator_unit_size = pyo.Param(model.new_generators, within=pyo.PositiveReals)

    # costs
    model.generator_marginal_cost = pyo.Param(model.generators, model.years, within=pyo.NonNegativeReals)
    model.generator_capex_cost = pyo.Param(model.new_generators, model.years, within=pyo.NonNegativeReals)

    # capacity factor: 1.0 for dispatchable generators, hourly factor for renewables
    model.generator_capacity_factor = pyo.Param(model.generators, model.hours, within=pyo.NonNegativeReals)

    # demand - defined at each time in each zone
    model.demand = pyo.Param(model.zones, model.times, within=pyo.NonNegativeReals)
    # value of lost load ($/MWh); kept high so lost load is only used when truly necessary
    # TODO: why is this intialized
    model.voll = pyo.Param(within=pyo.NonNegativeReals, initialize=1_000_000.0, mutable=True)

    """Variables"""
    model.generation_output = pyo.Var(model.generators, model.times, within=pyo.NonNegativeReals)
    # capacity added in a given year (persists into all subsequent years); only new-build generators can add capacity
    if discrete_additions:
        # additions must come in whole unit_size increments: units is the decision variable,
        # generation_added_capacity is just units * unit_size, used everywhere else unchanged
        model.generation_added_units = pyo.Var(model.new_generators, model.years, within=pyo.NonNegativeIntegers)

        def added_capacity_rule(model, gen, year):
            return model.generation_added_units[gen, year] * model.generator_unit_size[gen]

        model.generation_added_capacity = pyo.Expression(model.new_generators, model.years, rule=added_capacity_rule)
    else:
        model.generation_added_capacity = pyo.Var(model.new_generators, model.years, within=pyo.NonNegativeReals)
    model.total_capacity = pyo.Var(model.generators, model.years, within=pyo.NonNegativeReals)
    # unserved demand; keeps the model feasible even when capacity/build limits can't otherwise meet demand
    model.lost_load = pyo.Var(model.zones, model.times, within=pyo.NonNegativeReals)

    """Objective"""
    def objective_rule(model):
        # total cost objective
        opex = sum(
            model.generator_marginal_cost[gen, year] * model.generation_output[gen, year, hour]
            for gen in model.generators
            for year, hour in model.times
        )
        capex = sum(
            model.generator_capex_cost[gen, year] * model.generation_added_capacity[gen, year]
            for gen in model.new_generators
            for year in model.years
        )
        lost_load_cost = sum(
            model.voll * model.lost_load[zone, year, hour]
            for zone in model.zones
            for year, hour in model.times
        )
        return opex + capex + lost_load_cost

    # least cost objective
    model.objective = pyo.Objective(
        rule=objective_rule,
        sense=pyo.minimize
    )


    """Constraints"""

    # 1. supply = demand + lost load in each zone
    def supply_equals_demand_rule(model, zone, year, hour):
        return sum(
            model.generation_output[gen, year, hour]
            for gen in model.generators
            if model.generator_zone[gen] == zone
        ) + model.lost_load[zone, year, hour] == model.demand[zone, year, hour]

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
            model.total_capacity[gen, year] * model.generator_capacity_factor[gen, hour]
        )

    model.generator_output_limit = pyo.Constraint(
        model.generators,
        model.times,
        rule=generator_output_limit_rule
    )

    return model
