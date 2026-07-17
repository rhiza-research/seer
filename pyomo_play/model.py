import pyomo.environ as pyo

def seer_model():
    # define a capacity planning model
    model = pyo.AbstractModel()

    """Indices"""
    model.zones = pyo.Set()
    model.years = pyo.Set(ordered=True)
    model.hours = pyo.Set(ordered=True)
    # time points are years and hours
    model.times = model.years * model.hours
    # individual generators (each tied to one zone; existing units and new-build options are separate entries)
    model.generators = pyo.Set()

    """Parameters"""
    # GENERATOR ATTRIBUTES
    model.generator_zone = pyo.Param(model.generators, within=pyo.Any)
    model.generator_expandable = pyo.Param(model.generators, within=pyo.Boolean)
    # nominal capacity: fixed capacity for existing generators, reference unit size for new-build ones (unused directly)
    model.generator_capacity = pyo.Param(model.generators, within=pyo.NonNegativeReals)
    # only meaningful for non-expandable (existing) generators, gates when they're available
    model.generator_start_year = pyo.Param(model.generators, within=pyo.Reals)
    model.generator_retirement_year = pyo.Param(model.generators, within=pyo.Reals)
    # only meaningful for expandable (new-build) generators, caps capacity added in a single year
    model.generator_annual_build_limit = pyo.Param(model.generators, within=pyo.NonNegativeReals)

    # COSTS
    model.generator_capex_cost = pyo.Param(model.generators, model.years, within=pyo.NonNegativeReals)
    model.generator_marginal_cost = pyo.Param(model.generators, model.years, within=pyo.NonNegativeReals)

    # AVAILABILITY: 1.0 for dispatchable generators, an hourly capacity factor for renewables
    model.generator_availability_factor = pyo.Param(model.generators, model.hours, within=pyo.NonNegativeReals)

    # DEMAND
    model.demand = pyo.Param(model.zones, model.times, within=pyo.NonNegativeReals)
    # penalty cost ($/MWh) for unserved demand; kept high so lost load is only used when truly necessary
    model.lost_load_penalty = pyo.Param(within=pyo.NonNegativeReals, initialize=1_000_000.0, mutable=True)

    """Variables"""
    model.generation_output = pyo.Var(model.generators, model.times, within=pyo.NonNegativeReals)
    # capacity added in a given year (persists into all subsequent years); forced to 0 for non-expandable generators
    model.generation_added_capacity = pyo.Var(model.generators, model.years, within=pyo.NonNegativeReals)
    model.total_capacity = pyo.Var(model.generators, model.years, within=pyo.NonNegativeReals)
    # unserved demand; keeps the model feasible even when capacity/build limits can't otherwise meet demand
    model.lost_load = pyo.Var(model.zones, model.times, within=pyo.NonNegativeReals)

    """Objective"""
    def objective_rule(model):
        opex = sum(
            model.generator_marginal_cost[gen, year] * model.generation_output[gen, year, hour]
            for gen in model.generators
            for year, hour in model.times
        )
        capex = sum(
            model.generator_capex_cost[gen, year] * model.generation_added_capacity[gen, year]
            for gen in model.generators
            for year in model.years
        )
        return opex + capex

    model.objective = pyo.Objective(
        rule=objective_rule,
        sense=pyo.minimize
    )

    """Constraints"""

    # 1. supply equals demand, summed across all generators in a zone
    def supply_equals_demand_rule(model, zone, year, hour):
        return sum(
            model.generation_output[gen, year, hour]
            for gen in model.generators
            if model.generator_zone[gen] == zone
        ) == model.demand[zone, year, hour]

    model.supply_equals_demand = pyo.Constraint(
        model.zones,
        model.times,
        rule=supply_equals_demand_rule
    )

    # 2. capacity added in a year is bounded by the annual build limit; existing generators can't add capacity
    def build_limit_rule(model, gen, year):
        if pyo.value(model.generator_expandable[gen]):
            return model.generation_added_capacity[gen, year] <= model.generator_annual_build_limit[gen]
        return model.generation_added_capacity[gen, year] == 0

    model.build_limit = pyo.Constraint(
        model.generators,
        model.years,
        rule=build_limit_rule
    )

    # 3. total capacity: existing generators are simply present while active; expandable generators
    #    accumulate whatever capacity has been added up to and including this year
    def total_capacity_rule(model, gen, year):
        if pyo.value(model.generator_expandable[gen]):
            prior_and_current_years = [
                y for y in model.years if model.years.ord(y) <= model.years.ord(year)
            ]
            return model.total_capacity[gen, year] == sum(
                model.generation_added_capacity[gen, y] for y in prior_and_current_years
            )

        is_active = pyo.value(model.generator_start_year[gen]) <= year <= pyo.value(model.generator_retirement_year[gen])
        return model.total_capacity[gen, year] == (model.generator_capacity[gen] if is_active else 0)

    model.total_capacity_constraint = pyo.Constraint(
        model.generators,
        model.years,
        rule=total_capacity_rule
    )

    # 4. generator output is limited by total capacity, derated by hourly availability
    def generator_output_limit_rule(model, gen, year, hour):
        return model.generation_output[gen, year, hour] <= (
            model.total_capacity[gen, year] * model.generator_availability_factor[gen, hour]
        )

    model.generator_output_limit = pyo.Constraint(
        model.generators,
        model.times,
        rule=generator_output_limit_rule
    )

    return model
