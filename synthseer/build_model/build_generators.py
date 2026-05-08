import pypsa
import pandas
import warnings

from synthseer.build_model.time_utils import get_valid_years, annual_to_snapshots

def build_generators(model, gen_constants, gen_costs, re_profiles, modular_expansion=False):
    # convert simple csv inputs into multi-index dataframes to ease processing
    # index gen_constants by generator
    gen_constants = gen_constants.set_index("generator")
    # index gen_costs by generator and cost_type
    gen_costs = gen_costs.set_index(["generator", "cost_type"])
    # make sure columns for years are numeric
    gen_costs.columns = gen_costs.columns.astype(int)

    # no generators
    ngen = len(gen_constants)
    if ngen == 0:
        warnings.warn("No generators found in the model.")
        return
    
    for i in range(ngen):
        gen = gen_constants.iloc[i]
        name, zone, carrier, expandable, gen_type = gen.name, gen.zone, gen.fuel, gen.expandable, gen.tech

        if expandable:
            """Build option.
            In the PyPSA framework, we need to create a build option for each year.
            """ 
            y0, y1 = gen.start_year, gen.retirement_year
            capex = gen_costs.loc[(name, "capex"), :]
            mc = gen_costs.loc[(name, "marginal_cost"), :]
            unit_capacity = gen.unit_size if modular_expansion else None
            # get the renewable profile for the generator type if required.
            p_max_pu = re_profiles.loc[(year, slice(None)), (zone, gen_type)] if is_renewable(gen_type) else 1
            for year in get_valid_years(model, y0, y1):
                gen_year_name = name + "_" + str(year)
                model.add("Generator", gen_year_name, bus = zone,
                          carrier = carrier,
                          build_year = year,
                          lifetime = gen.life,
                          # power limits
                          p_nom_extendable = True,
                          p_nom_mod = unit_capacity,
                          p_nom_max = gen.annual_build_limit,
                          # costs
                          marginal_cost = mc[year], capital_cost = capex[year],
                          # ramping limits
                          ramp_up_rate = gen.ramp_up_rate, ramp_down_rate = gen.ramp_down_rate)
                # time series parameters must be added through the generators_t dataframe
                model.generators_t.p_max_pu[gen_year_name] = p_max_pu

        else:  
            """Existing generator. Create a single generator for the entire lifetime.
            """
            y0, y1 = gen.start_year, gen.retirement_year
            years = get_valid_years(model, y0, y1)
            if len(years) == 0:
                warnings.warn(f"No valid years found for generator {name}.")
                continue
            
            # TODO: do we need to add capex for existing generators?
            # capex = gen_costs.loc[(name, "capex"), :]
            mc = annual_to_snapshots(model, gen_costs.loc[(name, "marginal_cost"), :])

            if is_renewable(gen_type):
                p_max_pu = re_profiles.loc[slice(None), (zone, gen_type)]
            else:
                p_max_pu = 1
            
            model.add("Generator", name, bus = zone,
                        carrier = carrier,
                        build_year = years[0],
                        lifetime = gen.life,
                        # power limits
                        p_nom = gen.capacity,
                        p_nom_extendable = False,
                        # ramping limits
                        ramp_up_rate = gen.ramp_up_rate, ramp_down_rate = gen.ramp_down_rate)
            # time series parameters must be added through the generators_t dataframe
            model.generators_t.p_max_pu[name] = p_max_pu
            model.generators_t.marginal_cost[name] = mc

def is_renewable(carrier):
    return carrier in ["PV", "wind"]

