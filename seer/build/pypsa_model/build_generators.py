import pypsa
import pandas
import warnings

from .time_utils import get_valid_years, annual_to_snapshots

def build_generators(model, gen_constants, gen_costs, re_profiles, modular_expansion=False):
    # carrier set to be returned after processing
    carriers = set()
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
        return carriers
    
    for i in range(ngen):
        gen = gen_constants.iloc[i]
        name, zone, carrier, expandable, gen_type = gen.name, gen.zone, gen.fuel, gen.expandable, gen.tech
        carriers.add(carrier)

        if expandable:
            """Build option.
            In the PyPSA framework, we need to create a build option for each year.
            """ 
            y0, y1 = gen.start_year, gen.retirement_year
            valid_years = get_valid_years(model, y0, y1)
            capex = gen_costs.loc[(name, "capex"), :]
            # fill missing marginal costs with 0
            mc = gen_costs.loc[(name, "marginal_cost"), :].fillna(0)
            unit_capacity = gen.unit_size if modular_expansion else None

            for year in valid_years:
                gen_year_name = name + "_" + str(year)
                # get the renewable profile for the generator type if required.
                p_max_pu = mask_renewable_profile(
                    re_profiles[(zone, gen_type)],
                    valid_years,
                    year,
                    gen.life
                ) if is_renewable(gen_type) else 1
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
            mc = annual_to_snapshots(model, gen_costs.loc[(name, "marginal_cost"), :]).fillna(0)

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
    return carriers

def is_renewable(carrier):
    return carrier in ["PV", "wind"]

# TODO: check that this makes sense
def mask_renewable_profile(profile, valid_years, build_year, lifetime, default=0.0):
    active_years = set(range(int(build_year), int(build_year) + int(lifetime)))
    valid_years = set(valid_years)

    years = profile.index.get_level_values(0)
    allowed_years = active_years & valid_years
    return profile.where(years.isin(allowed_years), default)

