import pypsa
import pandas

from synthseer.build_model.time_utils import get_valid_years, annual_to_snapshots

def build_generators(model, gen_constants, gen_costs, modular_expansion=False):
    ngen = len(gen_constants)
    if ngen == 0:
        return
    
    for i in range(ngen):
        gen = gen_constants.loc[i]
        name, zone, carrier, expandable = gen.generator, gen.zone, gen.fuel, gen.expandable
        if expandable:
            """Build option.
            In the PyPSA framework, we need to create a build option for each year.
            """ 
            y0, y1 = gen.start_year, gen.retirement_year
            capex = gen_costs.loc[(name, "capex"), :]
            mc = gen_costs.loc[(name, "marginal_cost"), :]
            unit_capacity = gen.unit_size if modular_expansion else None
            p_max_pu = get_renewable_profile(model, carrier) if is_renewable(carrier) else 1
            for year in get_valid_years(model, y0, y1):
                model.add("Generator", name + "_" + str(year), bus = zone,
                          carrier = carrier,
                          build_year = year,
                          lifetime = gen.life,
                          # power limits
                          p_max_pu = p_max_pu,
                          p_nom_extendable = True,
                          p_nom_mod = unit_capacity,
                          p_nom_max = gen.annual_build_limit,
                          # costs
                          marginal_cost = mc.loc[year], capital_cost = capex.loc[year],
                          # ramping limits
                          ramp_up_rate = gen.ramp_up_rate, ramp_down_rate = gen.ramp_down_rate)

        else:  
            """Existing generator.
            In the PyPSA framework, we need to create an existing generator for each year.
            """
            y0, y1 = gen.start_year, gen.retirement_year
            years = get_valid_years(model, y0, y1)
            # TODO: do we need to add capex for existing generators?
            # capex = gen_costs.loc[(name, "capex"), :]
            mc = annual_to_snapshots(model, gen_costs.loc[(name, "marginal_cost"), :])
            p_max_pu = get_renewable_profile(model, carrier) if is_renewable(carrier) else 1
            
            model.add("Generator", name, bus = zone,
                        carrier = carrier,
                        build_year = years[0],
                        lifetime = gen.life,
                        # power limits
                        p_max_pu = p_max_pu,
                        p_nom = gen.capacity,
                        p_nom_extendable = False,
                        # costs
                        marginal_cost = mc,
                        # ramping limits
                        ramp_up_rate = gen.ramp_up_rate, ramp_down_rate = gen.ramp_down_rate)

def is_renewable(carrier):
    return carrier in ["solar", "wind", "PV"]

def build_existing_generators(model, existing_generator_df, pnom_pu_df):
    ngen = len(existing_generator_df)
    
    if ngen == 0:
        return
    
    for i in range(ngen):
        name = existing_generator_df.loc[i, "name"]
        zone = existing_generator_df.loc[i, "zone"]
        carrier = existing_generator_df.loc[i, "carrier"]
        p_nom = existing_generator_df.loc[i, "capacity"]
        marginal_cost = existing_generator_df.loc[i, "marginal_cost"]

        if is_renewable(carrier):
            p_max_pu = pnom_pu_df.loc[(carrier, zone), :]
        else:
            p_max_pu = 1

        model.add("Generator", name, bus = zone,
                  carrier = carrier,
                  p_nom = p_nom,
                  p_max_pu = p_max_pu,
                  p_nom_extendable = False, # an existing generator is not extendable
                  marginal_cost = marginal_cost)

def create_generator_build_options(model, generator_df, pnom_pu_df):
    ngen = len(generator_df)
    if ngen == 0:
        return
    
    for i in range(ngen):
        name = generator_df.loc[i, "name"]
        zone = generator_df.loc[i, "zone"]
        carrier = generator_df.loc[i, "carrier"]
        p_nom_mod = generator_df.loc[i, "unit_capacity"]
        p_nom_max = generator_df.loc[i, "capacity"]
        marginal_cost = generator_df.loc[i, "marginal_cost"]
        capital_cost = generator_df.loc[i, "capital_cost"]
        build_year = generator_df.loc[i, "build_year"]
        lifetime = generator_df.loc[i, "lifetime"]

        if is_renewable(carrier):
            p_max_pu = pnom_pu_df.loc[(carrier, zone), :]
        else:
            p_max_pu = 1

        model.add("Generator", name, bus = zone,
                  carrier = carrier,
                  p_max_pu = p_max_pu,
                  p_nom_extendable = True,
                  p_nom_mod = p_nom_mod,
                  p_nom_max = p_nom_max,
                  marginal_cost = marginal_cost,
                  capital_cost = capital_cost,
                  build_year = build_year,
                  lifetime = lifetime)
