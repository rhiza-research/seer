import pypsa
import pandas

def is_renewable(carrier):
    return carrier in ["solar", "wind"]

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
