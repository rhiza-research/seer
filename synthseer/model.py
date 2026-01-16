import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pypsa
from pypsa.common import annuity

# copied from pypsa docs
def set_obj_weights(model, years, year_step=1):
    model.investment_period_weightings["years"] = year_step

    r = 0.08 # discount rate - should change by country, but this is reasonable for LMICs
    start_at = 0 # if 0, discount factor is 1 for the first year
    for period, nyears in model.investment_period_weightings.years.items():
        discounts = [(1 / (1 + r) ** t) for t in range(start_at, start_at + nyears)]
        model.investment_period_weightings.at[period, "objective"] = sum(discounts)
        start_at += nyears

def is_re(source):
    return source in ["solar", "wind", "hydro"]

def create_model_from_data(input_data, voll=2000, modular_expansion=False):
    model = pypsa.Network()
    # add the carriers to the model
    for source in input_data["sources"]:
        model.add("Carrier", source)
    model.add("Carrier", "AC")
    model.add("Carrier", "battery")

    model.snapshots = input_data["snapshots"]

    years = input_data["years"]
    model.investment_periods = years

    year_step = 1
    if len(years) > 1:
        year_step = np.diff(np.array(years))[0]
    set_obj_weights(model, years, year_step=year_step)

    # Add buses
    for z in input_data["zones"]:
        model.add("Bus", z, carrier="AC")

    # Add current generation & build options
    for z in model.buses.index:
        for source in input_data["sources"]:
            # current generation
            name_exist = source + " - " + str(z) + "_exist"
            p_max_pu = 1
            if is_re(source):
                p_max_pu = pd.Series(input_data["re_ts"][source].values, index=model.snapshots,
                                     name = name_exist)
            model.add("Generator", name_exist, bus = z, 
                      carrier = source,
                      marginal_cost = input_data["generation costs"].loc[(source, "marginal"), years[0]],
                      p_max_pu = p_max_pu,
                      p_nom_extendable = False,
                      p_nom = input_data["source capacities"].loc[(z, source), "capacity"])

            # build options for every year
            for year in years:
                # new generation
                name_new = source + " - " + str(z) + "_" + str(year)
                if is_re(source):
                    p_max_pu.name = name_new
                if modular_expansion:
                    p_nom_mod = input_data["typical_capacity"][source]
                else:
                    p_nom_mod = 0
                model.add("Generator", name_new, bus = z,
                          carrier = source,
                          build_year = year,
                          marginal_cost = input_data["generation costs"].loc[(source, "marginal"), year],
                          capital_cost = input_data["generation costs"].loc[(source, "capital"), year],
                          p_max_pu = p_max_pu,
                          p_nom_mod = p_nom_mod,
                          p_nom_extendable = True)

    # Add storage
    for z in model.buses.index:
        name_exist = "battery - " + z + "_exist"
        # existing storage
        model.add("StorageUnit", name_exist, bus = z,
                  carrier = "battery",
                  max_hours = 6,
                  marginal_cost = 0,
                  capital_cost = 0,
                  efficiency_store = 1, efficiency_dispatch = 1,
                  cyclic_state_of_charge = True,
                  p_nom = 200,
                  p_nom_extendable = False)

        for year in years:
            name_new = "battery - " + z + "_" + str(year)
            # new storage
            model.add("StorageUnit", name_new, bus = z,
                    carrier = "battery",
                    max_hours = 6,
                    marginal_cost = 0,
                    capital_cost = 0,
                    efficiency_store = 1, efficiency_dispatch = 1,
                    cyclic_state_of_charge = True,
                    build_year = year,
                    p_nom_extendable = True)
    # Limit battery deployment
    
    model.
    # Add load shedding generators
    for z in model.buses.index:
        model.add("Generator", "load shedding - " + z, bus = z,
                  marginal_cost = voll,
                  p_nom = input_data["zonal load"].loc[z, "peak power"],
                  p_nom_extendable = False)

    # Add load
    for z in model.buses.index:
        model.add("Load","load - " + z, bus = z,
                  p_set = input_data["load_ts"]["load_pu"] * input_data["zonal load"].loc[z, "peak power"])

    return model



