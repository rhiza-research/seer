import pandas as pd
import pypsa

def build_existing_storage(model, existing_storage_df):
    nstorage = len(existing_storage_df)
    if nstorage == 0:
        return
    
    for i in range(nstorage):
        name = existing_storage_df.loc[i, "name"]
        zone = existing_storage_df.loc[i, "zone"]
        carrier = existing_storage_df.loc[i, "carrier"]
        p_nom = existing_storage_df.loc[i, "capacity"]
        max_hours = existing_storage_df.loc[i, "max_hours"]
        marginal_cost = existing_storage_df.loc[i, "marginal_cost"]
        capital_cost = existing_storage_df.loc[i, "capital_cost"]
        efficiency_store = existing_storage_df.loc[i, "efficiency_store"]
        efficiency_dispatch = existing_storage_df.loc[i, "efficiency_dispatch"]

        model.add("StorageUnit", name, bus = zone,
            carrier = carrier,
            max_hours = max_hours,
            marginal_cost = marginal_cost,
            capital_cost = capital_cost,
            efficiency_store = efficiency_store,
            efficiency_dispatch = efficiency_dispatch,
            cyclic_state_of_charge = True,
            p_nom = p_nom,
            p_nom_extendable = False)

def create_storage_build_options(model, storage_df):
    nstorage = len(storage_df)
    if nstorage == 0:
        return
    
    for i in range(nstorage):
        name = storage_df.loc[i, "name"]
        zone = storage_df.loc[i, "zone"]
        carrier = storage_df.loc[i, "carrier"]
        p_nom_mod = storage_df.loc[i, "unit_capacity"]
        p_nom_max = storage_df.loc[i, "capacity"]
        marginal_cost = storage_df.loc[i, "marginal_cost"]
        capital_cost = storage_df.loc[i, "capital_cost"]
        build_year = storage_df.loc[i, "build_year"]
        lifetime = storage_df.loc[i, "lifetime"]

        model.add("StorageUnit", name, bus = zone,
            carrier = carrier,
            max_hours = max_hours,
            marginal_cost = marginal_cost,
            capital_cost = capital_cost,