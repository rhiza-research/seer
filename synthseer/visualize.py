import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def viz_model(model):
    fig, axs = plt.subplots(3, 1, figsize=(10,10))
    new_gens = model.generators[model.generators.build_year > 0]
    new_gens["source"] = new_gens.index.str.split("-").str[0].str.strip()
    new_gens = new_gens[["source", "build_year", "p_nom_opt"]].groupby(["source", "build_year"]).sum()
    df_unstacked = new_gens["p_nom_opt"].unstack("build_year")
    df_unstacked.plot.bar(ax=axs[0])

    
    new_storage = model.storage_units[model.storage_units.build_year > 0]
    new_storage["storage"] = new_storage.index.str.split("-").str[0].str.strip()
    new_storage = new_storage[["storage", "build_year", "p_nom_opt"]].groupby(["storage", "build_year"]).sum()
    df_unstacked = new_storage["p_nom_opt"].unstack("build_year")
    df_unstacked.plot.bar(ax=axs[1])
    

    # plot the dispatch by type
    # group columns by type where this is the column name before - 
    dispatch = model.generators_t.p.groupby(lambda x: x.split("-")[0], axis=1).sum()
    dispatch.plot(ax=axs[2])

    plt.show()