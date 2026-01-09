import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def viz_model(model):
    new_gens = model.generators[model.generators.build_year > 0]
    new_gens["source"] = new_gens.index.str.split("-").str[0].str.strip()
    new_gens = new_gens[["source", "build_year", "p_nom_opt"]].groupby(["source", "build_year"]).sum()
    df_unstacked = new_gens["p_nom_opt"].unstack("build_year")
    df_unstacked.plot.bar(figsize=(10,5))
    plt.show()