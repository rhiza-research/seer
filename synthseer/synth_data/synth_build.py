import pandas as pd
import numpy as np
from data import sources
from data import synth_re_ts

def generate_synthetic_gen_options(zone, start_year=2025, end_year=2051):
    years = range(start_year, end_year)
    data = []

    for y in years:
        t = (y - start_year) / (end_year - start_year)

        wind_capex  = 1400 - t * (1400 - 900)
        solar_capex = 900  - t * (900 - 500)
        ccgt_capex  = 1000 - t * (1000 - 900)
        coal_capex  = 2500 + t * (3500 - 2500)

        ccgt_mc  = 55 + t * (65 - 55)
        coal_mc  = 38 + t * (50 - 38)

        data += [
            [zone, "wind", 200, 0, round(wind_capex,1), y],
            [zone, "solar", 200, 0, round(solar_capex,1), y],
            [zone, "ccgt", 300, round(ccgt_mc,1), round(ccgt_capex,1), y],
            [zone, "coal", 400, round(coal_mc,1), round(coal_capex,1), y],
        ]

    df = pd.DataFrame(data, columns=[
        "zone","carrier","capacity","marginal_cost","capital_cost","build_year"
    ])

    df.to_csv("build_generators.cvs", index=False)

def generate_synthetic_storage_options(zone, start_year=2025, end_year=2051):
    years = range(start_year, end_year)

    data = []

    for y in years:
        t = (y - start_year) / (end_year - start_year)

        capex = 800 - t * (800 - 300)
        eff   = 0.95 + t * (0.97 - 0.95)

        data.append([
            f"North_Battery_{y}",
            zone,
            4,
            round(eff,3),
            round(eff,3),
            100,
            2,
            round(capex,1),
            y
        ])

    df = pd.DataFrame(data, columns=[
        "name","zone","max_hours","efficiency_store","efficiency_dispatch",
        "capacity","marginal_cost","capital_cost","build_year"
    ])

    df.to_csv("build_storage.csv", index=False)

def generate_synthetic_pnom_pu(zone, start_year=2025, end_year=2051, rez="1h"):
    re_ts = synth_re_ts(str(start_year), str(end_year), freq=rez, use_local=True)

if __name__ == "__main__":
    generate_synthetic_gen_options("North")
    generate_synthetic_storage_options("North")