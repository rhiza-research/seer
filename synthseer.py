import pandas as pd
import numpy as np

import pypsa
from pypsa.common import annuity

"""
Some global constants that we use to generate simple but somewhat
realistic data
"""

# List of generation types considered (these names should be used consistently)
sources = ["gas", "wind", "solar", "coal", "nuclear"]
# Typical scale of single generator (in MW)
typical_capacity = {
    "gas" : 600, # single CCGT unit
    "solar" : 100, # standard utility scale
    "wind" : 100, # standard utility scale
    "coal" : 600, # single unit
    "nuclear" : 1000,
    "hydro" : 500
}

# capital and marginal costs
# capital - $/MWh, marginal - $/MWh-year
init_costs = pd.DataFrame(index=sources, columns=["capital", "marginal"],
                             data = np.array([[81_000, 70],
                                              [102_000, 5],
                                              [136_000, 5],
                                              [280_000, 110],
                                              [525_000, 15]]))

cost_changes = pd.DataFrame(index=sources, columns=["capital", "marginal"],
                             data = np.array([[0.01, 0.02],
                                              [-0.05, -0.03],
                                              [-0.01, -0.01],
                                              [0, 0],
                                              [0, 0]]))

mixes = pd.DataFrame(index=sources, columns=["india", "us", "france"])
mixes.loc[:, "india"] = [0.04, 0.09, 0.06, 0.78, 0.03]
mixes.loc[:, "us"] = [0.46, 0.08, 0.11, 0.16, 0.19]
mixes.loc[:, "france"] = [0.04, 0.06, 0.1, 0, 0.8]

# per capita energy consumption (kWh) by country income group
pcc = {
    "lic" : 300,
    "mic" : 2500,
    "hic" : 8000,
}

# typical load factor
lf = 0.5

## Functions to get realistic toy data

def synth_data(start_year, nzones=1, nyears=1, year_step=1, rez="1h",
               population=3e6, income_class="mic", mix="india", excess_cap=1.5):
    """
    Inputs
    ------
    start_year [int] - Starting year, eg 2025
    nyears [int] - Number of years
    year_step [int] - Delta between modelled years.
    """
    input_data = {}
    
    end_year = start_year + nyears*year_step
    years = [y for y in range(start_year, end_year, year_step)]
    input_data["years"] = years

    zones = [str(z) for z in range(nzones)]
    input_data["zones"] = zones

    # Generate a random fraction of population in each zone
    prop = np.random.rand(nzones)
    prop = prop / np.sum(prop)

    # Get peak load per zone
    energy = population * pcc[income_class] # annual consumption in kWh
    power_avg = energy / 8760
    peak = power_avg / lf / 1e3 # peak power in MW
    peak_zonal = pd.DataFrame(index=zones, columns=["peak power"], data=peak * prop)
    input_data["zonal load"] = peak_zonal

    # Get starting capacity (MW) of each generator type in each zone
    source_capacities = pd.DataFrame(index=pd.MultiIndex.from_product([zones, sources], names=["zone", "source"]),
                                     columns=["capacity", "counts"])
    for source in sources:
        for zone in zones:
            source_capacities.loc[(zone, source), "capacity"] = peak_zonal.loc[zone, "peak power"] * excess_cap * mixes.loc[source, mix]
            # For interest (currently this is not used), get typical generator counts
            source_capacities.loc[(zone, source), "counts"] = source_capacities.loc[(zone, source), "capacity"] / typical_capacity[source]
    input_data["source capacities"] = source_capacities

    # Generate the temporal snapshots that define the modeling periods
    snapshots = pd.DatetimeIndex([])
    for year in years:
        period = pd.date_range(
            start=f"{year}-01-01 00:00",
            end=f"{year}-12-31 23:00",
            freq=rez,
        )
        snapshots = snapshots.append(period)
    snapshots =  pd.MultiIndex.from_arrays([snapshots.year, snapshots])
    input_data["snapshots"] = snapshots

    # Generate synthetic load & re supply profiles - use pypsa data.
    load_ts = synth_load_ts(str(start_year), str(end_year), freq=rez)
    re_ts = synth_re_ts(str(start_year), str(end_year), freq=rez)
    # the ts indices must match snapshots
    load_ts.index = snapshots
    re_ts.index = snapshots

    input_data["load_ts"] = load_ts
    input_data["re_ts"] = re_ts

    # Get capital and marginal costs by year
    dy = np.array(years) - years[0]
    cost_type = ["capital", "marginal"]
    costs = pd.DataFrame(index = pd.MultiIndex.from_product([sources, cost_type], names=["source", "cost type"]),
                         columns=years)
    for s in sources:
        for c in cost_type:
            c0 = init_costs.loc[s, c]
            p = cost_changes.loc[s, c]
            costs.loc[(s, c)] = c0*((1+p)**dy)
    input_data["generation costs"] = costs

    return input_data


# Load time series at resolution freq based on PyPSA open data
def synth_load_ts(start="2025-01-01 00:00:00", end="2030-01-01 00:00:00", freq="h"):
    """
    Load the load time series provided by pypsa, and extend 
    over an entire modeling period of interest at given frequency.
    """
    # Set up the empty dataframe with all the desired time points
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    delta = pd.to_timedelta(freq)
    times = pd.date_range(start=start, end=end-delta, freq=freq)
    
    df = pd.DataFrame(index=times)
    # Add day of year and hour columns
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour
    df["year"] = df.index.year

    # Get the pypsa data
    url = "https://tubcloud.tu-berlin.de/s/9toBssWEdaLgHzq/download/time-series.csv"
    pypsa_data = pd.read_csv(url, index_col=0, parse_dates=True)[["load_mw"]]
    pypsa_data["doy"] = pypsa_data.index.dayofyear
    pypsa_data["hour"] = pypsa_data.index.hour
    # we need to make the pypsa load p.u. rather than mw
    pypsa_data["load_pu"] = pypsa_data["load_mw"] / pypsa_data["load_mw"].max()
    pypsa_data = pypsa_data.drop("load_mw", axis=1)

    # Join pypsa to empty dataframe on day of year and hour
    df = pd.merge(df, pypsa_data, on=["doy", "hour"], how="left")
    # We may need to fill some nan values (eg leap year)
    df = df.apply(lambda col: col.fillna(col.groupby(df["hour"]).transform('mean')))

    # drop working columns
    df= df.drop(["doy", "hour", "year"], axis=1)
    df.index = times
    return df

# Renewable output time series at resolution freq based on PyPSA open data
def synth_re_ts(start="2025-01-01 00:00:00", end="2030-01-01 00:00:00", freq="h"):
    """
    Load the RE time series provided by pypsa, and extend 
    over an entire modeling period of interest. 
    """
    # Set up the empty dataframe with all the desired time points
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    delta = pd.to_timedelta(freq)
    times = pd.date_range(start=start, end=end-delta, freq=freq)
    
    df = pd.DataFrame(index=times)
    # Add day of year and hour columns
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour

    # Get the pypsa data
    url = "https://tubcloud.tu-berlin.de/s/9toBssWEdaLgHzq/download/time-series.csv"
    pypsa_data = pd.read_csv(url, index_col=0, parse_dates=True)[["pv_pu", "wind_pu"]]
    pypsa_data["doy"] = pypsa_data.index.dayofyear
    pypsa_data["hour"] = pypsa_data.index.hour

    # Join pypsa to empty dataframe on day of year and hour
    df = pd.merge(df, pypsa_data, on=["doy", "hour"], how="left")
    # We may need to fill some nan values (eg leap year)
    df = df.apply(lambda col: col.fillna(col.groupby(df["hour"]).transform('mean')))

    # drop working columns
    df= df.drop(["doy", "hour"], axis=1)
    df.index = times
    df.columns = ["solar", "wind"]
    return df

# copied from pypsa docs
def set_obj_weights(model, years, year_step=1):
    model.investment_period_weightings["years"] = year_step

    r = 0.01
    T = 0
    for period, nyears in model.investment_period_weightings.years.items():
        discounts = [(1 / (1 + r) ** t) for t in range(T, T + nyears)]
        model.investment_period_weightings.at[period, "objective"] = sum(discounts)
        T += nyears

def is_re(source):
    return source in ["solar", "wind", "hydro"]

def create_model_from_data(input_data, voll=2000):
    model = pypsa.Network()
    model.add("Carrier", "AC")

    model.snapshots = input_data["snapshots"]

    years = input_data["years"]
    model.years = years

    year_step = 1
    if len(years) > 1:
        year_step = np.diff(np.array(years))[0]
    set_obj_weights(model, years, year_step=year_step)

    # Add buses
    for z in input_data["zones"]:
        model.add("Bus", z, carrier="AC")

    # Add current generation & build options
    for z in model.buses.index:
        for source in sources:
            p_max_pu = None
            if is_re(source):
                p_max_pu = input_data["re_ts"][source]
            # current generator
            model.add("Generator", str(z) + " - " + source + "_exist", bus = z, 
                      marginal_cost = input_data["generation costs"].loc[(source, "marginal"), years[0]],
                      p_max_pu = p_max_pu,
                      p_nom_extendable = False,
                      p_nom = input_data["source capacities"].loc[(z, source), "capacity"])
            # build options for every year
            for year in years:
                model.add("Generator", str(z) + " - " + source + "_" + str(year), bus = z,
                          marginal_cost = input_data["generation costs"].loc[(source, "marginal"), year],
                          capital_cost = input_data["generation costs"].loc[(source, "capital"), year],
                          p_max_pu = p_max_pu,
                          p_nom_extendable = True)

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

