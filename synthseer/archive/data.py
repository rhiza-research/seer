import pandas as pd
import numpy as np
import os
"""
Some global constants that we use to generate simple but somewhat
realistic data
"""

# List of generation types considered (these names should be used consistently)
sources = ["gas", "solar", "wind", "coal", "nuclear", "hydro"]
# Typical scale of single generator (in MW)
typical_capacity = {
    "gas" : 600, # single CCGT unit
    "solar" : 100, # standard utility scale
    "wind" : 100, # standard utility scale
    "coal" : 600, # single unit
    "nuclear" : 1000,
    "hydro" : 500,
    "battery" : 200 # reasonably large utility scale battery
}

# PyPSA units:
# capital  -> EUR/MW/year (annualized)
# marginal -> EUR/MWh

init_costs = pd.DataFrame(
    index=sources,
    columns=["capital", "marginal"],
    data=np.array([
        [85_000, 70],    # gas (CCGT + fuel + CO2)
        [4_000, 2],     # solar PV
        [65_000, 3],     # onshore wind
        [120_000, 100],  # coal (penalized by CO2)
        [180_000, 10],   # nuclear
        [90_000, 5],     # hydro
    ])
)

cost_changes = pd.DataFrame(
    index=sources,
    columns=["capital", "marginal"],
    data=np.array([
        [0.01, 0.02],    # gas
        [-0.06, -0.01],  # solar learning
        [-0.04, -0.01],  # wind learning
        [0.01, 0.02],    # coal worsening
        [0.00, 0.00],    # nuclear flat
        [0.00, 0.00],    # hydro flat
    ])
)

storage_init_costs = pd.DataFrame(
    index=["battery"],
    columns=["capital", "marginal"],
    data=np.array([
        [35_000, 0.5],   # battery (power capacity)
    ])
)

storage_cost_changes = pd.DataFrame(
    index=["battery"],
    columns=["capital", "marginal"],
    data=np.array([
        [-0.08, -0.02],
    ])
)


mixes = pd.DataFrame(index=sources, columns=["india", "us", "france"])
mixes.loc[:, "india"] = [0.04, 0.09, 0.06, 0.78, 0.03, 0]
mixes.loc[:, "us"] = [0.46, 0.08, 0.11, 0.16, 0.19, 0]
mixes.loc[:, "france"] = [0.04, 0.06, 0.1, 0, 0.8, 0]

# per capita energy consumption (kWh) by country income group
pcc = {
    "lic" : 300,
    "mic" : 2500,
    "hic" : 8000,
}

# typical load factor
lf = 0.5

def synth_data(start_year, nzones=1, nyears=1, year_step=1, rez="1h",
               population=3e6, income_class="mic", mix="india", excess_cap=1.5, load_growth=0.1):
    """
    Inputs
    ------
    start_year [int] - Starting year, eg 2025
    nyears [int] - Number of years
    year_step [int] - Delta between modelled years.
    """
    input_data = {}

    # Add sources and typical capacities to input data
    input_data["sources"] = sources
    input_data["typical_capacity"] = typical_capacity
    
    end_year = start_year + nyears*year_step
    years = [y for y in range(start_year, end_year, year_step)]
    input_data["years"] = years

    zones = [str(z) for z in range(nzones)]
    input_data["zones"] = zones

    # Generate a random fraction of population in each zone
    prop = np.random.rand(nzones)
    prop = prop / np.sum(prop)

    # Get peak load per zone in starting year
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
    re_ts = synth_re_ts(str(start_year), str(end_year), freq=rez, use_local=True)
    # the ts indices must match snapshots
    load_ts.index = snapshots
    re_ts.index = snapshots

    # Add load growth to the load time series
    y = np.array(years)
    growth = pd.DataFrame(index=years, columns=["growth factor"], data=(1+load_growth)**(y-y[0]))
    load_ts = load_ts.join(growth, on=load_ts.index.get_level_values(0))
    load_ts["load_pu"] = load_ts["load_pu"] * load_ts["growth factor"]
    load_ts = load_ts.drop(["key_0", "growth factor"], axis=1)

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

    # Get storage capital and marginal costs by year
    storage_costs = pd.DataFrame(index = pd.MultiIndex.from_product([["battery"], cost_type], names=["source", "cost type"]),
                                 columns=years)
    for c in cost_type:
        c0 = storage_init_costs.loc["battery", c]
        p = storage_cost_changes.loc["battery", c]
        storage_costs.loc[("battery", c)] = c0*((1+p)**dy)
    input_data["storage costs"] = storage_costs

    return input_data

# Functions to generate synthetic data

# Load time series at resolution freq based on PyPSA open data
def synth_load_ts(start="2025-01-01 00:00:00", end="2030-01-01 00:00:00", freq="h", use_local=True):
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
    # Look for time-series.csv in data/ otherwise use url
    if use_local and os.path.exists("data/time-series.csv"):
        pypsa_data = pd.read_csv("data/time-series.csv", index_col=0, parse_dates=True)[["load_mw"]]
    else:
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
def synth_re_ts(start="2025-01-01 00:00:00", end="2030-01-01 00:00:00", freq="h", use_local=True):
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
    if use_local and os.path.exists("data/time-series.csv"):
        pypsa_data = pd.read_csv("data/time-series.csv", index_col=0, parse_dates=True)[["pv_pu", "wind_pu"]]
    else:
        url = "https://tubcloud.tu-berlin.de/s/9toBssWEdaLgHzq/download/time-series.csv"
        pypsa_data = pd.read_csv(url, index_col=0, parse_dates=True)[["pv_pu", "wind_pu"]]
    # average pypsa data to freq
    pypsa_data = pypsa_data.resample(freq).mean()

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