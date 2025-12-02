import pandas as pd
import numpy as np

import pypsa
from pypsa.common import annuity

## Functions to get realistic toy data

# Load time series at resolution freq based on PyPSA open data
def synth_load_ts(start="2025-01-01 00:00:00", end="2030-01-01 00:00:00", freq="h",
                     annual_peak_mw=30):
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
    
    # Now, rescale load to specified value
    if type(annual_peak_mw) in [int, float]:
        df["load_mw"] = annual_peak_mw * df["load_pu"]
    elif pd.api.types.is_list_like(annual_peak_mw):
        # assume annual_peak_mw has time indices and "peak_mw" column
        annual_peak_mw.columns = ["peak_mw"]
        annual_peak_mw["year"] = annual_peak_mw.index.year
        df = pd.merge(df, annual_peak_mw, on="year", how="left")
        df["peak_mw"] = df["peak_mw"].ffill()
        df["load_mw"] = df["load_pu"] * df["peak_mw"]

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
    return df

# Get cost data from the pypsa data repository
def get_synth_costs(years):
    gen = ["CCGT", "onwind", "solar"]
    data_years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
    all_costs = {}
    for year in years:
        data_year = data_years[np.argmin(np.abs(data_years-year))]
        url = f"https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_{data_year}.csv"
        costs = pd.read_csv(url, index_col=[0, 1])
        costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
        costs = costs.value.unstack().fillna({"discount rate": 0.07, "lifetime": 20, "FOM": 0})
        costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]
        a = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
        costs["capital_cost"] = (a + costs["FOM"] / 100) * costs["investment"]
        costs = costs.loc[gen, ["marginal_cost", "capital_cost"]]
        all_costs[year] = costs
    return all_costs

def synth_cost_dfs(years):
    """
    Generate realistic capital cost ($/MW-year) and marginal cost ($/MWh)
    for Gas, Solar, and Onshore Wind using simple trend formulas.
    """

    years_arr = np.array(years)
    year0 = years_arr[0]

    # --- Marginal cost ($/MWh) ---
    params_mc = {
        "gas":   (50, 0.01),   # $50/MWh rising 1% per year
        "pv": (3,  0.00),   # stable low VOM
        "wind":  (6,  0.005),  # slight increase over time
    }

    # --- Capital cost ($/MW-year) ---
    params_capex = {
        "gas":   (200_000, 0.005),   # slow decline
        "pv": (120_000, 0.03),    # faster decline
        "wind":  (230_000, 0.015),   # moderate decline
    }

    # Trend formulas
    def decline(start, rate, years, year0):
        return start * (1 - rate) ** (years - year0)

    def growth(start, rate, years, year0):
        return start * (1 + rate) ** (years - year0)

    df_capex = pd.DataFrame({"Year": years})
    for col, (start, rate) in params_capex.items():
        df_capex[col] = decline(start, rate, years_arr, year0)

    df_mc = pd.DataFrame({"Year": years})
    for col, (start, rate) in params_mc.items():
        df_mc[col] = growth(start, rate, years_arr, year0)

    df_capex = df_capex.set_index("Year")
    df_mc = df_mc.set_index("Year")
    return df_capex, df_mc
    

# Set snapshots for the model given years and resolution within each year
def set_snapshots(model, years, resolution):
    snapshots = pd.DatetimeIndex([])
    for year in years:
        period = pd.date_range(
            start=f"{year}-01-01 00:00",
            end=f"{year}-12-31 23:00",
            freq=f"{resolution}h",
        )
        snapshots = snapshots.append(period)
    
    snapshots =  pd.MultiIndex.from_arrays([snapshots.year, snapshots])
    model.snapshots = snapshots
    model.years = years

# copied from pypsa docs
def set_obj_weights(model, years, year_step=1):
    model.investment_period_weightings["years"] = year_step

    r = 0.01
    T = 0
    for period, nyears in model.investment_period_weightings.years.items():
        discounts = [(1 / (1 + r) ** t) for t in range(T, T + nyears)]
        model.investment_period_weightings.at[period, "objective"] = sum(discounts)
        T += nyears

def add_buses(model, nzones):
    for i in range(nzones):
        model.add("Bus", str(i))
        
def add_build(model, year, p_max_pu,
                 cap_cost, marg_cost,
                 name=""):
    for bus in model.buses.index:
        model.add("Generator", str(bus) + "-" + name + str(year),
                  bus=bus,
                  build_year=year,
                  marginal_cost=marg_cost,
                  capital_cost=cap_cost,
                  p_max_pu=p_max_pu,
                  p_nom_extendable=True)
                  
