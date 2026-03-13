from signal import pthread_kill
import pypsa
import numpy as np
import pandas as pd
import calendar
import warnings
from synthseer.build_model.build_generators import build_generators

"""Global variables. 
TODO: can be moved to a config csv. 
"""
discount_rate = 0.08

def build_model(start_year=2025, end_year=2051, year_step=1, time_resolution="1h", data_path = "synthseer/data/Gambia/"):

    input_data = load_input_data(data_path)

    model = pypsa.Network()

    """TIME: Set up multi-period optimization. 
    Years are investment periods. Build decisions can be made in each year.
    Each year has some temporal resolution (snapshots) at which dispatch is modelled.
    """
    years = input_data["years"]["year"].tolist()
    model.investment_periods = years
    # If not modeling each year separately, one year should represent multiple years and be weighted accordingly.
    model.investment_period_weightings["years"] = get_year_steps(years)

    # Weight objective in each period by the discount factor.
    start_at = 0 # if 0, discount factor is 1 for the first year
    for period, nyears in model.investment_period_weightings.years.items():
        discounts = [(1 / (1 + discount_rate) ** t) for t in range(start_at, start_at + nyears)]
        model.investment_period_weightings.at[period, "objective"] = sum(discounts)
        start_at += nyears

    # Create snapshots for each year - this is the temporal resolution at which dispatch is modelled.
    # this must have a very specific multi-index format to work.
    snapshots = pd.DatetimeIndex(
        np.concatenate([
            pd.date_range(f"{y}-01-01 00:00", f"{y}-12-31 23:00", freq=time_resolution)
            for y in years
        ])
    )
    snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
    model.snapshots = snapshots

    """SPACE: Set up components of the model.
    Zones represent the spatial resolution of the model.
    """
    zones_df = input_data["zones"]
    for zone in zones_df["name"]:
        model.add("Bus", zone, carrier="AC")

    """DEMAND: Set up demand trajectory.
    """
    demandforecast = input_data["demandforecast"]
    demandprofile = input_data["demandprofile"]
    set_demand(model, demandforecast, demandprofile)

    """GENERATION: Set up existing generation and build options.
    """
    build_generators(model, input_data["generator_constants"], input_data["generator_costs"])

    """STORAGE: Set up existing storage and build options.
    """
    build_existing_storage(model, input_data["existing_storage_df"])
    create_storage_build_options(model, input_data["storage_df"], input_data["pnom_pu_df"])

    return model


def get_year_steps(years):
    years = np.array(years)
    steps = np.diff(years)
    # final time period is assumed to be 1 year
    steps = np.append(steps, [1])
    steps = pd.Series(steps, index=years)
    warnings.warn(f"Year step is {steps}")
    return steps


def set_demand(model, demandforecast, demandprofile):
    # for each zone in demand_forecast, get demand profile, multiply by demand forecast for each year, and set as load
    zones = demandforecast["zone"].unique()
    years = model.investment_periods
    for zone in zones:
        profile = demandprofile[demandprofile["zone"] == zone]
        forecast_peaks = demandforecast[demandforecast["zone"] == zone][demandforecast["type"] == "peak"]
        # create the time series of demand
        demand_ts = pd.Series(index=model.snapshots, name=zone)
        for year in years:
            annual_profile = add_leap_day(profile["value"], year)
            annual_scale = forecast_peaks[forecast_peaks["year"] == year]["value"].values
            demand_ts.loc[(year, slice(None))] = annual_profile.values * annual_scale
        model.add("Load", zone, bus=zone, p_set=demand_ts)
    return model


def add_leap_day(profile, year):
    if not calendar.isleap(year):
        return profile

    s = profile.copy()
    # last day (24 hours)
    extra = s.iloc[-24:]
    # give new indices
    extra.index = range(len(s), len(s) + 24)
    return pd.concat([s, extra])


def load_input_data(data_path):
    # zones
    zones_df = pd.read_csv(data_path + "zones.csv")
    # time periods
    years_df = pd.read_csv(data_path + "years.csv")
    # demand 
    demandforecast_df = pd.read_csv(data_path + "demandforecast.csv")
    demandprofile_df = pd.read_csv(data_path + "demandprofile.csv")
    # generator data
    generator_constants_df = pd.read_csv(data_path + "generator_constants.csv")
    generator_costs_df = pd.read_csv(data_path + "generator_costs.csv")
    """
    existing_generator_df = pd.read_csv(data_path + "existing_generators.csv")
    generator_df = pd.read_csv(data_path + "generators.csv")
    pnom_pu_df = pd.read_csv(data_path + "pnom_pu.csv")
    existing_storage_df = pd.read_csv(data_path + "existing_storage.csv")
    storage_df = pd.read_csv(data_path + "storage.csv")
    existing_load_df = pd.read_csv(data_path + "existing_load.csv")
    load_df = pd.read_csv(data_path + "load.csv")
    """
    return {
        "zones": zones_df,
        "years": years_df,
        "demandforecast": demandforecast_df,
        "demandprofile": demandprofile_df,
        "generator_constants": generator_constants_df,
        "generator_costs": generator_costs_df,
    }


def create_zones(model, zones_df):
    for zone in zones_df:
        model.add("Bus", zone["name"], carrier="AC")
    return model