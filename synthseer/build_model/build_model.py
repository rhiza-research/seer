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

    """RENEWABLES: Prepare renewable profiles for use in generator setup.
    """
    re_profiles = input_data["renewable_profiles"]
    # renewable profiles must match the model snapshots index
    re_profiles = prepare_renewable_profiles(model, re_profiles)

    """GENERATION: Set up existing generation and build options.
    """
    build_generators(model, input_data["generator_constants"], input_data["generator_costs"], re_profiles)

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


def prepare_renewable_profiles(model, re_profiles):
    # try to match renewable profiles to snapshots
    times = model.snapshots.get_level_values(1)
    dft = pd.DataFrame({'year': [dt.year for dt in times],
                        'month': [dt.month for dt in times], 
                        'day': [dt.day for dt in times], 
                        'hour': [dt.hour for dt in times]})
    # increment hours to be 1 indexed
    dft["hour"] = dft["hour"] + 1
    # determine tech and zones
    zone_tech = re_profiles[["zone", "tech"]].drop_duplicates()
    for zone, tech in zone_tech.itertuples(index=False):
        profile = re_profiles[(re_profiles["zone"] == zone) & (re_profiles["tech"] == tech)]
        dft = dft.merge(profile[["month", "day", "hour", "value"]], on=["month", "day", "hour"], how="left")
        # make value into a multi-indexed column
        dft = dft.rename(columns={"value": f"{zone}_{tech}"})
    # fill leap day
    dft = fill_missing_times(dft)
    # set the index to the model multi-index
    dft.index = model.snapshots
    # drop the year, month, day, hour columns
    dft = dft.drop(columns=["year", "month", "day", "hour"])
    # turn value columns into a multilevel column
    dft.columns = dft.columns.str.split("_", expand=True)
    return dft

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

def fill_missing_times(profile):
    # fill leap day with the values of the previous day
    # identify value columns (everything except time keys)

    value_cols = profile.columns.difference(["year", "month", "day", "hour"])
    def fill_nearest(g):
        return g.interpolate(method="nearest", limit_direction="both")

    profile[value_cols] = (
        profile
        .sort_values(["year", "hour", "month", "day"])
        .groupby(["year", "hour"], group_keys=False)[value_cols]
        .apply(fill_nearest)
    )
    return profile


def load_input_data(data_path):
    # zones
    zones_df = pd.read_csv(data_path + "zones.csv")
    # time periods
    years_df = pd.read_csv(data_path + "years.csv")
    # demand 
    demandforecast_df = pd.read_csv(data_path + "demandforecast.csv")
    demandprofile_df = pd.read_csv(data_path + "demandprofile.csv")
    # renewable profiles
    renewable_profiles_df = pd.read_csv(data_path + "renewable_profiles.csv")
    # generator data
    generator_constants_df = pd.read_csv(data_path + "generator_constants.csv")
    generator_costs_df = pd.read_csv(data_path + "generator_costs.csv")
    
    return {
        "zones": zones_df,
        "years": years_df,
        "demandforecast": demandforecast_df,
        "demandprofile": demandprofile_df,
        "renewable_profiles": renewable_profiles_df,
        "generator_constants": generator_constants_df,
        "generator_costs": generator_costs_df,
    }


def create_zones(model, zones_df):
    for zone in zones_df:
        model.add("Bus", zone["name"], carrier="AC")
    return model