"""Build a PyPSA Network from seer-schema input CSVs."""

import calendar
import warnings

import numpy as np
import pandas as pd
import pypsa

# TODO: move to a config csv
discount_rate = 0.08


def build(data_path, time_resolution="1h"):
    input_data = load_data(data_path)
    model = pypsa.Network()

    """TIME: Set up multi-period optimization.
    Years are investment periods. Build decisions can be made in each year.
    Each year has some temporal resolution (snapshots) at which dispatch is modelled.
    """
    years = input_data["years"]["year"].tolist()
    model.investment_periods = years
    model.investment_period_weightings["years"] = get_year_steps(years)

    # discount each period's objective terms relative to the first modeled year
    start_at = 0
    for period, nyears in model.investment_period_weightings.years.items():
        discounts = [(1 / (1 + discount_rate) ** t) for t in range(start_at, start_at + nyears)]
        model.investment_period_weightings.at[period, "objective"] = sum(discounts)
        start_at += nyears

    # snapshots need a (year, timestamp) multi-index to work with investment periods
    snapshots = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"{y}-01-01 00:00", f"{y}-12-31 23:00", freq=time_resolution) for y in years
    ]))
    model.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])

    """SPACE: Set up components of the model.
    Zones represent the spatial resolution of the model.
    """
    for zone in input_data["zones"]["name"]:
        model.add("Bus", zone, carrier="AC")

    """DEMAND: Set up demand trajectory.
    """
    set_demand(model, input_data["demand_forecast"], input_data["demand_profile"])

    """RENEWABLES: Prepare renewable profiles for use in generator setup.
    """
    re_profiles = prepare_renewable_profiles(model, input_data["renewable_profiles"])

    """GENERATION: Set up existing generation and build options.
    """
    gen_carriers = build_generators(model, input_data["generator_constants"], input_data["generator_costs"], re_profiles)

    """STORAGE: Set up existing storage and build options.
    """
    sto_carriers = build_storage(model, input_data["storage_constants"], input_data["storage_costs"])

    """DEFINE CARRIERS: Define carriers in the model.
    """
    for carrier in gen_carriers | sto_carriers:
        model.add("Carrier", carrier)

    return model


def get_year_steps(years):
    """Number of years each investment period represents (gap to the next period; 1 for the
    last period)."""
    years = np.array(years)
    steps = np.append(np.diff(years), [1])
    return pd.Series(steps, index=years)


def prepare_renewable_profiles(model, re_profiles):
    """Reshape renewable_profiles.csv (long format) into a wide (zone, tech)-columned frame
    indexed by model.snapshots, for lookup during generator setup."""
    times = model.snapshots.get_level_values(1)
    dft = pd.DataFrame({
        "year": [dt.year for dt in times], "month": [dt.month for dt in times],
        "day": [dt.day for dt in times], "hour": [dt.hour + 1 for dt in times],  # 1-indexed hours
    })

    zone_tech = re_profiles[["zone", "tech"]].drop_duplicates()

    column_keys = {}
    for zone, tech in zone_tech.itertuples(index=False):
        profile = re_profiles[(re_profiles["zone"] == zone) & (re_profiles["tech"] == tech)]
        dft = dft.merge(profile[["month", "day", "hour", "value"]], on=["month", "day", "hour"], how="left")
        col_name = f"__profile_{len(column_keys)}__"
        dft = dft.rename(columns={"value": col_name})
        column_keys[col_name] = (zone, tech)

    dft = fill_missing_times(dft)  # fills leap day with the nearest hour's value
    dft.index = model.snapshots
    dft = dft.drop(columns=["year", "month", "day", "hour"])
    dft.columns = pd.MultiIndex.from_tuples([column_keys[c] for c in dft.columns], names=["zone", "tech"])
    return dft


def set_demand(model, demand_forecast, demand_profile):
    """Load = normalized hourly profile x annual peak forecast, per zone/year."""
    years = model.investment_periods
    for zone in demand_forecast["zone"].unique():
        profile = demand_profile[demand_profile["zone"] == zone]
        forecast_peaks = demand_forecast[(demand_forecast["zone"] == zone) & (demand_forecast["type"] == "peak")]

        demand_ts = pd.Series(index=model.snapshots, name=zone)
        for year in years:
            annual_profile = add_leap_day(profile["value"], year)
            annual_scale = forecast_peaks[forecast_peaks["year"] == year]["value"].values
            demand_ts.loc[(year, slice(None))] = annual_profile.values * annual_scale
        model.add("Load", zone, bus=zone, p_set=demand_ts)


def add_leap_day(profile, year):
    """Extend a normalized 365-day profile with an extra day (copy of the last day) for leap
    years."""
    if not calendar.isleap(year):
        return profile
    extra = profile.iloc[-24:].copy()
    extra.index = range(len(profile), len(profile) + 24)
    return pd.concat([profile, extra])


def fill_missing_times(profile):
    """Fill the leap-day gap (nearest available hour) in a wide year/month/day/hour profile."""
    value_cols = profile.columns.difference(["year", "month", "day", "hour"])
    fill_nearest = lambda g: g.interpolate(method="nearest", limit_direction="both")
    profile[value_cols] = (
        profile.sort_values(["year", "hour", "month", "day"])
        .groupby(["year", "hour"], group_keys=False)[value_cols].apply(fill_nearest)
    )
    return profile


def build_generators(model, gen_constants, gen_costs, re_profiles, modular_expansion=False):
    """Add existing generators and build options to the model. Returns the set of carriers used."""
    carriers = set()
    gen_constants = gen_constants.set_index("generator")
    gen_costs = gen_costs.set_index(["generator", "cost_type"])
    gen_costs.columns = gen_costs.columns.astype(int)

    if len(gen_constants) == 0:
        warnings.warn("No generators found in the model.")
        return carriers

    for i in range(len(gen_constants)):
        gen = gen_constants.iloc[i]
        name, zone, carrier, expandable, gen_type = gen.name, gen.zone, gen.fuel, gen.expandable, gen.tech
        carriers.add(carrier)
        # ramp rates and unit_size are optional -- absent entirely means unconstrained
        ramp_up_rate = gen.get("ramp_up_rate")
        ramp_down_rate = gen.get("ramp_down_rate")
        is_renewable = bool(gen.isre)

        if expandable:
            # one PyPSA component per build-year vintage
            valid_years = get_valid_years(model, gen.start_year, gen.retirement_year)
            capex = gen_costs.loc[(name, "capex"), :]
            mc = gen_costs.loc[(name, "marginal_cost"), :].fillna(0)
            unit_capacity = gen.get("unit_size") if modular_expansion else None

            for year in valid_years:
                gen_year_name = f"{name}_{year}"
                p_max_pu = mask_renewable_profile(
                    re_profiles[(zone, gen_type)], valid_years, year, gen.life
                ) if is_renewable else 1
                model.add("Generator", gen_year_name, bus=zone,
                          carrier=carrier, build_year=year, lifetime=gen.life,
                          p_nom_extendable=True, p_nom_mod=unit_capacity, p_nom_max=gen.annual_build_limit,
                          marginal_cost=mc[year], capital_cost=capex[year],
                          ramp_up_rate=ramp_up_rate, ramp_down_rate=ramp_down_rate)
                model.generators_t.p_max_pu[gen_year_name] = p_max_pu

        else:
            # single component covering the unit's whole active lifetime
            years = get_valid_years(model, gen.start_year, gen.retirement_year)
            if len(years) == 0:
                warnings.warn(f"No valid years found for generator {name}.")
                continue

            mc = annual_to_snapshots(model, gen_costs.loc[(name, "marginal_cost"), :]).fillna(0)
            p_max_pu = re_profiles.loc[slice(None), (zone, gen_type)] if is_renewable else 1

            model.add("Generator", name, bus=zone,
                        carrier=carrier, build_year=years[0], lifetime=gen.life,
                        p_nom=gen.capacity, p_nom_extendable=False,
                        ramp_up_rate=ramp_up_rate, ramp_down_rate=ramp_down_rate)
            model.generators_t.p_max_pu[name] = p_max_pu
            model.generators_t.marginal_cost[name] = mc
    return carriers


def mask_renewable_profile(profile, valid_years, build_year, lifetime, default=0.0):
    """Zero out (default) a build option's profile outside its active-lifetime window."""
    active_years = set(range(int(build_year), int(build_year) + int(lifetime)))
    years = profile.index.get_level_values(0)
    return profile.where(years.isin(active_years & set(valid_years)), default)


def build_storage(model, storage_constants, storage_costs, modular_expansion=False):
    """Add existing storage units and build options to the model. Returns the carrier set
    (currently always {"battery"} -- see TODO in seer/models/README.md re: carrier parity)."""
    carriers = {"battery"}
    storage_constants = storage_constants.set_index("storage")
    storage_costs = storage_costs.set_index(["storage", "cost_type"])
    storage_costs.columns = storage_costs.columns.astype(int)

    if len(storage_constants) == 0:
        warnings.warn("No storage units found in the model.")
        return carriers

    for i in range(len(storage_constants)):
        sto = storage_constants.iloc[i]
        name, expandable = sto.name, sto.expandable

        if expandable:
            # one PyPSA component per build-year vintage
            capex = storage_costs.loc[(name, "capex"), :]
            unit_capacity = sto.unit_size_mwh if modular_expansion else None

            for year in get_valid_years(model, sto.start_year, sto.retirement_year):
                model.add("StorageUnit", f"{name}_{year}", bus=sto.zone,
                    carrier="battery", build_year=year, lifetime=sto.life,
                    p_nom_extendable=True, p_nom_mod=unit_capacity, p_nom_max=sto.annual_build_limit_mwh,
                    max_hours=sto.duration_hours,  # fixed duration
                    efficiency_store=sto.efficiency, efficiency_dispatch=sto.efficiency,
                    capital_cost=capex[year],
                )

        else:
            # single component covering the unit's whole active lifetime
            years = get_valid_years(model, sto.start_year, sto.retirement_year)
            if len(years) == 0:
                warnings.warn(f"No valid years found for storage {name}.")
                continue

            model.add("StorageUnit", name, bus=sto.zone,
                carrier="battery", build_year=years[0], lifetime=sto.life,
                p_nom=sto.capacity_mw, p_nom_extendable=False,
                max_hours=sto.duration_hours,
                efficiency_store=sto.efficiency, efficiency_dispatch=sto.efficiency,
            )
    return carriers


def get_valid_years(model, start_year=2025, end_year=2051):
    """Investment periods that fall within [start_year, end_year]."""
    return [year for year in model.investment_periods if start_year <= year <= end_year]


def annual_to_snapshots(model, annual_series):
    """Broadcast a year-indexed series onto model.snapshots (forward-filled for any year not
    present in the series)."""
    df = pd.DataFrame(index=model.snapshots)
    df["marginal_cost"] = df.index.get_level_values(0).map(annual_series)
    return df.ffill()


def load_data(data_path):
    """Read the seer-schema input CSVs at `data_path` into a dict of DataFrames."""
    return {
        "zones": pd.read_csv(data_path + "zones.csv"),
        "years": pd.read_csv(data_path + "years.csv"),
        "demand_forecast": pd.read_csv(data_path + "demand_forecast.csv"),
        "demand_profile": pd.read_csv(data_path + "demand_profile.csv"),
        "renewable_profiles": pd.read_csv(data_path + "renewable_profiles.csv"),
        "generator_constants": pd.read_csv(data_path + "generator_constants.csv"),
        "generator_costs": pd.read_csv(data_path + "generator_costs.csv"),
        "storage_constants": pd.read_csv(data_path + "storage_constants.csv"),
        "storage_costs": pd.read_csv(data_path + "storage_costs.csv"),
    }
