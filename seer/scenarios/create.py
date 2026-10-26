"""Create a seer-schema instance from scratch: zones, network, national demand and existing plants
from gridflow, everything else sampled from reference data. Everything is set per zone.
Examples:
    create("TUR", scenario={"capex": "high", "demand_growth": "low"}, seed=0)
    create("TUR", scenario={"all": "random"}, perturbation=0.5, seed=0, n_days=4, weights=[90, 92, 92, 91])

See scenario_parameters for parameters, groups and items.
"""
import os
import uuid
from pathlib import Path

import gridflow.model
import numpy as np
import pandas as pd

from ..models.utils.time_utils import to_profile_times
from ..schema import SeerInstance
from . import costs, demand_trajectory, profiles
from . import settings as scenario_settings
from .scenario_parameters import get_tech_parameter, load_table, parse

GRIDFLOW_DATA = Path(os.environ.get("GRIDFLOW_DATA", Path(__file__).parents[3] / "gridflow" / "data" / "global"))
RE_POTENTIAL_COLUMN = {"solar": "solar-utility", "wind": "onwind"}  # gridflow's zone_re_potential column per category
UNKNOWN_START_YEAR = 2010  # schema README convention for an existing unit with no known start year
NEVER_RETIRES = 100  # schema README convention for an existing unit with no retirement date
# levels for parameters the scenario doesn't set; re_profile has none, so gridflow's capacity factors stand
DEFAULT_LEVELS = {
    "dr": "mid", "voll": "mid", "demand_growth": "mid", "demand_profile": "random", "hydro": "mid",
    "capex_start": "mid", "capex_pathway": "mid", "fuel_start": "mid", "fuel_pathway": "mid", "line_capex": "mid",
}


def create(country, scenario=None, perturbation=None, seed=None, n_zones=5, years=(2025, 2030, 2035, 2040), n_days=12,
           weights=None, build_options=None, gridflow_data=GRIDFLOW_DATA):
    """Return a new seer instance for country, an ISO3 code (as gridflow expects), with scenario's
    levels set and perturbation's scales applied. Parameters the scenario doesn't set take
    DEFAULT_LEVELS. build_options: techs offered for building in every zone (default: tech_parameters.csv's
    build_option techs)."""
    scenario_options = parse(scenario, perturbation, defaults=DEFAULT_LEVELS, seed=seed)

    # Step 1 - Get gridflow starting model for the country
    region = gridflow.model.region([country],
                                   global_data_path=Path(gridflow_data),
                                   zone_stats_to_load=["population"],
                                   year=int(years[0]))
    region.create_network(method="network", n=n_zones)
    region.set_country_demand()
    region.set_zone_demand_shares()
    region.set_zone_re_potential()
    region.set_zone_plants()

    # Step 2 - planning settings: discount rate and voll
    settings = scenario_settings.sample(scenario_options)

    # Step 3 - Establish time and space parameters
    # space - zones and network
    zones = pd.DataFrame({"zone": region.zones.index, 
                          "country": region.zones["country"].to_numpy(),
                          "name": region.zones.index})
    # time - years and days
    years = pd.DataFrame({"year": [int(y) for y in years]})
    # network - a line for each zone pair with existing capacity or a shared border: gridflow's
    # transfer capacity, expandable at line_capex x gridflow's centroid-to-centroid length
    flow, length, geometry = region.grid.flow, region.grid.length, region.zones.geometry
    pairs = [(a, b) for i, a in enumerate(flow.index) for b in flow.index[i + 1:]
             if flow.loc[a, b] > 0 or geometry[a].intersects(geometry[b])]
    network = pd.DataFrame({
        "from": [a for a, _ in pairs], "to": [b for _, b in pairs], "capacity": [flow.loc[a, b] for a, b in pairs],
        "expandable": True,
        "capital": [costs.sample_line_capex(length.loc[a, b], list(years["year"]), scenario_options) for a, b in pairs],
    })
    # days - n_days representative days; spread evenly over year with equal weight by default
    weights = np.ones(n_days) if weights is None else np.asarray(weights, dtype=float)
    # evenly spaced representative days - the dates assigned to each day have no model meaning
    days = [pd.Timestamp(2025, 1, 1) + pd.Timedelta(days=int((i + 0.5) * 365 / n_days)) for i in range(n_days)]
    hoursofday = np.arange(24.0)
    times = pd.DataFrame({
        "time": to_profile_times([d + pd.Timedelta(hours=h) for d in days for h in hoursofday]),
        "weight": np.repeat(weights, len(hoursofday)),
    })

    # Step 4 - Demand
    # demand profiles - a library day per zone per representative day
    demand_profile = pd.concat([pd.DataFrame({"zone": zone, "time": times["time"], "value": profiles.sample_demand_profile(times, scenario_options)})
                                for zone in zones["zone"]], ignore_index=True)
    # demand trajectory
    demand_forecast = create_demand_forecast(region, demand_profile, times, years, scenario_options)

    # Step 5 - Supply
    # renewable profiles
    renewable_profiles = create_renewable_profiles(region, times, scenario_options)
    # generators and storage: existing units, plus build scenario_options in every zone
    if build_options is None:
        tech_parameters = load_table("technology/tech_parameters.csv")
        build_options = list(tech_parameters.loc[tech_parameters["build_option"], "tech"])
    first_year = int(years["year"].iloc[0])
    generator_constants = create_generator_constants(region, build_options, first_year)
    generator_costs = create_generator_costs(generator_constants, years, scenario_options)
    storage_constants = create_storage_constants(zones, build_options, first_year)
    storage_costs = create_storage_costs(storage_constants, years, scenario_options)

    return SeerInstance(
        settings=settings, zones=zones, years=years, times=times, network=network,
        demand_forecast=demand_forecast, demand_profile=demand_profile, renewable_profiles=renewable_profiles,
        generator_constants=generator_constants, generator_costs=generator_costs,
        storage_constants=storage_constants, storage_costs=storage_costs,
    )


"""Demand"""

def create_demand_forecast(region, demand_profile, times, years, scenario_options):
    """Each zone's peak in gridflow's demand year: its share of national energy divided by its
    profile's annual energy per MW of peak; then grown to the modeled years."""
    national = region.countries.set_index("ISO_A3")
    hours = times["weight"].to_numpy() / times["weight"].sum() * 8760  # hours each timepoint stands for
    rows = []
    for zone, share in region.zone_stats["demand_share"].items():
        country = national.loc[region.zones.loc[zone, "country"]]
        energy_mwh = country["demand_twh"] * 1e6 * share
        peak = energy_mwh / (demand_profile.loc[demand_profile["zone"] == zone, "value"].to_numpy() @ hours)
        demand_year = int(country["demand_year"])
        trajectory = demand_trajectory.sample(peak, [demand_year, *years["year"]], scenario_options)
        rows += [(zone, "peak", y, trajectory[y]) for y in years["year"]]
    return pd.DataFrame(rows, columns=["zone", "type", "year", "value"])


def profile_id(zone, category):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"seer/{zone}/{category}"))

"""Supply"""

def create_renewable_profiles(region, times, scenario_options):
    """One profile per zone per category: solar and wind from gridflow's capacity factor, and hydro
    (for hydro and ror) flat at one capacity factor for every zone -- each per scenario_options."""
    hydro_cf = profiles.sample_hydro(scenario_options)
    frames = []
    for zone, potential in region.zone_re_potential.iterrows():
        for category, column in RE_POTENTIAL_COLUMN.items():
            frames.append(pd.DataFrame({
                "profile_id": profile_id(zone, category), "zone": zone, "tech": column, "generator": None,
                "time": times["time"], "value": profiles.sample_re_profile(potential[column], category, times, scenario_options),
            }))
        frames.append(pd.DataFrame({
            "profile_id": profile_id(zone, "hydro"), "zone": zone, "tech": "hydro", "generator": None,
            "time": times["time"], "value": hydro_cf,
        }))
    return pd.concat(frames, ignore_index=True)


def create_generator_constants(region, build_options, first_year):
    """Existing plants (operating or under construction), aggregated per zone, tech, fuel, start
    year and life; plus a build option per zone for each generator tech in build_options."""
    plants = region.zone_plants[region.zone_plants["Status"].isin(["operating", "construction"])].copy()
    plants["start_year"] = plants["DateIn"].fillna(UNKNOWN_START_YEAR).astype(int)
    plants["life"] = (plants["DateOut"] - plants["start_year"] + 1).fillna(NEVER_RETIRES).astype(int)
    plants["fuel"] = plants["fuel"].fillna("")
    df = plants.groupby(["zone", "tech", "fuel", "start_year", "life"], as_index=False)["Capacity"].sum().rename(columns={"Capacity": "capacity"})
    df["generator"] = df["zone"] + "_" + df["tech"] + "_" + df["fuel"] + "_" + df["start_year"].astype(str) + "_" + df["life"].astype(str)
    df["expandable"] = False
    techs = [t for t in build_options if get_tech_parameter(t, "type") == "generator"]
    build_rows = pd.DataFrame([{
        "generator": f"{zone}_{tech}_new", "zone": zone, "tech": tech, "fuel": get_tech_parameter(tech, "fuel") or "",
        "capacity": np.nan, "expandable": True, "start_year": first_year, "life": int(get_tech_parameter(tech, "life")),
    } for zone in region.zones.index for tech in techs])
    df = pd.concat([df, build_rows], ignore_index=True)
    category = df["tech"].map(lambda t: "hydro" if t in profiles.HYDRO_TECHS else profiles.renewable_category(t))
    df["isre"] = category.notna()
    df["profile_id"] = [profile_id(z, c) if c else None for z, c in zip(df["zone"], category)]
    return df[["generator", "zone", "tech", "fuel", "capacity", "expandable", "isre", "profile_id", "start_year", "life"]]


def create_generator_costs(generator_constants, years, scenario_options):
    """Capital (build scenario_options only) and marginal cost trajectories, one draw per tech (capital) and per
    fuel (marginal), shared across zones."""
    years = list(years["year"])
    rows = []
    for g in generator_constants.itertuples():
        capital = costs.sample_capex(g.tech, years, scenario_options) if g.expandable else np.zeros(len(years))
        marginal = costs.sample_marginal(g.tech, g.fuel, years, scenario_options)
        rows += [[g.generator, "capital", *capital], [g.generator, "marginal", *marginal]]
    return pd.DataFrame(rows, columns=["generator", "cost_type", *[str(y) for y in years]])


def create_storage_constants(zones, build_options, first_year):
    """No existing storage from gridflow; a build option per zone for each storage tech in build_options."""
    techs = [t for t in build_options if get_tech_parameter(t, "type") == "storage"]
    return pd.DataFrame([{
        "storage": f"{zone}_{tech}_new", "zone": zone, "tech": tech, "capacity_mw": np.nan,
        "duration_hours": get_tech_parameter(tech, "duration_hours"), "efficiency": get_tech_parameter(tech, "efficiency"),
        "expandable": True, "start_year": first_year, "life": int(get_tech_parameter(tech, "life")),
    } for zone in zones["zone"] for tech in techs],
        columns=["storage", "zone", "tech", "capacity_mw", "duration_hours", "efficiency", "expandable", "start_year", "life"])


def create_storage_costs(storage_constants, years, scenario_options):
    """Capital ($/MWh of energy, one draw per tech, shared across zones) and zero marginal cost."""
    years = list(years["year"])
    rows = []
    for s in storage_constants.itertuples():
        capital = costs.sample_capex(s.tech, years, scenario_options)
        rows += [[s.storage, "capital", *capital], [s.storage, "marginal", *np.zeros(len(years))]]
    return pd.DataFrame(rows, columns=["storage", "cost_type", *[str(y) for y in years]])
