"""Create a seer-schema instance from scratch: zones, network, national demand and existing plants
from gridflow, everything else sampled from reference data. Everything is set per zone.
Examples:
    create("TUR", "high-capex, low-demand-growth", seed=0)
    create("TUR", "random, scale-0.5", seed=0, n_days=4, weights=[90, 92, 92, 91])
"""
import os
import uuid
from pathlib import Path

import gridflow.model
import numpy as np
import pandas as pd

from ..models.utils.time_utils import to_profile_times
from ..schema import SeerInstance
from . import settings as scenario_settings
from .costs import CapexSampler, MarginalSampler, per_key_rng, tech_parameter
from .demand_trajectory import DemandTrajectorySampler
from .perturb import parse
from .profiles import LoadProfileSampler, ReProfileSampler, renewable_category
from .spec import REFERENCE, sample_value

GRIDFLOW_DATA = Path(os.environ.get("GRIDFLOW_DATA", Path(__file__).parents[3] / "gridflow" / "data" / "global"))
RE_POTENTIAL_COLUMN = {"solar": "solar-utility", "wind": "onwind"}  # gridflow's zone_re_potential column per category
UNKNOWN_START_YEAR = 2010  # schema README convention for an existing unit with no known start year
NEVER_RETIRES = 100  # schema README convention for an existing unit with no retirement date


def create(country, spec=None, seed=None, n_zones=5, years=(2025, 2030, 2035, 2040), n_days=12, weights=None,
           build_options=None, gridflow_data=GRIDFLOW_DATA):
    """Return a new seer instance for country, an ISO3 code (as gridflow expects).
    Targets not in spec are sampled at mid; the demand profile defaults to random, and renewable
    profiles to gridflow's capacity factors. build_options: techs offered for building in every
    zone (default: scenarios.yaml's build_options)."""
    options = {
        "dr": "mid", "voll": "mid", "demand-growth": "mid", "demand-profile": "random",
        "capex-start": "mid", "capex-pathway": "mid", "fuel-start": "mid", "fuel-pathway": "mid",
        **parse(spec),
    }
    rng = np.random.default_rng(seed)

    # Step 1 - Get gridflow starting model for the country
    region = gridflow.model.region([country],
                                   global_data_path=Path(gridflow_data),
                                   zone_stats_to_load=["population"])
    region.create_network(method="network", n=n_zones)
    region.set_zone_demand_shares()
    region.set_zone_re_potential()
    region.set_zone_plants()

    # Step 2 - planning settings: discount rate and voll
    settings = scenario_settings.sample(options, rng)

    # Step 3 - Establish time and space parameters
    # space - zones and network
    zones = pd.DataFrame({"zone": region.zones.index, 
                          "country": region.zones["country"].to_numpy(),
                          "name": region.zones.index})
    # network - existing lines from gridflow's zone-to-zone transfer capacity, each zone pair once
    lines = region.grid.flow.rename_axis(index="from", columns="to").stack().rename("capacity").reset_index()
    network = lines[(lines["from"] < lines["to"]) & (lines["capacity"] > 0)].reset_index(drop=True)
    # time - years and days
    years = pd.DataFrame({"year": [int(y) for y in years]})
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
    # demand profiles
    demand_profile = create_demand_profile(zones, times, options, rng)
    # demand trajectory
    demand_forecast = create_demand_forecast(region, demand_profile, times, years, options, rng)

    # Step 5 - Supply
    # renewable profiles
    renewable_profiles = create_renewable_profiles(region, times, options, rng)
    # generators and storage: existing units, plus build options in every zone
    build_options = REFERENCE["build_options"] if build_options is None else build_options
    first_year = int(years["year"].iloc[0])
    generator_constants = create_generator_constants(region, build_options, first_year)
    generator_costs = create_generator_costs(generator_constants, years, options, rng)
    storage_constants = create_storage_constants(zones, build_options, first_year)
    storage_costs = create_storage_costs(storage_constants, years, options, rng)

    return SeerInstance(
        settings=settings, zones=zones, years=years, times=times, network=network,
        demand_forecast=demand_forecast, demand_profile=demand_profile, renewable_profiles=renewable_profiles,
        generator_constants=generator_constants, generator_costs=generator_costs,
        storage_constants=storage_constants, storage_costs=storage_costs,
    )


"""Demand"""

def create_demand_profile(zones, times, options, rng):
    """One sampled daily load shape per zone per representative day (peak-normalized)."""
    sampler = LoadProfileSampler()
    n_days = len(times) // len(HOURS)
    frames = []
    for zone in zones["zone"]:
        values = np.concatenate([sampler.sample(options["demand-profile"], HOURS, rng=rng, scale=options["scale"]) for _ in range(n_days)])
        frames.append(pd.DataFrame({"zone": zone, "time": times["time"], "value": values}))
    return pd.concat(frames, ignore_index=True)


def create_demand_forecast(region, demand_profile, times, years, options, rng):
    """Each zone's peak in gridflow's demand year: its share of national energy divided by its
    profile's annual energy per MW of peak; then grown to the modeled years."""
    national = region.countries.set_index("ISO_A3")
    hours = times["weight"].to_numpy() / times["weight"].sum() * 8760  # hours each timepoint stands for
    sampler = DemandTrajectorySampler()
    rows = []
    for zone, share in region.zone_stats["demand_share"].items():
        country = national.loc[region.zones.loc[zone, "country"]]
        energy_mwh = country["demand_twh"] * 1e6 * share
        peak = energy_mwh / (demand_profile.loc[demand_profile["zone"] == zone, "value"].to_numpy() @ hours)
        demand_year = int(country["demand_year"])
        trajectory = sampler.sample(options["demand-growth"], peak, [demand_year, *years["year"]], rng, options["scale"])
        rows += [(zone, "peak", y, trajectory[y]) for y in years["year"]]
    return pd.DataFrame(rows, columns=["zone", "type", "year", "value"])


def profile_id(zone, category):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"seer/{zone}/{category}"))

"""Supply"""

def create_renewable_profiles(region, times, options, rng):
    """One profile per zone per category (solar, wind), matched on gridflow's capacity factor --
    or, if the spec sets re-profile, on a capacity factor sampled at that level around it."""
    sampler = ReProfileSampler()
    n_days = len(times) // len(HOURS)
    frames = []
    for zone, potential in region.zone_re_potential.iterrows():
        for category, column in RE_POTENTIAL_COLUMN.items():
            cf = potential[column]
            if "re-profile" in options:
                cf = sample_value(options["re-profile"], cf, REFERENCE["renewable_profiles"][category], rng, options["scale"])
            curve = sampler.sample({"capacity_factor": cf}, HOURS, category)
            frames.append(pd.DataFrame({
                "profile_id": profile_id(zone, category), "zone": zone, "tech": column, "generator": None,
                "time": times["time"], "value": np.tile(curve, n_days),
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
    techs = [t for t in build_options if tech_parameter(t, "type") == "generator"]
    options = pd.DataFrame([{
        "generator": f"{zone}_{tech}_new", "zone": zone, "tech": tech, "fuel": tech_parameter(tech, "fuel") or "",
        "capacity": np.nan, "expandable": True, "start_year": first_year, "life": int(tech_parameter(tech, "life")),
    } for zone in region.zones.index for tech in techs])
    df = pd.concat([df, options], ignore_index=True)
    category = df["tech"].map(renewable_category)
    df["isre"] = category.notna()
    df["profile_id"] = [profile_id(z, c) if c else None for z, c in zip(df["zone"], category)]
    return df[["generator", "zone", "tech", "fuel", "capacity", "expandable", "isre", "profile_id", "start_year", "life"]]


def create_generator_costs(generator_constants, years, options, rng):
    """Capital (build options only) and marginal (thermal units with a fuel price reference) cost
    trajectories; 0 otherwise."""
    years = list(years["year"])
    capex, marginal = CapexSampler(), MarginalSampler()
    rng_for = per_key_rng(rng)  # one draw per tech (capital) and per fuel (marginal), shared across zones
    rows = []
    for g in generator_constants.itertuples():
        capital = np.zeros(len(years))
        if g.expandable:
            capital = capex.sample(options["capex-start"], options["capex-pathway"], g.tech, years, rng_for(g.tech), options["scale"])
        cost = np.zeros(len(years))
        if marginal.reference(g.tech, g.fuel, years) is not None:
            cost = marginal.sample(options["fuel-start"], options["fuel-pathway"], g.tech, g.fuel, years, rng_for(g.fuel), options["scale"])
        rows += [[g.generator, "capital", *capital], [g.generator, "marginal", *cost]]
    return pd.DataFrame(rows, columns=["generator", "cost_type", *[str(y) for y in years]])


def create_storage_constants(zones, build_options, first_year):
    """No existing storage from gridflow; a build option per zone for each storage tech in build_options."""
    techs = [t for t in build_options if tech_parameter(t, "type") == "storage"]
    return pd.DataFrame([{
        "storage": f"{zone}_{tech}_new", "zone": zone, "tech": tech, "capacity_mw": np.nan,
        "duration_hours": tech_parameter(tech, "duration_hours"), "efficiency": tech_parameter(tech, "efficiency"),
        "expandable": True, "start_year": first_year, "life": int(tech_parameter(tech, "life")),
    } for zone in zones["zone"] for tech in techs],
        columns=["storage", "zone", "tech", "capacity_mw", "duration_hours", "efficiency", "expandable", "start_year", "life"])


def create_storage_costs(storage_constants, years, options, rng):
    """Capital ($/MWh of energy, one draw per tech) and zero marginal cost per storage unit."""
    years = list(years["year"])
    capex = CapexSampler()
    rng_for = per_key_rng(rng)
    rows = []
    for s in storage_constants.itertuples():
        capital = capex.sample(options["capex-start"], options["capex-pathway"], s.tech, years, rng_for(s.tech), options["scale"])
        rows += [[s.storage, "capital", *capital], [s.storage, "marginal", *np.zeros(len(years))]]
    return pd.DataFrame(rows, columns=["storage", "cost_type", *[str(y) for y in years]])
