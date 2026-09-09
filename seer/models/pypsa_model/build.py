"""Build a PyPSA Network from seer-schema input CSVs."""

import warnings

import numpy as np
import pandas as pd
import pypsa

from ..time_utils import build_snapshot_calendar, load_times, resolve_profile

# TODO: move to a config csv
discount_rate = 0.08


def build(data_path):
    input_data = load_data(data_path)
    model = pypsa.Network()

    """TIME: Set up multi-period optimization.
    Years are investment periods. Build decisions can be made in each year.
    times.csv defines the within-year dispatch grid (see seer/inputs/README.md); years x times,
    projected onto each real year (skipping e.g. Feb 29 in non-leap years), gives the snapshots.
    """
    years = input_data["years"]["year"].tolist()
    calendar = build_snapshot_calendar(years, input_data["times"])
    model.investment_periods = years
    model.investment_period_weightings["years"] = get_year_steps(years)

    # discount each period's objective terms relative to the first modeled year
    start_at = 0
    for period, nyears in model.investment_period_weightings.years.items():
        nyears = int(nyears)
        discounts = [(1 / (1 + discount_rate) ** t) for t in range(start_at, start_at + nyears)]
        model.investment_period_weightings.at[period, "objective"] = sum(discounts)
        start_at += nyears

    model.snapshots = pd.MultiIndex.from_arrays([calendar["year"], calendar["datetime"]])

    """SPACE: Set up components of the model.
    Zones represent the spatial resolution of the model.
    """
    for zone in input_data["zones"]["name"]:
        model.add("Bus", zone, carrier="AC")

    """DEMAND: Set up demand trajectory.
    """
    set_demand(model, input_data["demand_forecast"], input_data["demand_profile"], calendar)

    """RENEWABLES: Prepare renewable profiles for use in generator setup.
    """
    re_profiles = prepare_renewable_profiles(input_data["renewable_profiles"], calendar)

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


def prepare_renewable_profiles(re_profiles, calendar):
    """Resolve renewable_profiles.csv into two wide frames indexed by (year, datetime)
    snapshots, for lookup during generator setup: generic (columns=(zone, tech)) and
    generator-specific (columns=generator name) -- the latter takes priority over the former
    when a generator has both (see seer/inputs/README.md's renewable_profiles.csv entry)."""
    generic = _resolve_wide(re_profiles[re_profiles["tech"].notna()], calendar, ["zone", "tech"])
    specific = _resolve_wide(re_profiles[re_profiles["generator"].notna()], calendar, ["generator"])
    return generic, specific


def _resolve_wide(profiles, calendar, key_cols):
    index = pd.MultiIndex.from_arrays([calendar["year"], calendar["datetime"]], names=[None, None])
    if profiles.empty:
        return pd.DataFrame(index=index)
    resolved = resolve_profile(profiles, calendar, key_cols=key_cols)
    resolved = resolved.merge(calendar[["year", "hour", "datetime"]], on=["year", "hour"])
    # a single key column pivots to a flat column index (e.g. bare generator names); multiple
    # key columns pivot to a MultiIndex (e.g. (zone, tech) tuples)
    col_spec = key_cols[0] if len(key_cols) == 1 else key_cols
    wide = resolved.pivot(index=["year", "datetime"], columns=col_spec, values="value")
    wide.index.names = [None, None]
    return wide


def set_demand(model, demand_forecast, demand_profile, calendar):
    """Load = normalized profile shape x annual peak forecast, per zone/year."""
    peak = demand_forecast[demand_forecast["type"] == "peak"][["zone", "year", "value"]].rename(columns={"value": "peak"})
    resolved = resolve_profile(demand_profile, calendar, key_cols=["zone"])
    resolved = resolved.merge(peak, on=["zone", "year"], how="left")
    resolved = resolved.merge(calendar[["year", "hour", "datetime"]], on=["year", "hour"])
    resolved["demand"] = resolved["value"] * resolved["peak"]

    for zone, group in resolved.groupby("zone"):
        group = group.sort_values(["year", "hour"])
        demand_ts = pd.Series(group["demand"].values, index=pd.MultiIndex.from_arrays([group["year"], group["datetime"]]))
        model.add("Load", zone, bus=zone, p_set=demand_ts)


def _marginal_cost(gen_costs, name):
    """gen_costs.loc[(name, "marginal_cost"), :], or all-zero if that row is absent (should
    always be present per seer/inputs/README.md, but fall back defensively rather than crash)."""
    if (name, "marginal_cost") not in gen_costs.index:
        return pd.Series(0.0, index=gen_costs.columns)
    return gen_costs.loc[(name, "marginal_cost"), :].fillna(0)


def build_generators(model, gen_constants, gen_costs, re_profiles, modular_expansion=False):
    """Add existing generators and build options to the model. Returns the set of carriers used.

    re_profiles: (generic, specific) tuple from prepare_renewable_profiles -- a generator's own
    specific profile takes priority over its tech's generic one when both exist."""
    generic_profiles, specific_profiles = re_profiles
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
        # isre guarantees a matching profile exists (see seer.inputs.epm.to_seer's matching
        # step) -- specific (by this generator's own name) takes priority over generic
        profile = specific_profiles[name] if name in specific_profiles.columns else generic_profiles.get((zone, gen_type))

        if expandable:
            # one PyPSA component per build-year vintage
            valid_years = get_valid_years(model, gen.start_year, gen.retirement_year)
            capex = gen_costs.loc[(name, "capex"), :]
            mc = _marginal_cost(gen_costs, name)
            unit_capacity = gen.get("unit_size") if modular_expansion else None

            for year in valid_years:
                gen_year_name = f"{name}_{year}"
                p_max_pu = mask_renewable_profile(
                    profile, valid_years, year, gen.life
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

            mc = annual_to_snapshots(model, _marginal_cost(gen_costs, name))
            p_max_pu = profile if is_renewable else 1

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
        "times": load_times(data_path.rstrip("/")),
        "demand_forecast": pd.read_csv(data_path + "demand_forecast.csv"),
        "demand_profile": pd.read_csv(data_path + "demand_profile.csv"),
        "renewable_profiles": pd.read_csv(data_path + "renewable_profiles.csv"),
        "generator_constants": pd.read_csv(data_path + "generator_constants.csv"),
        "generator_costs": pd.read_csv(data_path + "generator_costs.csv"),
        "storage_constants": pd.read_csv(data_path + "storage_constants.csv"),
        "storage_costs": pd.read_csv(data_path + "storage_costs.csv"),
    }
