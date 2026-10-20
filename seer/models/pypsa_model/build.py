"""Build a PyPSA Network from seer-schema input CSVs."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from ..utils.time_utils import build_snapshot_calendar, compute_time_weights, load_time_weights, load_times, resolve_profile


def build(data_path, annualize_capex=False):
    """annualize_capex: charge build options' capex as annual payments over their life instead
    of once in the build year (see build_year_capital_cost), as in the pyomo backend."""
    input_data = load_data(data_path)
    model = pypsa.Network()

    """TIME: Set up multi-period optimization.
    Years are investment periods. Build decisions can be made in each year.
    times.csv defines the within-year dispatch grid (see seer/schema/README.md); years x times,
    projected onto each real year (skipping e.g. Feb 29 in non-leap years), gives the snapshots.
    """
    years = input_data["years"]["year"].tolist()
    calendar = build_snapshot_calendar(years, input_data["times"], input_data["time_weights"])
    model.investment_periods = years
    model.investment_period_weightings["years"] = get_year_steps(years)

    # discount each period's objective terms relative to the first modeled year
    discount_rate = input_data["settings"].set_index("parameters")["value"]["dr"]
    start_at = 0
    for period, nyears in model.investment_period_weightings.years.items():
        nyears = int(nyears)
        discounts = [(1 / (1 + discount_rate) ** t) for t in range(start_at, start_at + nyears)]
        model.investment_period_weightings.at[period, "objective"] = sum(discounts)
        start_at += nyears
    model.meta["capex"] = {"annualize": annualize_capex, "discount_rate": float(discount_rate)}

    model.snapshots = pd.MultiIndex.from_arrays([calendar["year"], calendar["datetime"]])

    # how many real hours each modeled snapshot stands for -- 1.0 for a full 8760/8784-hour
    # year, greater than 1 for a representative-days grid (see time_utils.compute_time_weights).
    # PyPSA's own "objective" snapshot weighting is exactly this per-snapshot scale factor,
    # applied on top of (not instead of) investment_period_weightings' per-period scaling above.
    time_weight = compute_time_weights(calendar)
    model.snapshot_weightings["objective"] = [time_weight[row.year, row.hour] for row in calendar.itertuples()]

    """SPACE: Set up components of the model.
    Zones represent the spatial resolution of the model.
    """
    for zone in input_data["zones"]["name"]:
        model.add("Bus", zone, carrier="AC")

    """NETWORK: Lines between zones (network.csv is optional).
    """
    build_network(model, input_data["network"])

    """DEMAND: Set up demand trajectory.
    """
    set_demand(model, input_data["demand_forecast"], input_data["demand_profile"], calendar)
    add_load_shedding(model)

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
    for carrier in gen_carriers | sto_carriers | {LOAD_SHEDDING} | ({"AC"} if len(input_data["network"]) else set()):
        model.add("Carrier", carrier)

    return model


def get_year_steps(years):
    """Number of years each investment period represents (gap to the next period; 1 for the
    last period)."""
    years = np.array(years)
    steps = np.append(np.diff(years), [1])
    return pd.Series(steps, index=years)


def prepare_renewable_profiles(re_profiles, calendar):
    """Resolve renewable_profiles.csv into one wide frame (columns=profile_id) indexed by
    (year, datetime) snapshots -- generator_constants profile_id indicates profile correspondence"""
    index = pd.MultiIndex.from_arrays([calendar["year"], calendar["datetime"]], names=[None, None])
    if re_profiles.empty:
        return pd.DataFrame(index=index)
    resolved = resolve_profile(re_profiles, calendar, key_cols=["profile_id"])
    resolved = resolved.merge(calendar[["year", "hour", "datetime"]], on=["year", "hour"])
    wide = resolved.pivot(index=["year", "datetime"], columns="profile_id", values="value")
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


# value of lost load ($/MWh), as in the pyomo backend (abstract_model.py: model.voll)
VOLL = 1_000_000.0
LOAD_SHEDDING = "load_shedding"


def add_load_shedding(model):
    """One load-shedding generator per zone at VOLL, sized to the zone's peak load -- the
    pyomo backend's lost_load slack. Without it a capacity shortfall is infeasible rather than
    priced. Left out of the output tables (output.py), as lost load is on the pyomo side."""
    for zone, load in model.loads_t.p_set.items():
        model.add("Generator", f"{zone}_{LOAD_SHEDDING}", bus=model.loads.at[zone, "bus"],
                  carrier=LOAD_SHEDDING, p_nom=float(load.max()), marginal_cost=VOLL)


def _marginal_cost(gen_costs, name):
    """gen_costs.loc[(name, "marginal"), :], or all-zero if that row is absent (should
    always be present per seer/schema/README.md, but fall back defensively rather than crash)."""
    if (name, "marginal") not in gen_costs.index:
        return pd.Series(0.0, index=gen_costs.columns)
    return gen_costs.loc[(name, "marginal"), :].fillna(0)


def build_generators(model, gen_constants, gen_costs, re_profiles, modular_expansion=False):
    """Add existing generators and build options to the model. Returns the set of carriers used.

    re_profiles: prepare_renewable_profiles's wide (year, datetime) x profile_id frame."""
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
        # isre guarantees profile_id is set (see epm2seer.generator_params.match_re_profiles)
        profile = re_profiles[gen.profile_id] if is_renewable else None

        if expandable:
            # one PyPSA component per build-year vintage -- no upper bound: a build option
            # stays buildable for the rest of the modeled horizon once its start_year has
            # passed (life governs each vintage's own lifetime, not how long the option to
            # build remains open)
            valid_years = get_valid_years(model, gen.start_year)
            capex = gen_costs.loc[(name, "capital"), :]
            # each year's marginal cost (fuel prices change), not the build year's frozen for the
            # vintage's whole life -- same as existing units below and pyomo's mc[gen, year]
            mc = annual_to_snapshots(model, _marginal_cost(gen_costs, name))
            # PyPSA's own p_nom_mod default is 0.0 (continuous), not NaN -- a generator with
            # no known unit_size must fall back to that default explicitly, even under
            # modular_expansion, rather than passing a raw NaN into the LP
            unit_capacity = gen.get("unit_size") if modular_expansion else None
            if unit_capacity is not None and pd.isna(unit_capacity):
                unit_capacity = 0
            # build limits are optional -- absent means unconstrained. annual_build_limit bounds
            # each vintage on its own; total_build_limit bounds the sum across all vintages, which
            # PyPSA can't express per component -- recorded in model.meta and added as a custom
            # constraint at solve time (see solve.add_total_build_limits)
            annual_limit = gen.get("annual_build_limit")
            annual_limit = float("inf") if annual_limit is None or pd.isna(annual_limit) else float(annual_limit)
            total_limit = gen.get("total_build_limit")
            total_limit = float("inf") if total_limit is None or pd.isna(total_limit) else float(total_limit)
            if total_limit != float("inf") and len(valid_years) > 0:
                model.meta.setdefault("total_build_limits", {})[name] = {
                    "limit": total_limit, "components": [f"{name}_{year}" for year in valid_years],
                }

            for year in valid_years:
                gen_year_name = f"{name}_{year}"
                p_max_pu = mask_renewable_profile(
                    profile, valid_years, year, gen.life
                ) if is_renewable else 1
                model.add("Generator", gen_year_name, bus=zone,
                          carrier=carrier, build_year=year, lifetime=gen.life,
                          p_nom_extendable=True, p_nom_mod=unit_capacity, p_nom_max=min(annual_limit, total_limit),
                          capital_cost=build_year_capital_cost(model, capex[year], year, gen.life),
                          ramp_up_rate=ramp_up_rate, ramp_down_rate=ramp_down_rate)
                model.generators_t.p_max_pu[gen_year_name] = p_max_pu
                model.generators_t.marginal_cost[gen_year_name] = mc

        else:
            # single component covering the unit's whole active lifetime
            years = get_valid_years(model, gen.start_year, gen.start_year + gen.life - 1)
            if len(years) == 0:
                warnings.warn(f"No valid years found for generator {name}.")
                continue

            mc = annual_to_snapshots(model, _marginal_cost(gen_costs, name))
            p_max_pu = profile if is_renewable else 1

            # lifetime from the valid years, not gen.life itself: build_year is the first
            # modeled year, not start_year (gen.life from there would run past its last year,
            # e.g. Botswana's Moz_imports)
            model.add("Generator", name, bus=zone,
                        carrier=carrier, build_year=years[0], lifetime=_existing_lifetime(model, years),
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
            # one PyPSA component per build-year vintage -- no upper bound, same reasoning
            # as build_generators' expandable branch above
            capex = storage_costs.loc[(name, "capital"), :]
            # same NaN -> 0.0 fallback as build_generators above -- PyPSA's p_nom_mod default
            # is 0.0 (continuous), not NaN
            unit_capacity = sto.unit_size_mwh if modular_expansion else None
            if unit_capacity is not None and pd.isna(unit_capacity):
                unit_capacity = 0
            # a StorageUnit's p_nom is power (MW); the schema's storage build quantities and
            # capex are per MWh of energy capacity -- convert via the fixed duration
            # annual limit bounds each vintage; total limit spans all vintages -- recorded in
            # model.meta like generators' (see solve.add_total_build_limits)
            hours = sto.duration_hours
            annual_limit = sto.get("annual_build_limit_mwh")
            annual_limit = float("inf") if annual_limit is None or pd.isna(annual_limit) else annual_limit / hours
            total_limit = sto.get("total_build_limit_mwh")
            total_limit = float("inf") if total_limit is None or pd.isna(total_limit) else total_limit / hours
            valid_years = get_valid_years(model, sto.start_year)
            if total_limit != float("inf") and len(valid_years) > 0:
                model.meta.setdefault("storage_total_build_limits", {})[name] = {
                    "limit": total_limit, "components": [f"{name}_{year}" for year in valid_years],
                }

            for year in valid_years:
                model.add("StorageUnit", f"{name}_{year}", bus=sto.zone,
                    carrier="battery", build_year=year, lifetime=sto.life,
                    p_nom_extendable=True, p_nom_mod=(unit_capacity or 0) / hours,
                    p_nom_max=min(annual_limit, total_limit),
                    max_hours=hours,  # fixed duration
                    efficiency_store=sto.efficiency, efficiency_dispatch=sto.efficiency,
                    # cyclic within each period; add_daily_storage_balance makes it per day
                    cyclic_state_of_charge=True, cyclic_state_of_charge_per_period=True,
                    capital_cost=build_year_capital_cost(model, capex[year] * hours, year, sto.life),
                )

        else:
            # single component covering the unit's whole active lifetime
            years = get_valid_years(model, sto.start_year, sto.start_year + sto.life - 1)
            if len(years) == 0:
                warnings.warn(f"No valid years found for storage {name}.")
                continue

            model.add("StorageUnit", name, bus=sto.zone,
                carrier="battery", build_year=years[0], lifetime=_existing_lifetime(model, years),
                p_nom=sto.capacity_mw, p_nom_extendable=False,
                max_hours=sto.duration_hours,
                efficiency_store=sto.efficiency, efficiency_dispatch=sto.efficiency,
                    # cyclic within each period; add_daily_storage_balance makes it per day
                    cyclic_state_of_charge=True, cyclic_state_of_charge_per_period=True,
            )
    return carriers


def line_name(line):
    start = line.get("start_year")
    suffix = "" if start is None or pd.isna(start) else f"_from{int(start)}"
    return f"line_{line['from']}_{line['to']}{suffix}"


def build_network(model, network, modular_expansion=False):
    """Each network.csv row is one undirected line -- flow either way bounded by its capacity,
    same as the pyomo backend. Modeled as lossless bidirectional Links (a PyPSA Line would need
    impedances the schema doesn't carry): one fixed Link for existing capacity, online from the
    row's start_year (build_year; a zone pair can have several rows with different start years),
    and for an expandable line one extendable Link per build-year vintage from its start_year on,
    with no lifetime limit (lines have no life in the schema). total_build_limit spans all
    vintages -- recorded in model.meta like generators' (see solve.add_total_build_limits)."""
    for _, line in network.iterrows():
        name = line_name(line)
        start = line.get("start_year")
        online = {} if start is None or pd.isna(start) else {"build_year": int(start)}
        model.add("Link", name, bus0=line["from"], bus1=line["to"],
                  p_nom=0.0 if pd.isna(line["capacity"]) else line["capacity"], p_min_pu=-1, carrier="AC",
                  **online)
        if not bool(line.get("expandable", False)) or pd.isna(line.get("expandable", False)):
            continue

        valid_years = get_valid_years(model, line.get("start_year"))
        total_limit = line.get("total_build_limit")
        total_limit = float("inf") if total_limit is None or pd.isna(total_limit) else float(total_limit)
        unit_size = line.get("unit_size") if modular_expansion else None
        unit_size = 0 if unit_size is None or pd.isna(unit_size) else unit_size
        for year in valid_years:
            model.add("Link", f"{name}_{year}", bus0=line["from"], bus1=line["to"], p_min_pu=-1, carrier="AC",
                      build_year=year, lifetime=float("inf"), p_nom_extendable=True, p_nom_mod=unit_size,
                      p_nom_max=total_limit,
                      capital_cost=build_year_capital_cost(model, line["capital"], year, float("inf")))
        if total_limit != float("inf") and len(valid_years) > 0:
            model.meta.setdefault("line_total_build_limits", {})[name] = {
                "limit": total_limit, "components": [f"{name}_{year}" for year in valid_years],
            }


def build_year_capital_cost(model, capex, build_year, lifetime):
    """PyPSA capital_cost that makes a vintage's total capex `capex` ($/unit, total -- see
    seer/schema/README.md) incurred once, in its build year, like the pyomo backend.

    With multi_investment_periods=True (see solve.solve), PyPSA charges capital_cost x p_nom x
    the investment period's objective weighting in EVERY period the vintage is active
    (build_year <= period < build_year + lifetime). Dividing by the sum of those weightings and
    multiplying by the build period's own makes the total exactly capex discounted to the build
    year.

    Under model.meta["capex"]["annualize"] it's instead PyPSA's native annual payment, capex x
    CRF(discount_rate, lifetime), charged in each active period and never past the horizon --
    the same as pyomo's annualized_capex_coefficient. Lines (infinite lifetime) pay capex x
    discount_rate forever; with a zero discount rate they keep the one-off charge, as in pyomo."""
    capex_settings = model.meta.get("capex", {})
    if capex_settings.get("annualize"):
        r = capex_settings["discount_rate"]
        if r > 0:
            return capex * r / (1 - (1 + r) ** -lifetime)
        if lifetime != float("inf"):
            return capex / lifetime
    weights = model.investment_period_weightings["objective"]
    periods = weights.index
    active = (periods >= build_year) & (periods < build_year + lifetime)
    return capex * weights.loc[build_year] / weights[active].sum()


def _existing_lifetime(model, years):
    """Lifetime for an existing unit active in `years` (its valid investment periods): up to the
    period after its last one, or unbounded if that's the last modeled period -- so PyPSA's
    build_year + lifetime activity window ends with the unit's last year, as on the pyomo side."""
    periods = list(model.investment_periods)
    i = periods.index(years[-1])
    return periods[i + 1] - years[0] if i + 1 < len(periods) else float("inf")


def get_valid_years(model, start_year=None, end_year=None):
    """Investment periods that fall within [start_year, end_year]. Either bound may be
    omitted/NaN to mean unbounded on that side -- callers use this for build options
    (expandable=True), which only need a lower bound: once a technology's start_year has
    passed, the model should be free to build it in any later modeled year (life bounds an
    individual vintage's own operating lifetime, via `lifetime=gen.life` on each build-year
    component, not whether building is still allowed)."""
    lo = -float("inf") if start_year is None or pd.isna(start_year) else start_year
    hi = float("inf") if end_year is None or pd.isna(end_year) else end_year
    return [year for year in model.investment_periods if lo <= year <= hi]


def annual_to_snapshots(model, annual_series):
    """Broadcast a year-indexed series onto model.snapshots (forward-filled for any year not
    present in the series)."""
    df = pd.DataFrame(index=model.snapshots)
    df["marginal_cost"] = df.index.get_level_values(0).map(annual_series)
    return df.ffill()


def load_data(data_path):
    """Read the seer-schema input CSVs at `data_path` into a dict of DataFrames."""
    network_path = Path(data_path + "network.csv")
    network = pd.read_csv(network_path) if network_path.exists() else pd.DataFrame(columns=["from", "to", "capacity"])
    return {
        "network": network,
        "zones": pd.read_csv(data_path + "zones.csv"),
        "years": pd.read_csv(data_path + "years.csv"),
        "settings": pd.read_csv(data_path + "settings.csv"),
        "times": load_times(data_path.rstrip("/")),
        "time_weights": load_time_weights(data_path.rstrip("/")),
        "demand_forecast": pd.read_csv(data_path + "demand_forecast.csv"),
        "demand_profile": pd.read_csv(data_path + "demand_profile.csv"),
        "renewable_profiles": pd.read_csv(data_path + "renewable_profiles.csv"),
        "generator_constants": pd.read_csv(data_path + "generator_constants.csv"),
        "generator_costs": pd.read_csv(data_path + "generator_costs.csv"),
        "storage_constants": pd.read_csv(data_path + "storage_constants.csv"),
        "storage_costs": pd.read_csv(data_path + "storage_costs.csv"),
    }
