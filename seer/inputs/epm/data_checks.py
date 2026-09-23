"""Functions that check specific data quality issues and log warnings."""
import logging
import pandas as pd
logger = logging.getLogger(__name__)

def check_demand_forecast(demand_forecast, valid_zones):
    """Check the demand forecast data for quality issues and log warnings."""
    if "type" not in demand_forecast.columns:
        logger.warning("demand_forecast.csv is missing the 'type' column -- needs peak / energy")
        return demand_forecast
    demand_forecast["type"] = demand_forecast["type"].str.lower()
    if not set(demand_forecast["type"].unique()).issubset({"peak", "energy"}):
        logger.warning("demand_forecast.csv has invalid 'type' values -- needs peak / energy")
        return demand_forecast
    bad_zone = ~demand_forecast["zone"].isin(valid_zones)
    if bad_zone.any():
        logger.warning(
            f"demand_forecast.csv: dropping row(s) referencing zone(s) not in zones.csv: "
            f"{sorted(demand_forecast.loc[bad_zone, 'zone'].unique())}"
        )
        demand_forecast = demand_forecast[~bad_zone]
    return demand_forecast


def check_demand_profile(demand_profile, valid_zones):
    """Check the demand profile data for quality issues and log warnings."""
    bad_zone = ~demand_profile["zone"].isin(valid_zones)
    if bad_zone.any():
        logger.warning(
            f"demand_profile.csv: dropping row(s) referencing zone(s) not in zones.csv: "
            f"{sorted(demand_profile.loc[bad_zone, 'zone'].unique())}"
        )
        demand_profile = demand_profile[~bad_zone]
    return demand_profile

def check_zones(zones):
    """Check the zones data for quality issues and log warnings."""
    if "zone" not in zones.columns:
        raise ValueError("zcmap.csv is missing the 'zone' column -- needs zone")
    if "country" not in zones.columns:
        raise ValueError("zcmap.csv is missing the 'country' column -- needs country")
    zones = zones[["zone", "country"]]
    # check and drop duplicate zones
    if zones["zone"].duplicated().any():
        logger.warning("zcmap.csv had duplicate zones that were dropped")
        zones = zones.drop_duplicates(subset="zone", keep="first")
    return zones

def check_network(network, valid_zones):
    """Check the network data for quality issues and log warnings."""
    bad = ~(network["from"].isin(valid_zones) & network["to"].isin(valid_zones))
    if bad.any():
        bad_pairs = sorted(set(network.loc[bad, ["from", "to"]].itertuples(index=False, name=None)))
        logger.warning(f"pTransferLimit.csv: dropping row(s) referencing zone(s) not in zones.csv: {bad_pairs}")
        network = network[~bad]
    return network

def check_generators(generators, storage_plants, valid_zones):
    """Check the generator list for quality issues, log warnings, and return a clean list."""
    dupes = generators["generator"].duplicated(keep=False)
    if dupes.any():
        dup_names = sorted(generators.loc[dupes, "generator"].unique())
        logger.warning(f"Duplicate generator names (keeping first occurrence of each): {dup_names}")
        generators = generators.drop_duplicates(subset="generator", keep="first")

    if not storage_plants.empty:
        is_storage = generators["generator"].isin(storage_plants["plant"])
        if is_storage.any():
            logger.warning(
                f"Excluding generator(s) also in pStorageDataInput.csv (represented as "
                f"storage instead): {sorted(generators.loc[is_storage, 'generator'])}"
            )
            generators = generators[~is_storage]

    bad_zone = ~generators["zone"].isin(valid_zones)
    if bad_zone.any():
        logger.warning(
            f"Excluding generator(s) referencing zone(s) not in zones.csv: "
            f"{sorted(generators.loc[bad_zone, 'zone'].unique())}"
        )
        generators = generators[~bad_zone]

    return generators

def check_capex_trajectory(default_wide, custom_wide, model_years):
    """default_wide/custom_wide: pivoted (index=key, columns=year) capex trajectories.
    default_wide takes priority if it has any data at all; that chosen trajectory must cover
    every modeled year. Returns use_flat_default: True only when neither trajectory has any
    data at all (capex held flat, not declining, in that case)."""
    trajectory = default_wide if not default_wide.empty else custom_wide
    if trajectory.empty:
        logger.warning(
            "No capex trajectory data (pCapexTrajectoriesDefault/Custom) found -- "
            "holding capex flat (no cost decline) across all modeled years."
        )
        return True
    missing = set(model_years) - set(trajectory.columns)
    if missing:
        raise ValueError(f"Capex trajectory is missing modeled year(s): {sorted(missing)}")
    return False

def check_settings(settings):
    """Check settings exists and has a discount rate param."""
    dr_row = settings[settings["parameters"].str.upper() == "DR"]
    if dr_row.empty:
        raise ValueError("pSettings.csv has no 'DR' parameter -- can't determine discount rate.")
    return pd.DataFrame({"parameters": ["dr"], "value": [dr_row["value"].iloc[0]]})