"""Functions that check specific data quality issues and log warnings."""
import logging
import pandas as pd
logger = logging.getLogger(__name__)

def check_demand_forecast(demand_forecast):
    """Check the demand forecast data for quality issues and log warnings."""
    if "type" not in demand_forecast.columns:
        logger.warning("demand_forecast.csv is missing the 'type' column -- needs peak / energy")
        return demand_forecast
    demand_forecast["type"] = demand_forecast["type"].str.lower()
    if not set(demand_forecast["type"].unique()).issubset({"peak", "energy"}):
        logger.warning("demand_forecast.csv has invalid 'type' values -- needs peak / energy")
        return demand_forecast
    return demand_forecast

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

def check_settings(settings):
    """Check settings exists and has a discount rate param."""
    dr_row = settings[settings["parameters"].str.upper() == "DR"]
    if dr_row.empty:
        raise ValueError("pSettings.csv has no 'DR' parameter -- can't determine discount rate.")
    return pd.DataFrame({"parameters": ["dr"], "value": [dr_row["value"].iloc[0]]})