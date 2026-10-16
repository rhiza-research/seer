"""network.csv: one row per zone pair, with optional line-expansion parameters."""

from pathlib import Path
import warnings

import pandas as pd

from .data_checks import check_network

NETWORK_COLUMNS = ["from", "to", "capacity", "expandable", "start_year", "capital", "unit_size", "total_build_limit"]


def _one_row_per_pair(df, value_cols, source):
    """EPM lists each interconnector once per direction (A->B and B->A). seer lines are
    undirected (flow either way bounded by capacity), so keeping both rows would double every
    line's capacity. Keep the first row of each unordered zone pair; warn where the two
    directions disagree (the dropped direction's values are lost)."""
    pair = df.apply(lambda row: tuple(sorted((row["from"], row["to"]))), axis=1)
    differing = []
    for key, group in df.groupby(pair):
        if len(group) > 1 and len(group[value_cols].drop_duplicates()) > 1:
            differing.append(key)
    if differing:
        warnings.warn(f"{source}: the two directions of {sorted(differing)} disagree -- keeping the first row of each")
    return df[~pair.duplicated()].reset_index(drop=True)


def _expansion_params(new_transmission, valid_zones):
    """pNewTransmission (long: from, to, parameters, value) -> one row per candidate corridor.
    CostPerLine is $M per line; capital is $/MW of capacity added (total, incurred once in the
    build year, same as generator capex). The corridor can add up to MaximumNumOfLines lines of
    CapacityPerLine MW each."""
    wide = new_transmission.pivot_table(index=["from", "to"], columns="parameters", values="value").reset_index()
    wide = check_network(wide, valid_zones)
    wide = _one_row_per_pair(wide, [c for c in wide.columns if c not in ("from", "to")], "pNewTransmission.csv")
    return pd.DataFrame({
        "from": wide["from"],
        "to": wide["to"],
        "expandable": True,
        "start_year": wide.get("EarliestEntry"),
        "capital": wide["CostPerLine"] * 1e6 / wide["CapacityPerLine"],
        "unit_size": wide["CapacityPerLine"],
        "total_build_limit": wide["MaximumNumOfLines"] * wide["CapacityPerLine"],
    })


def build_network(input_dir, valid_zones):
    transfer_path = Path(f"{input_dir}pTransferLimit.csv")
    if transfer_path.exists():
        network = check_network(pd.read_csv(transfer_path), valid_zones)
        # original data has time varying capacities, but seer has static -> use average
        network = network.groupby(["from", "to"])["value"].mean().reset_index().rename(columns={"value": "capacity"})
        network = _one_row_per_pair(network, ["capacity"], "pTransferLimit.csv")
    else:
        network = pd.DataFrame(columns=["from", "to", "capacity"])
    network["expandable"] = False

    expansion_path = Path(f"{input_dir}pNewTransmission.csv")
    if expansion_path.exists():
        expansion = _expansion_params(pd.read_csv(expansion_path), valid_zones)
        # match a candidate corridor to its existing line regardless of direction; a corridor
        # with no existing line becomes a new row with zero existing capacity
        def pair_key(df):
            return df.apply(lambda row: tuple(sorted((row["from"], row["to"]))), axis=1)
        network["pair"] = pair_key(network) if len(network) else pd.Series(dtype=object)
        expansion["pair"] = pair_key(expansion)
        existing = network.set_index("pair")
        merged = expansion.set_index("pair").drop(columns=["from", "to"]).combine_first(existing)
        merged[["from", "to"]] = existing[["from", "to"]].reindex(merged.index).fillna(
            expansion.set_index("pair")[["from", "to"]].reindex(merged.index)
        )
        merged["capacity"] = merged["capacity"].fillna(0.0)
        merged["expandable"] = merged["expandable"].fillna(False).astype(bool)
        network = merged.reset_index(drop=True)

    return network.reindex(columns=NETWORK_COLUMNS)
