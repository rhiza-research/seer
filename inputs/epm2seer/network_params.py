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


def _capacity_steps(transfer, years):
    """pTransferLimit (long: from, to, season, year, value) -> network rows. Capacity per zone
    pair and modeled year is the mean over seasons; years before/after the data take its
    nearest year. The first modeled year's capacity is one row with no start_year, and each
    later increase is another row of the added MW with start_year = that year (see
    seer/schema/README.md's network.csv). Lines can't retire, so a decrease is ignored (warned)."""
    by_year = transfer.groupby(["from", "to", "year"])["value"].mean().unstack("year")
    if years is not None:
        by_year = by_year.reindex(columns=sorted(set(by_year.columns) | set(years))).ffill(axis=1).bfill(axis=1)
        by_year = by_year[sorted(years)]
    directed = by_year.reset_index()
    pairs = _one_row_per_pair(directed, list(by_year.columns), "pTransferLimit.csv")
    rows, decreasing = [], []
    for _, pair in pairs.iterrows():
        series = pair[list(by_year.columns)].astype(float)
        rows.append({"from": pair["from"], "to": pair["to"], "capacity": series.iloc[0], "start_year": pd.NA})
        level = series.iloc[0]
        for year, value in series.iloc[1:].items():
            if value > level + 1e-6:
                rows.append({"from": pair["from"], "to": pair["to"], "capacity": value - level, "start_year": year})
                level = value
            elif value < level - 1e-6:
                decreasing.append((pair["from"], pair["to"], year))
    if decreasing:
        warnings.warn(f"pTransferLimit.csv: capacity decreases (ignored -- lines can't retire): {decreasing}")
    return pd.DataFrame(rows, columns=["from", "to", "capacity", "start_year"])


def build_network(input_dir, valid_zones, years=None):
    transfer_path = Path(f"{input_dir}pTransferLimit.csv")
    if transfer_path.exists():
        network = _capacity_steps(check_network(pd.read_csv(transfer_path), valid_zones), years)
    else:
        network = pd.DataFrame(columns=["from", "to", "capacity", "start_year"])
    network["expandable"] = False

    expansion_path = Path(f"{input_dir}pNewTransmission.csv")
    if expansion_path.exists():
        expansion = _expansion_params(pd.read_csv(expansion_path), valid_zones)
        # each candidate corridor is its own row (capacity 0, available from its EarliestEntry --
        # absent means the first modeled year, so it can't collide with the base row), oriented
        # like the pair's existing line so the outputs' (from, to) rows line up
        orientation = {tuple(sorted(k)): k for k in zip(network["from"], network["to"])}
        ends = [orientation.get(tuple(sorted(k)), k) for k in zip(expansion["from"], expansion["to"])]
        expansion["from"], expansion["to"] = [e[0] for e in ends], [e[1] for e in ends]
        expansion["capacity"] = 0.0
        if years is not None:
            expansion["start_year"] = expansion["start_year"].fillna(min(years))
        network = pd.concat([network, expansion], ignore_index=True)

    return network.reindex(columns=NETWORK_COLUMNS)
