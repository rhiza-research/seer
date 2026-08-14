"""Load GAMS EPM .gdx inputs into raw and canonically-aligned dataframes."""

# Import relevant packages and modules
from pathlib import Path
import gamspy as gp
import pandas as pd

from .sync_symbol import build_symbol_registry, symbol_lookup, symbol_sync


def load_gdx_files(data_dir, mapping_path, strict=True, region=None):
    """
    Load every region's input.gdx under data_dir in a single pass, returning
    (all_data, all_data_raw):
        all_data     -- {region: {canonical_symbol_name: df}}, columns aligned
                         to the canonical schema via symbol_sync
        all_data_raw -- {region: {raw_symbol_name: df}}, no alignment applied

    Both dicts share exactly the same region keys, since both come from the
    same directory scan and the same single read of each region's .gdx file.
    all_data_raw exists for symbols (pGenDataExcel, pZoneIndex, ftfindex) that
    enrich_gen_data needs in their raw, un-pivoted form.

    If region is provided, only that region's subdirectory is loaded and processed
    """

    # Build the symbol mapping registry
    registry = build_symbol_registry(mapping_path)

    # Initialize an empty dictionaries to store the data
    all_data = {}
    all_data_raw = {}

    for country_dir in Path(data_dir).iterdir():

        if region is not None and country_dir.name != region:
            continue

        gdx_file = country_dir / "input.gdx"
        if not gdx_file.exists():
            continue

        print(f"Processing {country_dir.name}")

        # Convert all symbols into dataframes
        m = gp.Container()
        m.read(str(gdx_file))

        country_data = {}
        country_data_raw = {}

        for raw_symbol in m.listSymbols():

            # Convert symbol record into dataframe
            symbol_obj = m[raw_symbol]
            df = pd.DataFrame(symbol_obj.records)

            # Store the raw symbol data
            country_data_raw[raw_symbol] = df

            # Find the corresponding resolved symbol name in the symbol registry
            resolved_symbol = symbol_lookup(raw_symbol, registry)
            if resolved_symbol is None:
                continue

            # Rename columns of symbol data and append to country data
            country_data[resolved_symbol] = symbol_sync(df, symbol_obj, resolved_symbol, registry, strict)

        all_data[country_dir.name] = country_data
        all_data_raw[country_dir.name] = country_data_raw

    return all_data, all_data_raw
