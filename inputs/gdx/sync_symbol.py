"""GDX symbol registry and canonical-schema column alignment."""

# Import relevant packages
import numpy as np
import pandas as pd

# Symbols whose real dimensionality varies by region (not corruption — the raw
# GAMS symbol itself differs). Keyed by canonical symbol name, then by the
# actual number of non-value columns present.
VARIANT_SCHEMAS = {
    "pVREgenProfile": {
        4: ["generator", "season", "day", "hour", "value"],
        5: ["zone", "tech", "season", "day", "hour", "value"],
    },
}

def build_symbol_registry(mapping_path):
    """
    Build the GDX file symbol mapping schema
    {canonical_symbol: {raw_names: [...], columns: [...]}} from the mapping CSV.
    """

    # Read in the CSV file for symbol mapping
    mapping_df = pd.read_csv(mapping_path)

    # Eliminate any whitespaces in names
    for col in mapping_df.select_dtypes(include="object"):
        mapping_df[col] = mapping_df[col].str.strip()

    # Extract the column names for each symbol
    col_names = [c for c in mapping_df.columns if c.startswith("col_")]

    # Initialize empty dictionary to store symbol mapping
    symbol_registry = {}

    # Build the dictionary
    for _, row in mapping_df.iterrows():

        # Define the raw and standardized symbol name
        resolved = row["canonical"]
        raw_name = row["raw"]

        # Define the columns
        cols = [row[c] for c in col_names if pd.notna(row[c]) and row[c] != ""]

        # Initialize a new canonical symbol if it doesn't exist in dictionary
        if resolved not in symbol_registry:
            symbol_registry[resolved] = {"raw_names": [], "columns": cols}

        # Append raw name aliases to the canonical symbol
        symbol_registry[resolved]["raw_names"].append(raw_name)

    return symbol_registry


def symbol_lookup(raw_symbol, registry):
    """
    Look up the raw symbol name in the symbol registry.
    Return the canonical name for a raw GAMS symbol, or None if unmapped."""

    for resolved_name, dict_items in registry.items():
        if raw_symbol in dict_items["raw_names"]:
            return resolved_name

    return None


def symbol_sync(df, symbol_obj, resolved_name, registry, strict=False):
    """
    Map symbol column names to the canonical schema for resolved_name.
    Aligns dataframe columns to the canonical schema using ordered
    subsequence + left-right anchoring logic.

    Special handling:
        - element_text is protected and mapped explicitly
        - No other column may map to element_text
        - Symbols in VARIANT_SCHEMAS use the schema matching the actual
          number of raw dimensions present, instead of a single fixed schema
    """

    if resolved_name not in registry:
        return df

    # Obtain the maximum expected number of columns for the symbol
    expected_cols = registry[resolved_name].get("columns", [])
    if not expected_cols:
        return df

    # Some symbols have different dimensions across regions.
    # Override the expected schema based on actual number of non-value columns present
    # such that symbol_sync works during operation
    if resolved_name in VARIANT_SCHEMAS:
        n_actual = len([c for c in df.columns if c != "value"]) if "value" in df.columns else len(df.columns) - 1
        variant_cols = VARIANT_SCHEMAS[resolved_name].get(n_actual)
        if variant_cols:
            expected_cols = variant_cols

    # If symbol does not have any columns or symbol df is empty, return the original df
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=expected_cols)

    # Create a copy of the symbol dataframe for column re-naming and processing
    df = df.copy()

    # Check that "value" column exists or rename last column to value
    if "value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "value"})

    # Extract all expected column names from schema excluding "value" column
    expected_dims = [c for c in expected_cols if c != "value"]

    # Extract all actual column names from df excluding "value" column
    actual_dim_cols = [c for c in df.columns if c != "value"]

    # Explicitly map element_text column if present in dataframe
    rename_map = {}

    # Remove element_text from canonical columns for dataframe
    if "element_text" in expected_dims and "element_text" in df.columns:
        rename_map["element_text"] = "element_text"
        actual_dim_cols.remove("element_text")

    # Determine the remaining dimensions that need to be mapped
    remaining_dims = [d for d in expected_dims if d != "element_text"]

    # Determine the total number of columns that need to be mapped
    n = len(actual_dim_cols)

    # Use a left-to-right anchoring scheme for renaming columns
    def left_to_right_mapping(expected_dims, n):
        if n == 0:
            return []

        if n >= len(expected_dims):
            return expected_dims[:n]

        if n == 1:
            return [expected_dims[0]]

        left_anchor = expected_dims[0]
        right_anchor = expected_dims[-1]

        remaining = n - 2
        middle = expected_dims[1:-1]

        return [left_anchor] + middle[:remaining] + [right_anchor]

    # Apply mapping for remaining columns based on column order defined in schema
    mapped_dims = left_to_right_mapping(remaining_dims, n)

    if n < len(remaining_dims):
        dropped = [d for d in remaining_dims if d not in mapped_dims]
        print(f"  [warn] {resolved_name}: only {n} raw dims for {len(remaining_dims)} expected "
              f"({remaining_dims}) — dropping {dropped}, mapped as {mapped_dims}")

    # Assign renamed dimensions
    for i in range(len(mapped_dims)):
        rename_map[actual_dim_cols[i]] = mapped_dims[i]
    df = df.rename(columns=rename_map)

    # Add extra columns that are not in the actual df but in expected column schema as NaNs
    for col in expected_dims:
        if col not in df.columns:
            df[col] = np.nan

    # If "value" column doens't exist in the dataframe, add column with NaNs
    if "value" not in df.columns:
        df["value"] = np.nan

    # Reorder columns to match the expected column schema
    df = df[expected_cols]

    # Raise ValueError if mismatch across expected schema and actual columns
    if strict and list(df.columns) != expected_cols:
        raise ValueError(f"{resolved_name}: schema mismatch after alignment")

    return df
