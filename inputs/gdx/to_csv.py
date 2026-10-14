"""Write processed per-region symbol dataframes out to CSV files."""

from pathlib import Path


def write_to_csv(all_data: dict, output_dir):
    """Write one CSV per symbol per region under output_dir/<region>/<symbol>.csv."""

    # Create the output directory if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop through each country/region and generate csv for each dataframe
    for region_name, region_data in all_data.items():

        # Create region-specific directory
        region_dir = output_dir / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

        # Loop through the symbol dataframes for each region
        for symbol_name, df in region_data.items():

            # Skip writing if dataframe is empty
            if df is None or df.empty:
                continue

            # Define the file path and write out the dataframe to a csv
            df.to_csv(region_dir / f"{symbol_name}.csv", index=False)

        # Print logging message
        print(f"Finished writing all symbols for {region_name}")

    return
