import numpy as np
import pandas as pd
import pypsa

def get_valid_years(model, start_year=2025, end_year=2051):
    years = model.investment_periods
    valid_years = []
    for year in years:
        if year >= start_year and year <= end_year:
            valid_years.append(year)
    return valid_years

def annual_to_snapshots(model, annual_series):
    snapshots = model.snapshots
    # create a dataframe with snapshots as the index and year column
    df = pd.DataFrame(index=snapshots)
    df["marginal_cost"] = df.index.get_level_values(0).map(annual_series)
    # fill missing values with the previous value
    df = df.ffill()
    return df