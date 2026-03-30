import pandas as pd
import numpy as np

def process_re_data(data_path="data/Gambia/"):
    """Process the renewable data from the GAMS input files to a consistent format."""
    re_profiles_path = data_path + "pVREProfile.csv"
    re_profiles = pd.read_csv(re_profiles_path)
    # parse the months column into integers
    col_name_options = ["month", "Month", "Month of the Year", "season"]
    col = None
    for col_name in col_name_options:
        if col_name in re_profiles.columns:
            col = col_name
            break
    if col is None:
        raise ValueError("No valid month column found in the renewable data.")
    # parse the months column into integers - "m1" -> 1, "m2" -> 2, etc.
    re_profiles[col] = re_profiles[col].str.extract('m(\d+)').astype(int)
    # parse the day column into integers - "d1" -> 1, "d2" -> 2, etc.
    re_profiles["day"] = re_profiles["day"].str.extract('d(\d+)').astype(int)
    # parse the hour column into integers - "t1" -> 1, "t24" -> 24, etc.
    re_profiles["hour"] = re_profiles["hour"].str.extract('t(\d+)').astype(int)
    # rename the column to "month"
    re_profiles.rename(columns={col: "month"}, inplace=True)

    return re_profiles

if __name__ == "__main__":
    data_path = "gams2pypsa/data/processed/Gambia/"
    re_profiles = process_re_data(data_path)
    re_profiles.to_csv(data_path + "renewable_profiles.csv", index=False)
