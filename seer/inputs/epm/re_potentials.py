"""Build renewable_profiles.csv from EPM input data."""
import pandas as pd

from ._time_convert import convert_season_day_hour


def build_re_potentials(data_path="data/Gambia/"):
    """Process renewable potentials into a seer schema file: renewable_profiles.csv.
    Files that are read containing renewable data:
    - pVREProfile.csv: renewable potentials (by zone, tech, month, day, hour)
    Returns a dataframe: renewable_profiles.
    """
    re_profiles = pd.read_csv(data_path + "pVREProfile.csv")

    # the month column's name varies across sources
    col_name_options = ["month", "Month", "Month of the Year", "season"]
    col = next((c for c in col_name_options if c in re_profiles.columns), None)
    if col is None:
        raise ValueError("No valid month column found in the renewable data.")
    re_profiles = re_profiles.rename(columns={col: "season"})

    re_profiles = convert_season_day_hour(re_profiles)
    return re_profiles[["zone", "tech", "time", "value"]]
