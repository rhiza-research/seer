import pandas as pd
import numpy as np

def generate_synthetic_load(start="2025-01-01 00:00", end="2050-12-31 23:00", base_growth=0.015):
    dates = pd.date_range(start, end, freq="H")
    loads = []

    for ts in dates:
        year_index = ts.year - 2025
        growth = (1 + base_growth) ** year_index
        
        hour = ts.hour
        dayofyear = ts.dayofyear
        
        # Daily shape (evening peak)
        daily = 0.75 + 0.25 * np.sin((hour - 7) / 24 * 2 * np.pi)
        
        # Seasonal shape (winter peak)
        seasonal = 0.9 + 0.2 * np.cos((dayofyear - 1) / 365 * 2 * np.pi)
        
        load = daily * seasonal
        loads.append(load * growth)

    # Normalize so peak in 2025 = 1
    df = pd.DataFrame({"time": dates, "North": loads})
    peak_2025 = df[df.time.dt.year == 2025]["North"].max()
    df["North"] = df["North"] / peak_2025

    df.to_csv("data/load.csv", index=False)

if __name__ == "__main__":
    generate_synthetic_load()
