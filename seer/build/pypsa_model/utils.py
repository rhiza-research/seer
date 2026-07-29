import pandas as pd

def to_float(x):
    """
    Convert scalars, Series, or DataFrames to float dtype.
    """

    if hasattr(x, "astype"):
        return x.astype(float)

    return float(x)