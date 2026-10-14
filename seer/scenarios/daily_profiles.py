"""Shared primitives for daily-profile libraries (both demand and renewable): resampling a
native-resolution curve onto an arbitrary hour grid, and circular hour-of-day distance. Signal-
specific feature extraction and sampling strategy live in each element's library module (demand/library.py, later renewables/) -- demand uses
a 2D (load_factor, peak_hour) blend-based k-NN, renewable uses a 1D (capacity_factor)
nearest-pick, different enough that forcing them into one shared function isn't worth it.
"""
import numpy as np


def circular_hour_distance(h1, h2):
    d = abs(h1 - h2) % 24
    return min(d, 24 - d)


def resample_circular(hours, values, target_hours):
    """Evaluate the periodic (24h) curve defined by (hours, values) at arbitrary target_hours,
    via linear interpolation on a 3x-tiled (wrapped) copy of the curve."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(hours)
    hours, values = hours[order], values[order]
    extended_hours = np.concatenate([hours - 24, hours, hours + 24])
    extended_values = np.concatenate([values, values, values])
    return np.interp(target_hours, extended_hours, extended_values)
