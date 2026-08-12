"""Shared noise helper for synthetic data generation.

Every randomized quantity in seer/inputs/synthetic is a root value scaled by the same kind of
multiplicative noise: ~ N(1, pct), clamped to be non-negative (so nothing ever gets scaled by
a negative factor).
"""

import numpy as np


def noise_multiplier(pct, size=None):
    return np.clip(np.random.normal(1, pct, size=size), 0, None)
