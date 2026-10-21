"""Tokenizer for scenario spec strings (parsed by perturb.parse, used by both create and perturb).

A spec is a string ("high-cost, low-demand-growth" or "high-cost+low-demand-growth"), or an iterable
of such strings (e.g. a set {"high-cost", "low-demand-growth"}), or None.

Also loads reference_data/scenarios.yaml (REFERENCE) and applies its low / mid / high / random rule
(sample_value).
"""
from pathlib import Path

import numpy as np
import yaml

REFERENCE = yaml.safe_load((Path(__file__).parent / "reference_data" / "scenarios.yaml").read_text())


def tokenize(spec):
    """Split a spec into normalized (lowercase, stripped) tokens."""
    if spec is None:
        return []
    parts = [spec] if isinstance(spec, str) else list(spec)
    tokens = []
    for part in parts:
        for token in part.replace("+", ",").split(","):
            token = token.strip().lower()
            if token:
                tokens.append(token)
    return tokens


def sample_value(level, start, ref, rng, scale=1.0):
    """low / mid / high: ref[level] with REFERENCE's noise. random: uniform between start and
    min(low, start) / max(high, start), shrunk toward start by scale. Works elementwise on arrays
    (e.g. a cost trajectory) with one draw shared by every element, so a trajectory stays smooth."""
    if level == "random":
        low, high = np.minimum(ref["low"], start), np.maximum(ref["high"], start)
        low, high = start - scale * (start - low), start + scale * (high - start)
        return low + rng.uniform() * (high - low)
    return ref[level] * (1 + rng.uniform(-REFERENCE["noise"], REFERENCE["noise"]))
