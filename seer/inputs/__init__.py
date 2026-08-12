"""Two ways to produce seer-schema input data: convert an EPM model (seer.inputs.epm)
or generate synthetic data (seer.inputs.synthetic)."""

from . import epm
from . import synthetic

__all__ = ["epm", "synthetic"]
