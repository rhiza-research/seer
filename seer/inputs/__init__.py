"""Two ways to produce seer-schema input data: convert a GAMS model (seer.inputs.gams)
or generate synthetic data (seer.inputs.synthetic)."""

from . import gams
from . import synthetic

__all__ = ["gams", "synthetic"]
