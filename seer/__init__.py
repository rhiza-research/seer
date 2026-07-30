"""seer: capacity-expansion modeling toolkit.

- seer.inputs: generate seer-schema input data (GAMS conversion or synthetic generation)
- seer.models: build a runnable model from seer-schema data (pyomo or PyPSA)
- seer.viz: visualization helpers for solved model instances
"""

from . import inputs
from . import models
from . import viz

__all__ = ["inputs", "models", "viz"]
