"""seer: capacity-expansion modeling toolkit.

- seer.inputs: generate seer-schema input data (GAMS conversion or synthetic generation)
- seer.build: build a runnable model from seer-schema data (pyomo or PyPSA)
"""

from . import inputs
from . import build

__all__ = ["inputs", "build"]
