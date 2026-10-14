"""seer: capacity-expansion modeling toolkit.

- seer.schema: the seer-schema instance wrapper and validation
- seer.models: build a runnable model from seer-schema data (pyomo or PyPSA)
- seer.scenarios: perturb (and later create) seer-schema instances, by element
- seer.viz: visualization helpers for solved model instances

Converting external models (EPM) into the seer schema lives outside this package, in inputs/.
"""

from . import schema
from . import models
from . import scenarios
from . import viz

__all__ = ["schema", "models", "scenarios", "viz"]
