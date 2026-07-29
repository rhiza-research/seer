"""Generate synthetic seer-schema input data (no external model conversion required)."""

from .synth_load import generate_synthetic_load

# synth_build.py has a pre-existing broken import (`from data import sources`/`synth_re_ts`,
# neither of which resolves) and is intentionally left as-is, so it's not re-exported here to
# avoid breaking this package's import. Once fixed, import it directly:
#   from seer.inputs.synthetic import synth_build

__all__ = ["generate_synthetic_load"]
