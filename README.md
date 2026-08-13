# seer

A capacity-expansion modeling toolkit. 
The seer schema generically specifies a power systems capacity expansion model that can 
be instantiated from scratch or by converting other model specifications; the model can
be solved with multiple backends.

## Repo structure

### `seer/`
The installable package. See [seer/README.md](seer/README.md) for package organization
(`inputs/`, `models/`, `viz/`).

### `jobs/`
Jobs to convert data into the seer schema or generate synthetic models.

### `experiments/`
Various experiments exploring machine learning accelerations and conventional solver performance.

### `data/`
Local working data: raw/processed EPM exports, seer-schema datasets (real and synthetic), and
reference material. To be used minimally for shared data.

## Getting started
- Dependencies are managed with `uv` (see `pyproject.toml` / `uv.lock`).
- To generate a synthetic dataset and solve it: see `seer/inputs/synthetic/generate_data.py`
  (or `jobs/generate_synthetic_instances.py` for batches) and `seer/models/pyomo_model`.
- To convert an existing EPM/GAMS model: `jobs/epm_to_seer.py`.
