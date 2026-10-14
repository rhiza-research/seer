# seer

A capacity-expansion modeling toolkit. 
The seer schema generically specifies a power systems capacity expansion model that can 
be instantiated from scratch or by converting other model specifications; the model can
be solved with multiple backends.

## Repo structure

### `seer/`
The installable package. See [seer/README.md](seer/README.md) for package organization
(`schema/`, `models/`, `scenarios/`, `viz/`).

### `inputs/`
Conversion of external model data into the seer schema, kept outside the core package:
- `inputs/gdx/` (imported as `gdx`) - raw EPM GAMS .gdx files -> canonical per-symbol CSVs.
- `inputs/epm2seer/` (imported as `epm2seer`) - EPM per-symbol CSVs -> seer schema.

### `jobs/`
Jobs to convert data into the seer schema and solve models.

### `experiments/`
Various experiments exploring machine learning accelerations and conventional solver performance.

### `data/`
Local working data: raw/processed EPM exports, seer-schema datasets (real and synthetic), and
reference material. To be used minimally for shared data.

## Getting started
- Dependencies are managed with `uv` (see `pyproject.toml` / `uv.lock`).
- To convert an existing EPM/GAMS model: `jobs/extract_gams.py` (gdx -> CSV), then `jobs/epm_to_seer.py`.
- To solve a seer-schema model: `jobs/solve_pypsa_model.py` or `seer/models/pyomo_model`.
