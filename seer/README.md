# seer package structure

The seer schema specifies a capacity expansion planning model. This schema can be
instantiated from other modeling specifications (such as EPM) or through synthetic
data generation. Given a model specification in the schema, a model can be instantiated
and solved in multiple frameworks, including PyPSA and pyomo.
In future, machine learning solvers will be added to solve a seer model.

The seer package contains tools to convert models to or generate new models in the seer schema. 
It contains tools to instantiate seer schema specifications as PyPSA or pyomo models. 
It also contains tools to visualize and explore model results. The package elements can therefore
be used in the following sequence going from generating model specifications, instantiating models, solving
them and interacting with the outputs.
```
inputs/ (epm2seer)  ->  seer-schema CSVs  ->  seer.scenarios (optional)  ->  seer.models  ->  solved model  ->  seer.viz
```
The seer schema is documented in [seer/schema/README.md](schema/README.md).

## Input conversion (outside the package)
Converting external models into the seer schema lives in the top-level `inputs/` folder
(`gdx`, `epm2seer`), not in `seer/` -- see the repo [README](../README.md).

## `seer/scenarios/`
Generates new seer-schema instances, organized by element. Each element package keeps its
shared reference/library code next to a `perturb` module (and, later, a `create` module):

- **`seer/scenarios/demand/`** - `library.py` (daily load-profile library and feature-matched
  sampling), `perturb.py` (`DemandGrowthPerturbation`, `LoadCurvePerturbation`).
- **`seer/scenarios/costs/`** - `reference.py` (reference cost/fuel tables, trajectory curves),
  `perturb.py` (`CostPerturbation`, `MarginalCostPerturbation`).
- `base.py` (`Perturbation`, `apply_all`) and `daily_profiles.py` (circular-hour resampling
  shared by profile libraries) sit at the top level.

Reference data the scenarios draw on lives in `seer/reference_data/`.

## `seer/models/`
Builds a solvable model from a seer schema model specification. A seer schema can be instantiated
in any of the following frameworks:

- **`seer/models/pyomo_model/`** - pyomo instantiation of the seer model.
- **`seer/models/pypsa_model/`** - pypsa instantiation of the seer model.

## `seer/viz/`
Visualization methods for solved models.
