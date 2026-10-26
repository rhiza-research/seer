# seer package structure

The seer schema specifies a capacity expansion planning model.
The schema is specified in a list of csv's and aims to be expressive but not exhaustive, allowing
for representation of the core functionalities of capacity planning problems. 

A seer schema instance can be created by translating another model specification (e.g. EPM). 

Given a model specified in the schema, the model can be solved in multiple frameworks, here including PyPSA and pyomo.
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
Generates seer-schema instances from a `scenario` (parameter -> level, e.g. `{"fuel:gas": "high",
"capex:solar-utility|onwind": "low"}`) and a `perturbation` (parameter -> 0-1 scale, e.g.
`{"capex_pathway": 0.3}`, or one number for every parameter):

- **`create.py`** - `create(country, scenario, perturbation)` builds a new instance from gridflow
  (zones, network, demand, existing plants) plus sampled reference data.
- **`perturb.py`** - `perturb(instance, scenario, perturbation)` sets / perturbs an existing instance.
- **`scenario_parameters.py`** - reference data; parameters, groups and items, and `get_parameter_levels`
  (every parameter's low / mid / high); `parse` (dicts -> one `ScenarioOptions`) and `sample_value`,
  the one rule for setting / perturbing a value.
- **`settings.py`, `demand_trajectory.py`, `profiles.py`, `costs.py`** - each parameter's `sample`
  (create) and `resample` (perturb) functions.

Reference data lives in `seer/scenarios/reference_data/`: `scenario_data.yaml` (low / mid / high
values), `technology/` (tech parameters, cost start values and pathways), `fuels/` (fuel price
start values and pathways), `network/` (line cost start values and pathways) and `profiles/` (demand and renewable daily-profile libraries).

## `seer/models/`
Builds a solvable model from a seer schema model specification. A seer schema can be instantiated
in any of the following frameworks:

- **`seer/models/pyomo_model/`** - pyomo instantiation of the seer model.
- **`seer/models/pypsa_model/`** - pypsa instantiation of the seer model.

## `seer/viz/`
Visualization methods for solved models.
