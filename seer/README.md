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
seer.inputs  ->  seer-schema CSVs  ->  seer.models  ->  solved model  ->  seer.viz
```
The seer schema is documented in [seer/schema/README.md](schema/README.md).

## `seer/inputs/`
Produces seer schema data. Currently, two input pathways are supported:

- **`seer/inputs/epm/`** - converts EPM csv model files into the seer schema
  (`epm_to_seer`).
- **`seer/inputs/synthetic/`** - generates seer schema data from scratch (`generate_all_data`),
  representing synthetic models. This package is useful for testing and solver-scaling experiments. Each parameter group (demand, generators, storage, renewables, network) has its own module (`demand_params.py`, `generator_params.py`, etc.).


## `seer/models/`
Builds a solvable model from a seer schema model specification. A seer schema can be instantiated
in any of the following frameworks:

- **`seer/models/pyomo_model/`** - pyomo instantiation of the seer model.
- **`seer/models/pypsa_model/`** - pypsa instantiation of the seer model.

## `seer/viz/`
Visualization methods for solved models.
