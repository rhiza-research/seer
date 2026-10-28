# {country}: perturbed seer instances

Perturbed versions of the seer instance in `base.zip` (from `{base}` in the seer repo), made with
`seer.scenarios.perturb` (values rounded to {significant_figures} significant figures; cost levels from the
`{cost_dataset}` reference dataset, falling back to the others where it has none), solved with {model}.
The run's settings are in `config.yaml`.

The perturbations come from the runs in `config.yaml`: each run expands its scenario elements (`grid`:
every combination; `iterate`: each element's values alone) and makes `iters` perturbations of each
scenario, each with its own seed; runs marked solve are solved. A perturbation whose inputs equal an
earlier one's isn't solved again (its run.json gives `duplicate_of`).

{runs_table}

## Layout

- `base.zip`: the base instance's seer-schema csvs, as in the seer repo.
- `manifest.csv`: one row per perturbation: id, run, scenario name, scenario, perturbation, noise, seed,
  inputs hash, code version, solve status and objective.
- `perturbations/<id>/perturbation.json`: everything needed to regenerate that perturbation.
- `perturbations/<id>/inputs.zip`: the perturbed instance, as seer-schema csvs (and perturbation.json).
- `perturbations/<id>/outputs/{model}.zip`: for solved perturbations, the output tables (and run.json).
- `perturbations/<id>/outputs/{model}.json`: for solved perturbations, run.json: status, objective, timing.

## perturbation.json

- `base`: the instance perturbed (path in the seer repo, sha256 of its csvs; the same csvs as `base.zip`).
- `design`: the run that made it and its scenario's name (e.g. `capex=low,fuel=high`; `base` for none).
- `perturb`: the arguments to `perturb(base, scenario, perturbation, seed, significant_figures, cost_dataset, noise)`.
- `inputs_sha256`: sha256 of the perturbed instance's tables (equal for identical instances).
- `code`: seer's git commit; whether the tree had uncommitted changes (and a sha256 of that diff);
  sha256 of `seer/scenarios/reference_data`, which holds every parameter's low / mid / high levels.

## Reproducing a perturbation

Install seer at `code.git_commit`, then:

    from seer.ccai_dataset.provenance import regenerate
    instance = regenerate(json.load(open("perturbations/<id>/perturbation.json")), "base.zip")

`regenerate` checks the base instance against `base.sha256`. The result equals `inputs.zip` when
`code.reference_data_sha256` matches and `code.git_dirty` is false.
