# seer models

Two backends implement the same seer-schema capacity-expansion model: `seer.models.pypsa_model`
(built on PyPSA) and `seer.models.pyomo_model` (built on Pyomo).

## Creating a model

- **`seer.models.pypsa_model.build(data_path)`** reads the seer-schema CSVs at `data_path` and
  builds a PyPSA `Network` directly including buses, generators, storage units, etc.
- **`seer.models.pyomo_model.build(data_path)`** does the same, but Pyomo separates the model
  *structure* from its *data*: `seer_base_model()` in `abstract_model.py` declares an
  `AbstractModel` (sets, params, variables, constraints, objective) independent of any data,
  and `build.py`'s `load_data()` reads the seer schema into a data dict that's used to create a
  concrete instance of it.

Both expose `build(data_path)`; call `solve(model)` (also exported by
both) to solve the built model and print a summary.

## Solved outputs

Both backends expose the same long-format output tables via `output.py`'s
`build_output_tables(model)`, so results can be compared/plotted the same way regardless of
which backend solved the model. Returns `{"generator_capacity": df, "storage_capacity": df,
"dispatch": df}`:

| table | columns | granularity | notes |
| --- | --- | --- | --- |
| `generator_capacity` | `zone`, `tech`, `generator`, `year`, `capacity` (MW) | annual | capacity in service that year |
| `storage_capacity` | `zone`, `tech`, `storage`, `year`, `capacity` (MWh) | annual | capacity in service that year |
| `dispatch` | `zone`, `tech`, `name`, `year`, `hour`, `dispatch` (MW) | hourly | generator output and storage net dispatch in one table (distinguish via `tech`/`name`); positive = net supply to the grid, storage charging is negative |

`hour` is a 1-indexed position within that year's representative profile, not a calendar
day-of-year. It's always 1-8760 under pyomo. Under PyPSA, a leap year gets an extra day
(1-8784, see `pypsa_model/build.py`'s leap-day handling) — so the hour range can differ between
backends and between years; don't assume a fixed range when plotting.
