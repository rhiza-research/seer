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
`build_output_tables(model, data_path)`, so results can be compared/plotted the same way regardless of
which backend solved the model. Returns `{"generator_capacity": df, "storage_capacity": df,
"dispatch": df, "line_capacity": df, "lost_load": df, "line_flow": df, "storage_soc": df, "prices": df,
"summary": df}`:

| table | columns | granularity | notes |
| --- | --- | --- | --- |
| `generator_capacity` | `zone`, `tech`, `fuel`, `generator`, `year`, `capacity` (MW) | annual | capacity in service that year |
| `storage_capacity` | `zone`, `tech`, `fuel`, `storage`, `year`, `capacity` (MWh) | annual | capacity in service that year |
| `dispatch` | `zone`, `tech`, `fuel`, `name`, `year`, `hour`, `datetime`, `dispatch` (MW) | per `times.csv` | generator output and storage net dispatch in one table (distinguish via `tech`/`name`); `zone`/`tech`/`fuel` are joined from the input `generator_constants.csv`/`storage_constants.csv` by name, so both backends report them identically (`fuel` empty for storage); positive = net supply to the grid, storage charging is negative |
| `line_capacity` | `from`, `to`, `year`, `capacity` (MW) | annual | line capacity in service that year (existing + any expansion), one row per zone pair |
| `lost_load` | `zone`, `year`, `hour`, `datetime`, `lost_load` (MW) | per `times.csv` | unserved demand per zone, priced at `settings.csv`'s `voll` (pyomo's `lost_load` variable; PyPSA's per-zone load-shedding generator, which is left out of the other tables) |
| `line_flow` | `from`, `to`, `year`, `hour`, `datetime`, `flow` (MW) | per `times.csv` | power flow per zone pair, positive from `from` to `to` (pyomo's `line_flow`; PyPSA's Link `p0`), summed over a pair's lines like `line_capacity`; empty for a single zone. Flows can differ between backends where several flow patterns are equally cheap |
| `storage_soc` | `zone`, `tech`, `fuel`, `storage`, `year`, `hour`, `datetime`, `soc` (MWh) | per `times.csv` | stored energy at the end of each hour (pyomo's `soc`; PyPSA's `state_of_charge`) |
| `prices` | `zone`, `year`, `hour`, `datetime`, `price` ($/MWh) | per `times.csv` | marginal price of demand in each zone: the dual of the supply = demand constraint, undiscounted and per real hour (pyomo's dual divided by `year_discount_factor` x `time_weight`; PyPSA's `marginal_price`, which PyPSA scales the same way). Empty values for a MILP (`discrete_additions`), which has no duals |
| `summary` | `metric`, `value`, `unit` | whole run | `objective` and the cost terms that sum to it (`generator_capex`, `storage_capex`, `line_capex`, `variable_cost`: fuel + VOM, `lost_load_cost`), all $ discounted to the first modeled year; `demand` and `lost_load`: MWh over all modeled years (each hour x its `times.csv` weight) |

In `dispatch.csv`, time is indexed redundantly via `datetime` and `year`, `hour` column so there are multiple ways to plot, label, or aggregate the data.
