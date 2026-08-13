# seer model outputs

Both backends expose the same long-format output tables, each via its own small module (`output.py`,
`network_flows.py`, `storage_soc.py`) so results can be compared/plotted the same way regardless
of which backend solved the model:

- `seer.models.pyomo_model.output.build_output_tables(instance)` /
  `seer.models.pypsa_model.output.build_output_tables(model)` -> `{"generator_capacity": df,
  "storage_capacity": df, "dispatch": df}`
- `seer.models.pyomo_model.network_flows.build_network_flows(instance)` -> a single
  `network_flows` DataFrame (pyomo only for now - see the note below)
- `seer.models.pyomo_model.storage_soc.build_storage_soc(instance)` /
  `seer.models.pypsa_model.storage_soc.build_storage_soc(model)` -> a single `storage_soc`
  DataFrame

## `generator_capacity`
Columns: `zone`, `tech`, `generator`, `year`, `capacity` (MW). One row per (generator, year):
total capacity in service that year (existing capacity, or everything built up to and including
that year for a build option).

## `storage_capacity`
Columns: `zone`, `tech`, `storage`, `year`, `capacity` (MWh). Same annual granularity as
`generator_capacity`.

## `dispatch`
Columns: `zone`, `tech`, `name`, `year`, `hour`, `dispatch` (MW). Hourly; combines generator
output and storage's net dispatch into one table (distinguish them via `tech`/`name`). Positive
= net supply to the grid (generator output, or storage discharging); storage charging draws from
the grid, so its dispatch goes negative.

## `network_flows`
Columns: `from`, `to`, `year`, `hour`, `flow` (MW). Hourly flow on each transmission line.
Positive = flowing from `from` to `to` (see `network.csv`/the pyomo abstract model's `line_flow`
variable). **Pyomo only for now** - PyPSA's model builder doesn't create any `Line` components
from `network.csv` yet (that was deferred - see seer/inputs/README.md's `network.csv` entry), so
there's nothing yet for a PyPSA `network_flows.py` to read.

## `storage_soc`
Columns: `zone`, `tech`, `storage`, `year`, `hour`, `soc` (MWh). Hourly state of charge for each
storage unit.

## `hour` convention
`hour` is a 1-indexed position within that year's representative profile, not a calendar
day-of-year. It's always 1-8760 under the pyomo backend. Under PyPSA, an investment year that's
a calendar leap year gets an extra day (1-8784, see `pypsa_model/build.py`'s leap-day handling) -
so the hour range can differ between backends and between years. Handle that in plotting rather
than assuming a fixed range.

## PyPSA vintage consolidation
PyPSA needs one component per (build option, investment-period vintage) for anything
expandable - e.g. `name_2025`, `name_2030`, ... - since PyPSA requires a distinct component per
build year. `seer.models.pypsa_model._consolidate` collapses these back into a single logical
row per base name (matching the fixed per-option naming already used on the pyomo side), summing
capacity/dispatch/soc across every vintage active in a given year/hour. Shared by that backend's
`output.py` and `storage_soc.py`.
