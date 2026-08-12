# seer schema capacity expansion model specification
The seer schema consists of a set of standard csv files that specify a capacity planning
model spanning one or more zones (which could correspond to regions or countries) and years.
The seer schema can be produced by converting EPM models to the schema (as done in `seer/inputs/gams`)
or by generating a synthetic model (as done in `seer/inputs/synthetic`). 

Every file is a flat csv with a header row.

## Contents

- [Time and space parameters](#time-and-space-parameters)
  - [`zones.csv`](#zonescsv)
  - [`years.csv`](#yearscsv)
  - [`network.csv`](#networkcsv)
- [Demand parameters](#demand-parameters)
  - [`demandforecast.csv`](#demandforecastcsv)
  - [`demandprofile.csv`](#demandprofilecsv)
- [Renewable potentials](#renewable-potentials)
  - [`renewable_profiles.csv`](#renewable_profilescsv)
- [Generator parameters](#generator-parameters)
  - [`generator_constants.csv`](#generator_constantscsv)
  - [`generator_costs.csv`](#generator_costscsv)
- [Storage parameters](#storage-parameters)
  - [`storage_constants.csv`](#storage_constantscsv)
  - [`storage_costs.csv`](#storage_costscsv)

## time and space parameters.
### `zones.csv`
One row per model zone.
Number of rows is zones.

| column | contents |
| --- | --- |
| `zone` | short zone id, referenced by every other file's `zone` column |
| `country` | country this zone belongs to |
| `name` | display name |

### `years.csv`
Years over which to run planning; each year is an investment period, i.e. build decisions happen
at an annual cadence.
Number of rows is years.

| column | contents |
| --- | --- |
| `year` | a modeled investment-period year |

### `network.csv`
One row per transmission line connecting two zones. Optional - a dataset with no inter-zone
lines (e.g. single-zone) can omit this file. Line capacity is fixed for now (no line expansion).
Number of rows is lines.

| column | contents |
| --- | --- |
| `from` | zone id at one end of the line (matches `zones.csv`) |
| `to` | zone id at the other end |
| `capacity` | line's transfer capacity, MW (flow is bounded by ± this value in either direction) |

## demand parameters.
### `demandforecast.csv`
Annual demand trajectory per zone.
Number of rows is zones x years (x 2 if both energy and peak are specified).

| column | contents |
| --- | --- |
| `zone` | zone id (matches `zones.csv`) |
| `type` | `peak` (annual peak demand, MW) or `energy` (annual energy demand, GWh). Only `peak` is currently consumed by the model builders — `energy` is carried through but not yet used. |
| `year` | forecast year (matches `years.csv`) |
| `value` | forecast value in the units implied by `type` |

### `demandprofile.csv`
Normalized annual load curve at hourly resolution; a single load curve is used in all years. 
Number of rows is zones x 8760.

| column | contents |
| --- | --- |
| `zone` | zone id |
| `season` | month, encoded as `m1`…`m12` |
| `day` | day of month, encoded as `d1`…`d31` (representative days, not calendar dates) |
| `hour` | hour of day, encoded as `t1`…`t24` |
| `value` | demand as a fraction of annual peak (unitless, ~0–1). At build time this is multiplied by `demandforecast`'s `peak` value for the corresponding year to get MW at that snapshot (`seer/models/pypsa_model/build.py:set_demand`). |

## renewable potentials
### `renewable_profiles.csv`
Normalized annual capacity-factor curve at hourly resolution, per zone/technology; a
single curve is used in all years. Number of rows is zones x renewable techs x 8760.

| column | contents |
| --- | --- |
| `zone` | zone id |
| `tech` | renewable technology; matches `generator_constants.tech` for renewable generators |
| `month` | month of year, `1`–`12` |
| `day` | day of month, `1`–`31` (representative days) |
| `hour` | hour of day, `1`–`24` |
| `value` | capacity factor: available output as a fraction of nameplate capacity (unitless, 0–1) |

## generator parameters
### `generator_constants.csv`
One row per generator (existing unit or a build option), with its fixed technical and
financial parameters. Number of rows is number of generators.

| column | contents |
| --- | --- |
| `generator` | generator name (unique id, used to join `generator_costs.csv`) |
| `zone` | zone id the generator sits in |
| `tech` | technology category |
| `fuel` | fuel/carrier used to generate |
| `capacity` | nameplate capacity, MW (existing generators only; ignored/NaN for build options) |
| `capex` | scaling multiplier (dimensionless) applied to the default capex trajectory when computing `generator_costs.capex` |
| `fompermw` | fixed O&M cost, $/MW/year |
| `minlimitshare` | minimum output as a fraction of capacity |
| `overloadfactor` | allowed overload above nameplate capacity, as a fraction |
| `ramp_down_rate` / `ramp_up_rate` | max ramp rate, fraction of capacity per hour |
| `reservecost` | cost of providing reserve capacity, $/MW |
| `reslimshare` | max reserve contribution as a fraction of capacity |
| `life` | asset lifetime, years |
| `heat_rate` | fuel consumption rate per unit of output (e.g. GJ/MWh); combined with fuel price to derive `marginal_cost` in `generator_costs.csv` |
| `vom` | variable O&M cost, $/MWh |
| `retech` | renewable-technology flag (`1.0`/`-0.0`): whether output is driven by `renewable_profiles.csv` |
| `retirement_year` | last year the unit is available |
| `start_year` | first year the unit is available (or first year it can be built, for build options) |
| `unit_size` | MW increment used for modular capacity expansion |
| `annual_build_limit` | max MW addable per year; NaN/empty for non-expandable (existing) units |
| `expandable` | `True` if this is a build option (capacity decided by the optimizer per investment year), `False` if it's fixed existing capacity |

### `generator_costs.csv`
Cost trajectories per generator, wide by year. Number of rows is generators x 2 (`capex`,
`marginal_cost`); columns are `generator`, `cost_type`, then one column per modeled year.

| column | contents |
| --- | --- |
| `generator` | generator name (matches `generator_constants.generator`) |
| `cost_type` | `capex` ($/MW, annualized capital cost for build options) or `marginal_cost` ($/MWh, fuel price × `heat_rate`) |
| `<year>` (one column per modeled year) | cost value for that year, in the units implied by `cost_type` |

## storage parameters.
### `storage_constants.csv`
One row per storage unit (existing or build option), analogous to
`generator_constants.csv`. Number of rows is number of storage units.

| column | contents |
| --- | --- |
| `storage` | storage unit name (unique id, joins `storage_costs.csv`) |
| `zone` | zone id |
| `carrier` | storage carrier/technology |
| `capacity_mwh` | energy capacity, MWh |
| `duration_hours` | `capacity_mwh / capacity_mw`, hours of storage at rated power |
| `capacity_mw` | power capacity, MW |
| `capex_mwh` | capital cost, $/MWh |
| `efficiency` | round-trip efficiency component (unitless, 0–1), applied to both charging and discharging |
| `life` | asset lifetime, years |
| `start_year` | first year available (or buildable) |
| `retirement_year` | last year available |
| `unit_size_mwh` | MWh increment used for modular capacity expansion |
| `annual_build_limit_mwh` | max MWh addable per year; NaN/empty for non-expandable (existing) units |
| `expandable` | `True` if this is a build option, `False` if fixed existing capacity |

### `storage_costs.csv`
Cost trajectory per storage unit, wide by year. Number of rows is storage units x cost
types (currently just `capex`); columns are `storage`, `cost_type`, then one column per
modeled year.

| column | contents |
| --- | --- |
| `storage` | storage unit name (matches `storage_constants.storage`) |
| `cost_type` | `capex` ($/MWh, annualized capital cost for build options) |
| `<year>` (one column per modeled year) | capex value for that year |
