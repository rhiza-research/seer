# seer schema capacity expansion model specification
The seer schema consists of a set of standard csv files that specify a capacity planning
model spanning one or more zones (which could correspond to regions or countries) and years.
The seer schema can be produced by converting EPM models to the schema (as done in `inputs/epm2seer`),
and new instances can be derived from existing ones (`seer/scenarios`).

Every file is a csv with a header row.

## Contents

Sections follow a fixed order -- settings, time and space, demand, supply -- matching
`seer.schema.instance.SeerInstance`'s own field order.

- [Settings parameters](#settings-parameters)
  - [`settings.csv`](#settingscsv)
- [Time and space parameters](#time-and-space-parameters)
  - [`zones.csv`](#zonescsv)
  - [`years.csv`](#yearscsv)
  - [`times.csv`](#timescsv)
  - [`network.csv`](#networkcsv)
- [Demand parameters](#demand-parameters)
  - [`demand_forecast.csv`](#demand_forecastcsv)
  - [`demand_profile.csv`](#demand_profilecsv)
- [Supply parameters](#supply-parameters)
  - [`renewable_profiles.csv`](#renewable_profilescsv)
  - [`generator_constants.csv`](#generator_constantscsv)
  - [`generator_costs.csv`](#generator_costscsv)
  - [`storage_constants.csv`](#storage_constantscsv)
  - [`storage_costs.csv`](#storage_costscsv)

## settings parameters
### `settings.csv`
This file holds the model's global settings. Each row is one parameter.

| column | contents |
| --- | --- |
| `parameters` | parameter name |
| `value` | parameter value |

**Required:** a discount rate parameter (`dr`).

## time and space parameters
### `zones.csv`
Each row is one model zone, so the row count is the number of zones.

| column | contents |
| --- | --- |
| `zone` | zone id, referenced by every other file's `zone` column |
| `country` | country this zone belongs to |
| `name` | display name |

**Required:** every zone id referenced elsewhere must be listed here. `country` and `name` are
convenience fields that enable visualization and analysis but aren't used in the
optimization model.

### `years.csv`
This file lists the years to plan over. Each year is an investment period, since build
decisions happen annually. The row count is the number of years.

| column | contents |
| --- | --- |
| `year` | a modeled investment-period year |

**Required:** every modeled year must be included.

### `times.csv`
This is the within-year dispatch grid, which can be at any resolution (hourly, sub-hourly,
or a handful of representative days). It's independent of `years.csv`, which sets the annual
build cadence. The number of rows is the number of dispatch timepoints per year.

| column | contents |
| --- | --- |
| `time` | a time-of-year, formatted `MM-DD HH:MM:SS` (e.g. `01-15 13:00:00`)|

`time` deliberately carries no year, since it is applied to every modeled year (see
`demand_profile.csv`/`renewable_profiles.csv` below). Rows should be in chronological order. 

### `network.csv`
Each row is one transmission line connecting two zones, so the row count is the number of
lines. This file is *optional*: a dataset with no inter-zone lines (e.g. single-zone) can
omit it.

| column | contents |
| --- | --- |
| `from` | zone id at one end|
| `to` | zone id at the other end |
| `capacity` | line's transfer capacity in MW |

## demand parameters
### `demand_forecast.csv`
This file gives the annual demand trajectory for each zone. The row count is zones times
years (times two if both energy and peak are given).

| column | contents |
| --- | --- |
| `zone` | zone id |
| `type` | `peak` (annual peak demand, MW) or `energy` (annual energy demand, GWh). Only `peak` is currently consumed by the model builders — `energy` is carried through but not yet used. |
| `year` | forecast year|
| `value` | forecast value in the units implied by `type` |

**Required:** demand data for every year in `years.csv`.

### `demand_profile.csv`
This file gives a normalized load curve at whatever resolution `times.csv` specifies. The row
count is zones times the rows in `times.csv` (and times years too, only if given per year -
see below).

| column | contents |
| --- | --- |
| `zone` | zone id |
| `time` | matches a `times.csv` value |
| `year` | *optional.* Omit entirely (or leave a single/null value throughout) to give one template curve reused for every modeled year; give a real value per row to provide a distinct curve per year - in which case every (zone, year, time) implied by `years.csv` x `times.csv` must be present. |
| `value` | unitless demand as a fraction of annual peak. This is multiplied by `demand_forecast`'s `peak` value for the corresponding year to get MW at that snapshot. |

## supply parameters
### `renewable_profiles.csv`
This file gives a normalized capacity-factor curve at `times.csv` resolution. Each profile has
a `profile_id` (a uuid), which matches `generator_constants.profile_id` - that's the only
column a generator needs to find its profile. `zone`/`tech`/`generator` are optional,
free-form descriptive metadata; nothing about them is required or matched on.

| column | contents |
| --- | --- |
| `profile_id` | uuid, unique per distinct profile; matches `generator_constants.profile_id` |
| `time` | matches a `times.csv` value |
| `value` | capacity factor: available output as a fraction of nameplate capacity (unitless, 0–1) |
| `zone` | optional metadata: zone id, if useful context for this profile |
| `tech` | optional metadata: renewable technology or resource-class label, if useful context for this profile |
| `generator` | optional metadata: generator name, if useful context for this profile |
| `year` | *optional*, same template-vs-per-year rule as `demand_profile.csv` above |

`generator_constants.isre` (see below) indicates whether a generator has a profile here. Every
`isre=True` generator must have a `profile_id` assigned.

### `generator_constants.csv`
Each row is one generator (an existing unit or a build option), with its fixed technical and
financial parameters. The row count is the number of generators.

| column | contents |
| --- | --- |
| `generator` | generator name (unique id, joins `generator_costs.csv`) |
| `zone` | zone id the generator sits in |
| `tech`* | technology category (e.g. `CCGT`, `solar-utility`, `onwind`) |
| `fuel`* | fuel/carrier (e.g. `gas`, `coal`); empty for renewables/storage/import |
| `capacity` | nameplate capacity, MW. Existing generators only |
| `expandable` | `True` for a build option, `False` for fixed existing capacity |
| **renewable** | |
| `isre` | renewable-technology flag (boolean) |
| `profile_id` | uuid matching a `renewable_profiles.csv` profile. **Required if `isre`** |
| **expandable units** | |
| `life` | asset lifetime, years. **Required if `expandable`** |
| `annual_build_limit` | max MW addable per year. *Optional* -- absent means unconstrained (`inf`) |
| **existing units** | |
| `start_year` | first year available. *Optional* |
| `retirement_year` | last year available. *Optional* |
| **optional, any generator** | |
| `ramp_up_rate` / `ramp_down_rate` | max ramp rate, fraction of capacity per hour |
| `unit_size` | MW increment used for modular capacity expansion |

`tech`/`fuel`\* are aligned to [PyPSA's technology-data](https://github.com/PyPSA/technology-data)
naming.

If `annual_build_limit` is absent on a build option, it defaults to unconstrained (`inf`), the
same default PyPSA uses. If `retirement_year` is absent, the unit stays active through the
full modeled horizon - not unknown or never-active.

### `generator_costs.csv`
This file gives each generator's cost trajectory. There's one column per modeled year (e.g.
`2025`, `2030`, ...), holding the cost value for that year in the units implied by
`cost_type`. **Required:** one row each for `capex` and `marginal_cost`, per generator, with a
column for every year in `years.csv`.

| column | contents |
| --- | --- |
| `generator` | generator name (matches `generator_constants.generator`) |
| `cost_type` | `capex` ($/MW, annualized capital cost for build options) or `marginal_cost` ($/MWh) |

### `storage_constants.csv`
Each row is one storage unit (existing or a build option), analogous to
`generator_constants.csv`. The row count is the number of storage units.

| column | contents |
| --- | --- |
| `storage` | storage unit name (unique id, joins `storage_costs.csv`) |
| `zone` | zone id |
| `tech`* | storage technology/carrier (e.g. `battery storage`, `PHS`) |
| `capacity_mw` | power capacity, MW. Existing units only |
| `duration_hours` | hours of storage at rated power |
| `expandable` | `True` for a build option, `False` for fixed existing capacity |
| **build options** | |
| `life` | asset lifetime, years. **Required if `expandable`** |
| `annual_build_limit_mwh` | max MWh addable per year. **Required if `expandable`** |
| **existing units** | |
| `start_year` | first year available. *Optional* |
| `retirement_year` | last year available. *Optional* |
| **optional, any storage unit** | |
| `efficiency` | round-trip efficiency, applied to both charging and discharging (unitless, 0–1) |
| `unit_size_mwh` | MWh increment used for modular capacity expansion |

`tech`\* is aligned to [PyPSA's technology-data](https://github.com/PyPSA/technology-data)
naming, the same vocabulary as `generator_constants.tech`.

Energy capacity is calculated as `capacity_mw * duration_hours`.
`annual_build_limit_mwh` is empty for existing units. An absent `unit_size_mwh` means
unconstrained (continuous) build. An absent `retirement_year` means the unit stays active
through the full modeled horizon.

### `storage_costs.csv`
This file gives each storage unit's cost trajectory. There's one column per modeled year (e.g.
`2025`, `2030`, ...), holding the cost value for that year in the units implied by
`cost_type`. **Required:** one row each for `capex` and `marginal_cost`, per storage unit, with
a column for every year in `years.csv`.

| column | contents |
| --- | --- |
| `storage` | storage unit name (matches `storage_constants.storage`) |
| `cost_type` | `capex` ($/MWh, annualized capital cost for build options) or `marginal_cost` ($/MWh) |

`marginal_cost` is usually 0: storage doesn't burn fuel, and raw variable-cost data for it is
sparse where present at all, so 0 is used whenever the source data has no real figure.
