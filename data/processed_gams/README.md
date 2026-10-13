# processed_gams

Hand-cleaned per-region EPM csvs, one folder per region -- the input to stage 2 of the
pipeline (`jobs/epm_to_seer.py`, EPM csvs -> seer schema).

Each region folder starts as an as-copied duplicate of `data/extracted_gams/<region>`
(stage 1's output, `jobs/extract_gams.py`, raw `.gdx` -> per-symbol csvs -- always safe to
regenerate, never hand-edited). Files in *this* folder are then edited in place to fix
region-specific data issues that aren't worth building generic handling for in the
conversion pipeline itself.

**Re-running `jobs/extract_gams.py` never touches this folder.** If a region needs to be
re-cleaned from scratch, re-copy it from `data/extracted_gams/<region>`.

Every manual edit should be logged in this file's section for that region: which file(s),
what changed, and why. Each section starts with a counts table (`original` = count in
`extracted_gams`, `retained` = count in the final seer schema, `expandable`/`existing` =
split of `retained`). `original - retained` is broken out in a bullet below each table
(storage reclassification + genuine drops).

A generator is expandable if it has cost data (at minimum `Capex`) and either a real
`BuildLimitperYear`/`MaxTotalBuild` or none at all (an unconstrained build option is valid --
see `seer/inputs/epm/generator_params.py`). It's existing/non-expandable if it has a real
`Capacity`. A generator with neither -- no `Capacity` and no `Capex` -- has nothing usable
and is dropped.

Two recurring raw-data issues, fixed generically in code or per-region as noted below:
- **duplicate-of-`Capacity` artifact**: many existing/committed plants (`Status=1`/`2`) have
  `BuildLimitperYear` (sometimes `MaxTotalBuild`) set to the same value as their own
  `Capacity` (or a shared placeholder across many plants) -- not a real build decision, since
  the plant already exists. Fixed per-region by removing `BuildLimitperYear` for those plants.
- **`MaxTotalBuild` fallback**: `generator_params.py` falls back to `MaxTotalBuild` for
  `annual_build_limit` whenever both `BuildLimitperYear` and `Capacity` are missing (real
  sizing data otherwise discarded) -- restricted to that case since `MaxTotalBuild` has the
  same duplicate-artifact problem when `Capacity` is present.

## CAPP

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 417 | 293 | 145 | 148 |

- 13 reclassified as storage; 4 dropped (no `Capacity`, no cost data at all).
- 107 generators (plus associated demand/network rows) are excluded by an existing, unrelated
  `to_seer.py` filter that drops anything referencing a zone not in `zones.csv` --
  `Burundi`/`Rwanda`/`DRC_East`/`Uganda`/`Tanzania` appear throughout CAPP's raw data with
  their own self-contained network (`pTransferLimit.csv`), but zero connections to any real
  CAPP zone -- confirmed cross-region contamination (likely EAPP data leaking into CAPP's raw
  `.gdx`), not a gap in `zcmap.csv`.
- Removed `BuildLimitperYear` for `Status=1/2` generators (duplicate-of-`Capacity` artifact).
- `pFuelPrice.csv` has no `Biomass` entry at all (only `Coal`/`Diesel`/`Gas`/`HFO`), and no
  sibling dataset prices biomass for CAPP's own countries (EAPP only has Kenya/Rwanda/Uganda;
  WAPP only West African countries) -- so `Forzi Akwaya` (Cameroon), `Nkayi Mill`/`Ouesso
  Biomass` (Congo) have no real biomass price to recover anywhere. `marginal_cost` for these
  (and `Forzi Akwaya_expansion`) in `data/seer/from_gams/CAPP/generator_costs.csv` was computed
  by hand using the average of EAPP's and WAPP's own biomass prices as a regional benchmark
  (not a real CAPP-specific number), multiplied by each generator's own `HeatRate` -- `12.5`,
  taken from `pGenDataInputDefault.csv`'s `(Cameroon/Congo, BiomassPlant, Biomass)` default,
  since neither generator has its own custom `HeatRate`. Re-running `jobs.epm_to_seer` for CAPP
  will **not** reproduce this -- it only lives in the committed output CSV.

## EAPP

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 1023 | 1001 | 447 | 554 |

- 20 reclassified as storage; 2 dropped (no `Capacity`, no cost data at all).
- Added `peak` demand rows for `Libya`/`Sudan` (`pDemandForecast.csv`) -- borrowed each
  zone's peak/energy ratio from a comparable neighbor.
- Removed `BuildLimitperYear` for `Status=1/2` generators (duplicate-of-`Capacity` artifact).

## Eswatini

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 41 | 36 | 22 | 14 |

- 4 reclassified as storage; 1 dropped (`SAPP_Connection` -- no `Capacity`, no cost data).
- `zcmap.csv`: collapsed a bidirectional trade-network file to one row per real zone.
- `Battery_Generic`: added missing `Capacity`/`UnitSize`/`BuildLimitperYear` (4h
  battery-duration assumption).

## Gambia

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 30 | 27 | 7 | 20 |

- 3 reclassified as storage; no other drops or edits.

## Guinea

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 41 | 40 | 23 | 17 |

- 1 reclassified as storage; no other drops.
- `zcmap.csv`: collapsed to a single zone.
- Removed `BuildLimitperYear` for 17 `Status=1` generators (duplicate-of-`Capacity` artifact).

## Malawi

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 590 | 530 | 335 | 195 |

- 55 reclassified as storage; 5 dropped (no `Capacity`, no cost data at all).
- `zcmap.csv`: collapsed to 14 real zones.
- Added zone/tech/fuel + `Capacity`/`UnitSize`/`BuildLimitperYear` for 13 `_new` storage
  candidates missing from `pGenDataInputCustom.csv` entirely.
- Code fix (`MaxTotalBuild` fallback) corrected 6 generators with a real, tailored
  `MaxTotalBuild` but no `Capacity`/`BuildLimitperYear`.
- Added `Capacity` (= `CapacityMWh` / 4, standard 4h battery-duration assumption) for 27
  storage plants with a `CapacityMWh` but no `Capacity` at all (`CommittedStorage2-*_NAM`,
  `Committed_Storage1-*_BWA`, `Li-ion_storage_C_MWI`, `SA_IRP_New_BESS_2031..2050`).

## Mexico

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 677 | 668 | 132 | 536 |

- 9 reclassified as storage; no other drops.
- `zcmap.csv`: collapsed a 9-zone bidirectional network to one row per zone.
- Split `SolarTracker` into its own tech (`solar-utility single-axis tracking`), fixing a
  profile collision with plain `Solar`.

## Rwanda

**BLOCKED -- wrong source file.** `data/raw_gams/Rwanda/input.gdx` contains SAPP-region data
(same 13 zones as Malawi's model), not Rwanda. Needs the real Rwanda `.gdx`.

- Added `Capacity` (= `CapacityMWh` / 4, standard 4h battery-duration assumption) for the same
  27 storage plants as Malawi (identical SAPP source data), so the pipeline runs on this
  placeholder data in the meantime.
- `pFuelPrice.csv` has no `FuelOil` entry at all (only `Biomass`/`Coal`/`Diesel`/`Gas`/`LNG`/
  `Uranium`), so every HFO-fueled generator (61 of them, across every SAPP zone) had no real
  price to recover from Rwanda's own data. Since this dataset is already confirmed to be the
  same underlying SAPP-region source as Malawi's (see the BLOCKED note above), and Malawi's own
  `pFuelPrice.csv` *does* have `FuelOil` priced for every one of these same zones (Angola,
  Botswana, DRC, Eswatini, Lesotho, Mozambique, Namibia, South_Africa, Tanzania, Zambia,
  Zimbabwe), `marginal_cost` for all 61 affected generators in
  `data/seer/from_gams/Rwanda/generator_costs.csv` was computed by hand using Malawi's own
  zone-matched `FuelOil` price (not a generic regional average -- an exact same-country,
  same-dataset-family match), multiplied by each generator's own `HeatRate`. Re-running
  `jobs.epm_to_seer` for Rwanda will **not** reproduce this -- it only lives in the committed
  output CSV.

## Serbia

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 277 | 192 | 127 | 65 |

- 83 reclassified as storage; 2 dropped (`RoR_Ex_ME`, `PV_Ex_ME` -- no `Capacity`, no cost
  data at all).
- Added `Capacity`/`UnitSize`/`BuildLimitperYear` for `PHP__Serbia` (8h pumped-hydro
  duration).
- Removed `BuildLimitperYear` for 87 `Status=1/2` generators (duplicate-of-`Capacity`
  artifact, e.g. `TPP_Tuzla3`: `Capacity=BuildLimitperYear=90`).
- Kept 5 `Status=1/2` generators expandable despite no `Capacity`, since their (generic)
  `BuildLimitperYear` is real usable data, not a duplicate artifact.
- `pFuelPrice.csv` is priced per **technology**, not per fuel commodity -- its own "fuel"
  column holds GAMS-ordinal-prefixed tech labels (`1.CoalEx`, `2.CoalNew`, `4.CCGTEx`,
  `5.CCGTNew`, `9.GreenHydrogen`, `15.Biomass`, ...) that match each generator's own raw
  `tech` value once the ordinal prefix is stripped (e.g. `TPP_Tuzla3`'s raw tech is literally
  `CoalEx`). Every other region in this dataset prices by fuel commodity instead, so
  `generator_params.build_costs`'s normal (fuel, zone) merge can't find a match here at all --
  128 generators (every coal/gas/biomass/hydrogen one) ended up with a NaN `marginal_cost`.
  Rather than add a Serbia-specific tech-keyed pricing path to the shared pipeline, these 128
  generators' `marginal_cost` rows in `data/seer/from_gams/Serbia/generator_costs.csv` were
  computed directly (ordinal-stripped tech + zone matched against `pFuelPrice.csv`, multiplied
  by each generator's own `HeatRate`) and patched into the output by hand. Re-running
  `jobs.epm_to_seer` for Serbia will **not** reproduce this fix -- it only lives in the
  committed output CSV.

## Tanzania

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 94 | 89 | 52 | 37 |

- 5 reclassified as storage; no other drops.
- `zcmap.csv`: collapsed to a single zone.
- `pFuelPrice.csv` has no `Biomass` entry at all (only `Coal`/`Diesel`/`FuelOil`/`Gas`/
  `Nuclear`), and EAPP -- the only sibling dataset that also has a `Tanzania` zone -- never
  prices biomass for it either (EAPP's own `Biomass` price only covers Kenya/Rwanda/Uganda).
  `marginal_cost` for `Biomass_TZ_Candidate`/`Biomass_TZ_Candidate_expansion`/`TPC_Bio`/
  `Tanwat_Bio` in `data/seer/from_gams/Tanzania/generator_costs.csv` was computed by hand using
  the same EAPP+WAPP biomass regional average as CAPP (see CAPP's section above), multiplied by
  each generator's own `HeatRate`. Re-running `jobs.epm_to_seer` for Tanzania will **not**
  reproduce this -- it only lives in the committed output CSV.

## Timor Leste

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 27 | 25 | 23 | 2 |

- 2 reclassified as storage; no other drops.
- Corrected `Battery_Generic`'s `CapacityMWh` (1,000,000 -> 20,000 -- implied an implausible
  200h storage duration; used the standard 4h battery-duration assumption instead).

## Turkiye

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 693 | 645 | 357 | 288 |

- 48 reclassified as storage; no other drops.
- `zcmap.csv`: collapsed to 9 real zones, dropping 3 vestigial interconnection labels.
- Removed `BuildLimitperYear` for all 288 `Status=1/2` generators (duplicate-of-`Capacity`
  artifact -- 100% of Turkiye's generators were landing `expandable=True` before this fix).
- Also dropped 99 `*_BESS_Com<year>-1h` storage-candidate slots with zero `CapacityMWh` for
  any year.

## WAPP

| | original | retained | expandable | existing |
|---|---|---|---|---|
| generators | 488 | 459 | 185 | 274 |

- 27 reclassified as storage; 2 dropped (`Genser`, `Ghana_PV` -- no `Capacity`, no cost data
  at all).
- `zcmap.csv`: collapsed a 16-zone bidirectional network to one row per zone.
- Removed `BuildLimitperYear` for 292 `Status=1/2` generators (duplicate-of-`Capacity`
  artifact -- WAPP had only 2 existing generators before this fix, now 274).
- Restored `OFSHORE_WIND_GBI` via the `MaxTotalBuild` fallback code fix.
