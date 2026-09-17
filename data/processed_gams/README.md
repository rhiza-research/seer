# processed_gams

Hand-cleaned per-region EPM csvs, one folder per region -- the input to stage 2 of the
pipeline (`jobs/epm_to_seer.py`, EPM csvs -> seer schema).

Each region folder starts as an as-copied duplicate of `data/extracted_gams/<region>`
(stage 1's output, `jobs/extract_gams.py`, raw `.gdx` -> per-symbol csvs -- always safe to
regenerate, never hand-edited). Files in *this* folder are then edited in place to fix
region-specific data issues that aren't worth building generic handling for in the
conversion pipeline itself (duplicate/contaminated rows, naming mismatches between raw
symbols, and the like).

**Re-running `jobs/extract_gams.py` never touches this folder.** If a region needs to be
re-cleaned from scratch, re-copy it from `data/extracted_gams/<region>`.

Every manual edit should be logged in this file's section for that region: which file(s),
what changed, and why -- so the fix can be re-applied (or reconsidered) if the region is
ever re-extracted from a newer raw `.gdx`.

## CAPP

_(no manual changes yet)_

## EAPP

_(no manual changes yet)_

## Eswatini

- **zcmap.csv**: replaced with a clean one-row-per-zone table. The raw file encoded a
  bidirectional zone/trade-partner network (4 rows for "Eswatini", one per partner, plus the
  reverse row for each) rather than a zone->country mapping. Confirmed against
  `pZoneIndex.csv` (the region's own canonical zone list: Eswatini, South_Africa, Mozambique,
  SAPP) that the real zones are exactly those 4 -- "Esw_Trade" was dropped: it has zero
  generators/demand anywhere in the region's data, and only shows up in
  `pExtTransferLimit.csv` as an external (unmodeled) trade boundary label, not a real zone.
  "SAPP" (Southern African Power Pool) is kept -- it has 27 generators attached, a real
  external-market zone with no local demand. Country is set to each zone's own name (no
  better country label available/needed here).

## Gambia

_(no manual changes yet)_

## Guinea

- **zcmap.csv**: replaced with a single zone. The raw file listed the entire WAPP
  interconnection network (Benin, Burkina, Gambia, Ghana, GuineaBissau, IvoryCoast, Liberia,
  Mali, Mauritania_grid, Mauritania_Nema, Niger, Nigeria, Senegal, SierraLeone, Togo) as
  bidirectional trade-partner rows -- but none of those zones have any generators or demand
  anywhere in Guinea's own data (Guinea is a single-zone model; the network rows look
  vestigial, likely copied wholesale from the shared WAPP symbol_mapping). Confirmed against
  `pZoneIndex.csv` (Guinea's own canonical zone list has exactly one entry: "Guinea").
  `pTransferLimit.csv`/`pExtTransferLimit.csv` reference only "Guinea" and "Export_Guinea"
  (an external export/import boundary label, not a real zone -- zero generators/demand, same
  pattern as Eswatini's "Esw_Trade"). Final: `Guinea,Guinea`.

## Malawi

- **zcmap.csv**: replaced with a clean one-row-per-zone table (raw file was a bidirectional
  trade-partner network across the SAPP region). Unlike Guinea, all 14 zones here are real --
  each has generators and/or demand in Malawi's own data (`Mozal`, an aluminium-smelter load,
  has demand but no generators), and `pTransferLimit.csv` references exactly this same set of
  14, nothing extra. Country is self-mapped for each, except `DRC_South` -> `DRC` (same
  country, two zones) and `Mozal` -> `Mozambique` (the smelter is physically in Mozambique).

- **pGenDataInputCustom.csv**: added zone/tech/fuel + `Capacity`/`UnitSize`/`BuildLimitperYear`
  rows for 13 storage candidates (`Sto_AGO_new`, `Sto_BWA_new`, `Sto_DRC_new`,
  `Sto_DRCSouth_new`, `Sto_LSO_new`, `Sto_MOZ_new`, `Sto_MWI_new`, `Sto_NAM_new`,
  `Sto_SWE_new`, `Sto_SWZ_new`, `Sto_TZA_new`, `Sto_ZAF_new`, `Sto_ZMB_new`) that exist in
  `pStorageDataInput.csv` but had no entry here at all -- so they had no zone/tech and were
  being dropped entirely. zone/tech/fuel copied from the identically-named unit without
  `_new` (e.g. `Sto_AGO_new` -> `Sto_AGO`'s zone, Angola). `Capacity` (MW) is derived as
  `CapacityMWh / 4` (a 4-hour duration assumption -- `Capacity` here only ever feeds
  `duration_hours` downstream, never used as an actual capacity ceiling for an expandable
  unit, so this is safe even though the resulting MW figure looks large).  `UnitSize` = 5 MW
  (default). `BuildLimitperYear` = 100,000 MW/year (deliberately large so it never binds --
  the real intent is just "buildable", not a meaningful per-year sub-limit).

## Mexico

- **zcmap.csv**: replaced with a clean one-row-per-zone table. The raw file was a
  bidirectional network among Mexico's own 9 zones (CEN, ORI, OCC, NOR, NTE, NES, PEN, SIBC,
  SIBCS) -- every "country" value was just another Mexican zone, no genuine country label
  anywhere. Confirmed against `pZoneIndex.csv` that these are exactly the 9 real zones, all
  with generators and demand. Since it's a single-country model, country is set to "Mexico"
  for all 9.

## Rwanda

_(no manual changes yet)_

## Serbia

_(no manual changes yet)_

## Tanzania

_(no manual changes yet)_

## Timor Leste

_(no manual changes yet)_

## Turkiye

_(no manual changes yet)_

## WAPP

_(no manual changes yet)_
