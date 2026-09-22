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

- **pDemandForecast.csv**: added `peak` rows for `Libya` and `Sudan` -- both zones had only
  `energy` demand forecast data in the raw source, no `peak` at all, leaving demand for those
  zones undefined (silently NaN) in both model backends, which only consume `peak`. Filled in
  by applying a peak/energy ratio borrowed from a comparable zone: Libya from Egypt (its
  neighbor, and the most stable ratio in the region, ~0.163 across all years), Sudan from
  South Sudan (its former union partner, ~0.187). Every other EAPP zone's own peak/energy
  ratio is fairly consistent region-wide (0.14-0.20), so the exact choice of comparator
  shouldn't swing results much.

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

- **pGenDataInputCustom.csv**: added `Capacity`/`UnitSize`/`BuildLimitperYear` rows for
  `Battery_Generic` -- it had neither `Capacity` (MW) nor `BuildLimitperYear` at all, so it
  landed as a non-expandable "existing" unit with no defined capacity (broken -- an existing
  unit needs one). Its `CapacityMWh=10,000` in `pStorageDataInput.csv` is a suspiciously round
  ceiling figure, the same "missing build-option parameters" pattern as Malawi's `_new`
  storage candidates -- same fix applied: `Capacity = CapacityMWh / 4` (4h duration
  assumption), `UnitSize = 5` MW, `BuildLimitperYear = 100,000` MW/year (deliberately large,
  never binds).

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

- **pGenDataInputCustom.csv**: removed the `BuildLimitperYear` row for 17 generators whose
  raw `Status` param is `1` (Baneah, Grand Chutes, Amaria, Kaleta, Koukoutamba, Donkea, Fomi,
  Garafiri, Souapiti, Karpowership_IPP, Kipe_IPP, GDE_IPP, Kaloum5, Kaloum3_IPP, Kaloum1_IPP,
  Kaloum2_IPP, Te_IPP) -- every one of Guinea's 40 generators had a non-null
  `BuildLimitperYear` (usually exactly equal to its own `Capacity`), so all 40 were landing as
  "expandable" build options, including plants built decades ago (e.g. Grand Chutes, StYr
  1954). `Status=1` cleanly separates these fixed/committed units (their `BuildLimitperYear`
  looks like a modeling artifact, not a genuine build-optimizer decision) from the real
  candidates (`Status=3`, e.g. `Battery_Generic`/`Geothermal_Generic`/the `PV` candidate).
  Not yet verified whether `Status` carries the same meaning in other regions -- Gambia's raw
  data has a third value (`Status=2`) whose meaning is unclear, so this fix is Guinea-specific
  for now, not rolled into the generic pipeline.

- **`PV` generator investigated, no change made**: initially looked like it had no usable
  renewable profile (only its *generator-specific* candidate in `pVREgenProfile.csv` was
  checked, which genuinely is incomplete -- 288 of 864 times -- and correctly gets dropped by
  `to_seer`'s completeness check). But Guinea's `pVREProfile.csv` already has a complete,
  valid *generic* `(zone=Guinea, tech=PV)` profile, which matches successfully -- confirmed
  `isre=True` with a real profile in the final schema. `Capacity=30000`/`BuildLimitperYear=
  1000` also look like placeholder/ceiling figures, but since this generator is expandable,
  `Capacity` is never actually used by either model backend (see seer/inputs/README.md) --
  only `BuildLimitperYear` (the real growth ceiling) matters, and 1000 MW/year is already
  generous relative to Guinea's system size (other generators range ~5-475 MW), so left as-is.

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

**BLOCKED -- wrong source file, skipped for now.** `data/raw_gams/Rwanda/input.gdx` does not
contain Rwanda data at all: its zones are the same 13 SAPP-region zones as Malawi's model
(Angola, Botswana, DRC, DRC_South, Eswatini, Lesotho, Malawi, Mozambique, Namibia,
South_Africa, Tanzania, Zambia, Zimbabwe) -- Rwanda itself (geographically East African,
part of EAPP, not SAPP) doesn't appear anywhere in the file, not even under an abbreviation
(checked generator names and zone labels for "RWA"). It's not a byte-for-byte duplicate of
Malawi's `input.gdx` (different size/hash, different generator set) -- just a different,
smaller SAPP-adjacent dataset that ended up under the "Rwanda" region name. This needs the
actual correct Rwanda `.gdx` supplied before this region can be processed; nothing in
`gdx`/`seer.inputs` can recover real Rwanda data from what's here.

## Serbia

- **pGenDataInputCustom.csv**: added `Capacity`/`UnitSize`/`BuildLimitperYear` rows for
  `PHP__Serbia` -- present in `pStorageDataInput.csv` (CapacityMWh=5440) but had no
  corresponding entry at all in `pGenDataInputCustom.csv`, so no zone/tech. Not the same unit
  as the similarly-named `PHP_New_Ser` (a separate, complete candidate with CapacityMWh=8000)
  -- confirmed distinct, not a naming variant of an existing entry. zone/tech/fuel copied from
  `PHP_New_Ser`'s own convention (`Serbia`/`PumpedHy`/`Battery`). `Capacity = CapacityMWh / 8`
  -- pumped-hydro storage in this dataset consistently uses an 8h duration (every other real
  `PHP_*`/`PSH_*` entry across regions is 8-21h, distinctly longer than battery's 4h), so 8h
  was used here rather than the battery-style 4h default used for Malawi's `_new` candidates
  / Eswatini's `Battery_Generic` (both genuinely battery storage). `UnitSize = 5` MW,
  `BuildLimitperYear = 100,000` MW/year (deliberately large, never binds).

- **`Bistrica`/`Derdap 3` investigated, no change made**: both are `tech=PumpedHy` generators
  (`Status=3`, full param set) but have no entry at all in `pStorageDataInput.csv` -- the
  opposite gap from `PHP__Serbia` (no `CapacityMWh` to anchor a duration/energy-capacity
  figure from, unlike the fixes above). Rather than fabricate storage parameters with no
  anchor, treating these as ordinary one-way generators (which is how they're already
  modeled) is a reasonable simplification -- these plants may be operated more like
  baseload/dispatchable generation than active storage arbitrage in practice. Left as
  generators, not converted to storage.

## Tanzania

- **zcmap.csv**: replaced with a single zone. Raw file had `Tanzania,TanTrade` /
  `TanTrade,Tanzania` (a bidirectional pair, same pattern as Eswatini/Guinea) -- `TanTrade`
  has zero generators/demand anywhere in the data, a pure trade-hub label like Eswatini's
  `Esw_Trade`/Guinea's `Export_Guinea`. Final: `Tanzania,Tanzania`. (This was identified early
  in the session but not actually applied until now -- caught during Tanzania's full
  schema-check pass, since the un-deduped file was also producing a swapped
  zone/country pair in `zones.csv`.)

## Timor Leste

- **pStorageDataInput.csv**: corrected `Battery_Generic`'s `CapacityMWh` from `1,000,000` to
  `20,000` -- combined with its `Capacity=5000` MW (from `pGenDataInputCustom.csv`), the raw
  value implied a 200-hour storage duration, wildly implausible for battery storage (compare
  the same file's `Grenfield_Battery`, a normal small battery at `CapacityMWh=40`). `Capacity
  =5000` also exactly matches this unit's own `MaxTotalBuild=5000`, the same "round ceiling
  figure standing in for a real value" pattern seen elsewhere (e.g. Guinea's `PV` candidate,
  Eswatini's own `Battery_Generic`) -- `CapacityMWh` looks like it was set to another round
  ceiling to match, rather than reflecting any real energy capacity. Since this unit is
  expandable, `capacity_mw` itself is never used as an actual ceiling (only
  `annual_build_limit_mwh` matters), but `duration_hours` (derived from `capacity_mwh /
  capacity_mw`) *is* a real physical property applied to whatever capacity actually gets
  built -- so it needed a sane value. Used the same 4h duration assumption applied to
  Eswatini's `Battery_Generic`: `CapacityMWh = Capacity * 4 = 20,000`.

## Turkiye

- **zcmap.csv**: replaced with a clean one-row-per-zone table. The raw file was a
  bidirectional trade-partner network among Turkiye's own 9 zones plus 3 vestigial
  interconnection labels (`Inter_Trakia`, `Inter_EastAna`, `Inter_SouthEast`) with zero
  generators/demand anywhere in the data -- same pattern as Guinea's `Export_Guinea`/
  Eswatini's `Esw_Trade`. Confirmed against `pZoneIndex.csv` that the real zones are exactly
  the 9 (`WestMed`, `WestAna`, `EastMed`, `EastAna`, `SouthEast`, `NorthWest`, `CenterAna`,
  `CenterBlack`, `Trakia`), all with generators and demand. Single-country model, so country
  is set to "Turkiye" for all 9.

- **pGenDataInputCustom.csv**: removed 167 generators entirely -- all `Status=2`, all
  far-future year-slots in "committed" build-candidate name-sequences (e.g.
  `Onshore_Com2030`-`Com2035`, `PV_Com2030`-`Com2035`, `BESS_Com2030`-`Com2035`, across every
  zone/tech). Unlike Guinea's `Status=1`/`Status=3` split, Turkiye's `Status=2` bucket is
  inconsistent: 349 total, but only 83 have a `Capacity` or `BuildLimitperYear` value at all --
  these 167 (the 266 empty ones minus 99 that are genuine storage units handled separately)
  have neither, nor `MaxTotalBuild`/`UnitSize`/`HeatRate` -- only cost/lifetime/start-year
  metadata (Capex, Life, FOM, StYr/RetrYr), nothing that specifies a quantity to build or an
  existing size. The earlier year-slots in the same name-sequences (e.g. `Onshore_Com2025`-
  `Com2029`) DO have real `Capacity`/`BuildLimitperYear` values and are kept -- this looks like
  the raw source's committed-capacity-addition schedule simply runs out a few years before the
  model horizon ends, not a fixable data-entry gap (no comparable anchor value exists to derive
  a number from, unlike the storage-duration fixes elsewhere in this file). Left as fully
  dropped rather than present-but-undefined.

- **pGenDataInputCustom.csv**/**pStorageDataInput.csv**: also removed 3 further generators
  (`Hydro_Sto_WestAna_Cand`, `ROR_NorthWest_Cand`, `ROR_WestAna_Cand` -- `Status=3` hydro
  candidates, same "no `Capacity`/`BuildLimitperYear`/`MaxTotalBuild`/`UnitSize` anywhere"
  gap as above) and 99 `*_BESS_Com<year>-1h` storage units (one per zone x year 2025-2035).
  Unlike the onshore/PV `Com` case, these BESS storage entries have **zero** `CapacityMWh`
  data across every single year, not just later ones -- confirmed by checking every
  `BESS_Com*` plant in `pStorageDataInput.csv` (all 99, 0 have `CapacityMWh`). The sibling
  `BESS_Cand<year>-<duration>h` family (candidate/expandable, different name suffix) is fully
  populated (`Capacity=100,000` ceiling, matching `CapacityMWh`, real `BuildLimitperYear`) and
  was kept as-is. No usable data exists for the `Com` BESS family at all, so dropped entirely.

## WAPP

- **zcmap.csv**: replaced with a clean one-row-per-zone table. Unlike Guinea/Turkiye, this raw
  file had no vestigial trade-hub entries -- confirmed against `pZoneIndex.csv` that all 16
  bidirectional-network zones are real (each has generators and demand in the region's own
  data): `Benin`, `Burkina`, `Gambia`, `Ghana`, `Guinea`, `GuineaBissau`, `IvoryCoast`,
  `Liberia`, `Mali`, `Niger`, `Nigeria`, `Senegal`, `SierraLeone`, `Togo`, `Mauritania_grid`,
  `Mauritania_Nema`. WAPP is a genuine multi-country interconnection pool (unlike Guinea's/
  Turkiye's single-country models), so country is set to each zone's own country -- except
  `Mauritania_grid`/`Mauritania_Nema`, both mapped to `Mauritania` (same country, two zones,
  same pattern as Malawi's `DRC`/`DRC_South`).

- **pGenDataInputCustom.csv**/**pStorageDataInput.csv**: removed 9 units entirely -- 8
  generators (`GT_Benin`, `CC_GAS_LNG_BEN`, `ONSHORE_WIND_BEN`, `WIND_CAND_GBI`,
  `OFSHORE_WIND_GBI`, `SONICHAR2`, `Hydro_Kaleta_viaGuinea`, `Hydro_Souapiti_viaGuinea`) and
  1 storage candidate (`Storage_CAND_2026_Senegal`), all `Status=2` or `Status=3` -- same
  "no `Capacity`/`BuildLimitperYear`/`MaxTotalBuild`+`BuildLimitperYear` anywhere" gap as
  Turkiye's dropped units: only cost/lifetime/heat-rate metadata, nothing that specifies a
  quantity to build or an existing size (`OFSHORE_WIND_GBI` has a lone `MaxTotalBuild=5000`
  but no `BuildLimitperYear`, so it's still undefined as a build option). No comparable
  sibling entry to derive a number from, so dropped rather than left present-but-undefined.
  Small relative to the region's size -- 8 of 460 generators (1.7%), 1 of 28 storage units
  (3.6%).

- **`Projet BESS 2030` investigated, no change made**: a battery (`tech=battery storage`)
  modeled as a plain generator (`Capacity=35`, expandable) rather than storage -- has no
  entry at all in `pStorageDataInput.csv`, the same gap as Serbia's `Bistrica`/`Derdap 3`
  (real storage-like plants with no `CapacityMWh` anchor to convert them into a proper
  storage unit). Left as an ordinary generator, consistent with that precedent.
