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

_(no manual changes yet)_

## Gambia

_(no manual changes yet)_

## Guinea

_(no manual changes yet)_

## Malawi

_(no manual changes yet)_

## Mexico

_(no manual changes yet)_

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
