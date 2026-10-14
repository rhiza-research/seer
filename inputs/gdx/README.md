## Package Description
The `gdx` package takes in GAMS EPM .gdx input files, extracts each region's relevant symbols and writes the data for each symbol out to a CSV.

## Definitions & Concepts
- Symbol: GAMS input file parameters (e.g. pGenDataInput, pVREProfile)
- Region: Country/Power Pool each with it's own input.gdx file
- Canonical Schema: Standardized symbol names and associated column headings for each symbol's parameter data. The schema is defined by `symbol_mapping.csv` and enforced by `sync_symbol.py`
- Raw vs Aligned Data: GAMS input files are processed as both raw and canonically aligned formats with data enrichment using raw forms to aid in lookup, while data standardization uses the aligned form

## Pipeline
```
input.gdx                                     (per region)
   │
   ▼
`load_gdx.py` + `sync_symbol`                 (read raw symbols & align names with canonical schema)
   │
   ▼
`enrich_data.py` + `standardize_profiles.py`  (standardize fuel, technology, and hourly profiles)
   │
   ▼
`to_csv.py`                                   (write out symbols to output csv files)
```

## Modules
### `load_gdx.py`
Reads input.gdx for each region and converts gdx files into pandas dataframes that can be accessed for each region and symbol through dictionary-based calls (e.g. all_data['Gambia']['pVREgenProfile'])

### `sync_symbol.py`
Builds the symbol registry to map raw symbols for each gdx file to the canonically defined symbol names and associated data column headers using `symbol_mapping.csv`

### `enrich_data.py`
Reconciles fuel and technology types for all generators based on defined `FUEL_NAME_MAP`

### `standardize_profiles.py`
Standardizes all hourly profile data for renewable generation and demand for conversion to seer schema

### `to_csv.py`
Writes one CSV per symbol per region to the specified output directory

## Notes
1. If new gdx files are processed which contain different raw symbol names, add the symbol and its canonical column layout to `symbol_mapping.csv`.
2. If a new symbol needs further modifications, add code to `enrich_data.py` to capture additional enrichment details.
3. If additional profiles are added, verify that `standardize_profiles.py` can handle their formatting for processing into seer schema.
4. If additional data standardization is needed before being piped to seer schema conversion, add scripts to the `gdx` package.