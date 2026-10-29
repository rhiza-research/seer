# Reference data

The values `seer.scenarios` samples scenarios and perturbations from. Every row of the cost tables has a
`source` column citing exactly where its numbers come from; this lists the datasets behind them. Monetary
values are in 2025 US dollars: source dollar years are inflated with the US BLS CPI-U, euros converted at
ECB annual average rates.

## Costs

| Table | Dataset label | Source |
|---|---|---|
| `technology/tech_cost_starts.csv`, `tech_cost_pathways.csv` | `atb` | [NREL Annual Technology Baseline](https://atb.nrel.gov/) 2024 v4.0.0 (Moderate = mid, Advanced / Conservative = low / high); [PyPSA technology-data](https://github.com/PyPSA/technology-data) costs_2030 for techs ATB doesn't cover (CHP, lignite, oil, battery inverter) |
| `technology/tech_cost_starts.csv` | `wb_epm` | starting capex from the World Bank [Electricity Planning Model](https://energydata.info/group/epm) (EPM) country models: 10th percentile / median / 90th percentile across models |
| `fuels/fuel_cost_starts.csv` | `pink_sheet` | [World Bank Commodity Price Data (The Pink Sheet)](https://www.worldbank.org/en/research/commodity-markets), annual prices; [EIA Annual Energy Outlook](https://www.eia.gov/outlooks/aeo/) 2025 where a row's `source` says so; PyPSA technology-data for lignite, uranium and biomass |
| `fuels/fuel_cost_pathways.csv` | `pink_sheet` | EIA Annual Energy Outlook 2025: Reference, High / Low Oil and Gas Supply cases (as ratios to 2025) |
| `fuels/fuel_cost_starts.csv` | `wb_epm` | fuel prices (`pFuelPrice`) from the EPM country models, as for capex |
| `network/line_cost_starts.csv`, `line_cost_pathways.csv` | | [Danish Energy Agency, Technology Data for Energy Transport](https://ens.dk/en/analyses-and-statistics/technology-data-energy-transport) (central estimate; lower / upper bounds) |

## Technology parameters

`technology/tech_parameters.csv`: lifetimes and storage efficiency / duration from PyPSA technology-data; heat
rates from NREL ATB 2025 (Moderate) where it has them, else from PyPSA efficiency; hydro and run-of-river
capacity-factor ranges from [Ember Yearly Electricity Data](https://ember-energy.org/data/yearly-electricity-data/)
(countries with at least 1 GW of hydro, 2015-2024).

## Profiles

| Library | Source |
|---|---|
| `profiles/demand/daily_profile_*.csv` | daily load shapes from the EPM country models' demand profiles (each day normalized to its own peak) |
| `profiles/renewable/daily_profile_*.csv` | daily solar and wind capacity-factor curves from the finalized seer instances in [seer-planning-models](https://github.com/rhiza-research/seer-planning-models) (`from_epm/<country>`, themselves derived from EPM) |

## Scenario parameters

`scenario_data.yaml` holds the low / mid / high values of the non-cost scenario parameters (discount rate,
value of lost load, demand growth, profile shapes, hydro), each cited by a tag defined at its top: the EPM
input catalog and country models, IRENA Renewable Power Generation Costs in 2024, the IEA Cost of Capital
Observatory and Electricity 2025, Federal Reserve FEDS Notes on discount rates, published value-of-lost-load
studies (Oseni & Pollitt 2015; Schröder & Kuckshinrichs 2015), Ember, the World Bank / ESMAP / Solargis
Global Photovoltaic Power Potential by Country, and the profile libraries above.
