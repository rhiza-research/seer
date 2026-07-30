"""Experiment: build and solve a PyPSA model from the seer-schema Gambia data, restricted to
just a few years (rather than the full horizon) for a quick test solve."""

import shutil
from pathlib import Path

import pandas as pd

from seer.models.pypsa_model import build, solve

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATA_PATH = REPO_ROOT / "data" / "seer" / "Gambia"
DATA_PATH = REPO_ROOT / "data" / "seer" / "Gambia_short"

N_YEARS = 3  # solve just the first few years, not the full horizon

# copy the full Gambia dataset, then trim years.csv down to the first N_YEARS years
# (years.csv is the only file that needs trimming: it's what drives model.investment_periods,
# everything else either isn't year-indexed or tolerates extra unused years/columns)
shutil.copytree(SOURCE_DATA_PATH, DATA_PATH, dirs_exist_ok=True)
years_df = pd.read_csv(SOURCE_DATA_PATH / "years.csv").head(N_YEARS)
years_df.to_csv(DATA_PATH / "years.csv", index=False)

model = build(f"{DATA_PATH}/")
solve(model)
