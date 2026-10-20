"""settings.csv lookups shared by both model backends."""

# value of lost load ($/MWh) when settings.csv has no `voll` -- kept high so lost load is only
# used when truly necessary
DEFAULT_VOLL = 1_000_000.0


def get_voll(settings_df):
    """settings.csv's optional `voll` ($/MWh), else DEFAULT_VOLL."""
    values = settings_df.set_index(settings_df["parameters"].astype(str).str.lower())["value"]
    return float(values["voll"]) if "voll" in values.index else DEFAULT_VOLL
