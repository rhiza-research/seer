"""Perturb an existing seer-schema instance: set parameters to fixed scenario levels, perturb chosen ones,
and / or add noise to all of them.
Examples:
    perturb(instance, scenario={"fuel:gas": "high", "demand_growth": "low"}, seed=0)
    perturb(instance, perturbation={"capex_pathway": 0.3}, seed=0)
    perturb(instance, scenario={"hydro": "low"}, perturbation=0.2, noise=0.02, seed=0)

See scenario_parameters for parameters, groups and items (parameters in neither are left as they are), and
noise.py for how noise jitters each one. Values in schema tables are rounded to significant_figures, so
written instances stay compact (and compress well).
"""
import numpy as np

from . import costs, demand_trajectory, profiles, settings
from .noise import add_noise
from .scenario_parameters import DEFAULT_COST_DATASET, parse

NOISE_STREAM = 1  # noise draws from their own random stream, so they don't shift the scenario / perturbation draws

SCHEMA_TABLES = ["settings", "demand_forecast", "demand_profile", "renewable_profiles", "generator_costs", "storage_costs"]


def perturb(seer_instance, scenario=None, perturbation=None, seed=None, significant_figures=6,
            cost_dataset=DEFAULT_COST_DATASET, noise=0):
    """Copy of seer_instance with scenario's levels set, then perturbation's scales applied, then relative
    noise added to every parameter, and SCHEMA_TABLES' float values rounded to significant_figures (None: no
    rounding). Cost levels come from cost_dataset."""
    scenario_options = parse(scenario, perturbation, seed=seed, cost_dataset=cost_dataset)
    out = seer_instance.copy()

    # settings
    out = settings.resample(out, scenario_options)

    # demand
    out = demand_trajectory.resample(out, scenario_options)
    out = profiles.resample_demand_profile(out, scenario_options)

    # renewables
    out = profiles.resample_re_profile(out, scenario_options)
    out = profiles.resample_hydro(out, scenario_options)

    # costs
    out = costs.resample_capex(out, scenario_options)
    out = costs.resample_marginal(out, scenario_options)
    out = costs.resample_line_capex(out, scenario_options)

    # noise, on everything
    out = add_noise(out, noise, np.random.default_rng(None if seed is None else [seed, NOISE_STREAM]))

    # rounding
    if significant_figures is not None:
        for name in SCHEMA_TABLES:
            df = getattr(out, name)
            for column in df.select_dtypes("float").columns:
                df[column] = round_significant(df[column].to_numpy(), significant_figures)
    return out


def round_significant(values, significant_figures):
    """values rounded to significant_figures (via decimal formatting, so they print short)."""
    return np.array([float(f"{v:.{significant_figures}g}") for v in values])
