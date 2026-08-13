"""Generate multiple synthetic seer-schema instances per case, from a YAML config.

Each case fully specifies a generate_all_data() call (zones, years, mix, noise, etc.) plus how
many instances to generate; every instance is an independent, freshly-noised draw from the same
case parameters (see seer/inputs/synthetic/noise.py).
"""

from pathlib import Path

import yaml

from seer.inputs.synthetic.generate_data import generate_all_data

# case keys passed straight through as generate_all_data kwargs (n_zones/years/num_instances
# are handled separately below)
CASE_KWARGS = [
    "country_name", "zone_peak_demand_mw", "annual_discount_rate", "starting_generator_mix",
    "starting_reserve_margin", "storage_capacity_pct", "demand_growth_rate_mean",
    "noise_pct", "mesh_level", "congestion_level",
]


def generate_cases(config_path, output_root):
    config = yaml.safe_load(Path(config_path).read_text())
    output_root = Path(output_root)

    for case_name, case in config["cases"].items():
        n_zones = case["n_zones"]
        years = case["years"]
        num_instances = case.get("num_instances", 1)
        kwargs = {key: case[key] for key in CASE_KWARGS if key in case}

        for i in range(num_instances):
            instance_path = output_root / case_name / f"instance_{i:03d}"
            generate_all_data(n_zones, years, instance_path, **kwargs)
            print(f"Wrote {case_name} instance {i} to {instance_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    generate_cases(
        config_path=repo_root / "jobs" / "synthetic_cases.yaml",
        output_root=repo_root / "data" / "seer" / "synthetic",
    )
