"""
Seer schema instance.
Object wrapper for a seer schema instance that can be validated and modified."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

required_fields = [
    "settings",
    # time and space
    "zones",
    "years",
    "times",
    # demand
    "demand_forecast",
    "demand_profile",
    # supply
    "renewable_profiles",
    "generator_constants",
    "generator_costs",
    "storage_constants",
    "storage_costs",
]

optional_fields = ["network"]  # single-zone models can omit network.csv entirely
empty_network = pd.DataFrame(columns=["from", "to", "capacity"])

all_fields = required_fields + optional_fields


@dataclass
class SeerInstance:
    """seer schema instance."""

    settings: pd.DataFrame
    # time and space
    zones: pd.DataFrame
    years: pd.DataFrame
    times: pd.DataFrame
    # demand
    demand_forecast: pd.DataFrame
    demand_profile: pd.DataFrame
    # supply
    renewable_profiles: pd.DataFrame
    generator_constants: pd.DataFrame
    generator_costs: pd.DataFrame
    storage_constants: pd.DataFrame
    storage_costs: pd.DataFrame
    network: pd.DataFrame = field(default_factory=lambda: empty_network.copy())

    @classmethod
    def read(cls, data_path, validate: bool = True) -> "SeerInstance":
        """Instantiate seer instance from `data_path`
        (can be a local dir or remote URL, e.g. gs://bucket/prefix).
        If `validate` is True, the instance will be validated against the schema rules.
        """
        data_path = str(data_path).rstrip("/")
        values = {name: pd.read_csv(f"{data_path}/{name}.csv") for name in required_fields}
        # load network.csv if it exists, otherwise use empty network
        try:
            values["network"] = pd.read_csv(f"{data_path}/network.csv")
        except FileNotFoundError:
            values["network"] = empty_network.copy()
        # instantiate and validate if requested
        instance = cls(**values)
        if validate:
            instance.validate()
        return instance

    def write(self, output_path) -> None:
        """Write seer instance to `output_path`
        (can be a local dir or remote URL, e.g. gs://bucket/prefix).
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in all_fields:
            df = getattr(self, name)
            if name == "network" and df.empty:
                continue
            df.to_csv(output_dir / f"{name}.csv", index=False)

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in all_fields}

    def copy(self) -> "SeerInstance":
        return replace(self, **{name: getattr(self, name).copy() for name in all_fields})

    def validate(self) -> None:
        from .validation import validate_seer_instance
        validate_seer_instance(self)
