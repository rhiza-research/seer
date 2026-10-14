"""Base interface for seer-schema perturbations: pure SeerInstance -> SeerInstance transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Sequence

import numpy as np

from seer.schema import SeerInstance


class Perturbation(ABC):
    """A named, parameterized SeerInstance -> SeerInstance transform. Subclasses are dataclasses
    whose fields are the perturbation's own parameters, and must not mutate the instance they're
    given (start from `instance.copy()`)."""

    @abstractmethod
    def apply(self, instance: SeerInstance, rng: np.random.Generator) -> SeerInstance: ...

    @property
    def name(self) -> str:
        return type(self).__name__

    def params(self) -> dict:
        return asdict(self)


def apply_all(
    instance: SeerInstance, perturbations: Sequence[Perturbation], seed: int | None = None
) -> tuple[SeerInstance, list[dict]]:
    """Apply perturbations in order under one seeded rng. Returns the final instance and a
    {name, params} log per perturbation, for ML-dataset labeling."""
    rng = np.random.default_rng(seed)
    log = []
    for p in perturbations:
        instance = p.apply(instance, rng)
        log.append({"name": p.name, "params": p.params()})
    return instance, log
