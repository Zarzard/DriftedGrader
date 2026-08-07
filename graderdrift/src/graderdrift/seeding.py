"""Reproducible component-wise RNG spawning.

Spec Section 1: "One master seed spawns named ``numpy.random.Generator``
children per component; document the spawning so runs are reproducible
component-wise."

We use ``numpy.random.SeedSequence(master).spawn(n)`` and bind the children to a
*fixed, ordered* list of component names. The ordering below is the contract:
appending a new component name at the END preserves every existing component's
stream; inserting or reordering does NOT and must be avoided. Each component
therefore has an independent, master-seed-derived, order-stable Generator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed, append-only ordering. Do not reorder or insert -- append only.
COMPONENTS: tuple[str, ...] = (
    "teacher",
    "drift",
    "stream",
    "labels",
    "probe",
    "policy",
    "rm",
)


@dataclass(frozen=True)
class RNGBundle:
    """Named per-component Generators plus the master seed that produced them."""

    master_seed: int
    generators: dict[str, np.random.Generator]

    def __getitem__(self, name: str) -> np.random.Generator:
        return self.generators[name]

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name in self.generators


def spawn_generators(master_seed: int, components: tuple[str, ...] = COMPONENTS) -> RNGBundle:
    """Spawn one independent Generator per component name from ``master_seed``.

    Reproducible: the same ``master_seed`` and ``components`` ordering always
    yields the same per-component streams.
    """
    ss = np.random.SeedSequence(int(master_seed))
    children = ss.spawn(len(components))
    gens = {name: np.random.default_rng(child) for name, child in zip(components, children)}
    return RNGBundle(master_seed=int(master_seed), generators=gens)
