"""YAML config loading and resolution.

Spec Section 1: "Every run is fully specified by one YAML config + one integer
seed." A resolved config (config + seed + any preset merge) is what gets copied
next to the run's logs so a run is reproducible from its output directory alone.

M0 keeps this deliberately thin: load YAML, attach the seed, deep-copy for the
resolved artifact. Schedule-preset merging (the ``schedule:`` block, C2) lands in
M1 -- ``resolve_config`` is the seam it will plug into.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} must be a mapping, got {type(cfg).__name__}")
    return cfg


def resolve_config(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Produce the fully-resolved, self-describing config for one run.

    Deep-copies the input, stamps the integer seed, and (in later milestones)
    will expand schedule presets in place. The result is what ``logging_io``
    serializes alongside ``log.jsonl``.
    """
    resolved = copy.deepcopy(cfg)
    resolved["seed"] = int(seed)
    return resolved
