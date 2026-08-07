"""M0 end-to-end smoke driver.

Exercises the full plumbing path -- config -> resolve -> per-component RNGs ->
teacher -> per-round JSONL rows + resolved-config copy -- with no science in it.
Its only job is the M0 definition of done: "one dummy run writes valid JSONL".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import resolve_config
from ..logging_io import RunLogger, write_resolved_config
from ..seeding import spawn_generators
from ..teacher import Teacher


def run_dummy(run_dir: str | Path, seed: int = 0, n_rounds: int = 3, d: int = 4) -> Path:
    """Write a tiny valid run to ``run_dir`` and return the log path."""
    cfg = {"exp": "dummy", "schedule": {"name": "S0"}, "d": d, "n_rounds": n_rounds}
    resolved = resolve_config(cfg, seed)
    write_resolved_config(run_dir, resolved)

    rng = spawn_generators(seed)
    w0 = rng["teacher"].normal(size=d)
    w0 /= np.linalg.norm(w0)
    teacher = Teacher(w=w0, s=0.0, beta=1.0)

    with RunLogger(run_dir) as logger:
        for k in range(n_rounds):
            dphi = rng["stream"].normal(size=d)
            y = teacher.label(dphi, rng["labels"])
            logger.log_row(
                round=k,
                seed=seed,
                schedule="S0",
                v_true=teacher.v(),
                op_agree=float(y),  # placeholder scalar so the row is non-trivially populated
                n_labels_cum=(k + 1),
                event_flags=[],
            )
    return Path(run_dir) / "log.jsonl"
