"""A0 single-run harness (spec v0.2 rungs; used by E1/E2/E3 drivers).

One run = one (schedule, seed). Each round: apply drift, sample + label the
operational stream, update the running RM direction, compute operational
diagnostics, probe the frozen anchors, and log one JSONL row. No estimator yet
(C7/M5) -- the m_k/cov/loglik fields stay null through M4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import drift
from ..anchors import build_battery_a0
from ..config import resolve_config
from ..diagnostics import fit_logistic_direction, op_agree, op_temp_inv
from ..logging_io import RunLogger, write_resolved_config
from ..probe import d_dir, d_norm, probe_battery
from ..seeding import spawn_generators
from ..streams import sample_operational_round
from ..teacher import Teacher


def run_a0(schedule_name: str, seed: int, K: int = 60, run_dir: str | Path | None = None,
           d: int = 24, hist_dim: int = 16, n_probe: int = 50) -> dict:
    """Run one A0 (schedule, seed) for K rounds; return per-round arrays.

    If ``run_dir`` is given, also writes ``log.jsonl`` + resolved config there.
    """
    ctx = drift.DriftContext.a0(d=d, hist_dim=hist_dim)
    rngs = spawn_generators(seed)
    sched = drift.get_preset(schedule_name)
    if sched.rung != "A0":
        raise ValueError(f"{schedule_name} is not an A0 preset")

    # Teacher: unit v0 in S_hist, beta = 1.
    w0 = np.zeros(d)
    w0[ctx.hist_idx] = rngs["teacher"].standard_normal(hist_dim)
    w0 /= np.linalg.norm(w0)
    teacher = Teacher(w=w0.copy(), s=0.0, beta=1.0)

    # Frozen battery + baseline probe (n_probe draws/anchor).
    battery = build_battery_a0(teacher.v(), ctx, rngs["probe"])
    p_hat_0 = probe_battery(battery, rngs["probe"], prob_fn=teacher.prob, n_probe=n_probe)

    logger = None
    if run_dir is not None:
        logger = RunLogger(run_dir)
        write_resolved_config(run_dir, resolve_config(
            {"exp": "a0", "schedule": schedule_name, "K": K, "d": d}, seed))

    out = {k: [] for k in ("round", "D_dir", "D_norm", "op_agree", "op_flip",
                           "op_temp_inv_std", "op_temp_inv_raw", "c_k", "norm_v", "angle_v")}
    acc_gaps: list[np.ndarray] = []
    acc_labels: list[int] = []
    what = np.zeros(d)
    v0 = teacher.v().copy()

    for k in range(1, K + 1):
        # 1) drift (S0 = no-op)
        v_new, flags = sched.apply_round(k, teacher.v(), ctx, rngs["drift"])
        teacher.w = v_new / teacher.beta

        # 2) operational stream + labels
        opr = sample_operational_round(k, ctx, teacher.label, rngs["stream"], rngs["labels"])
        acc_gaps.append(opr.gaps)
        acc_labels.extend(opr.labels.tolist())

        # 3) running RM direction (warm-started)
        X = np.vstack(acc_gaps)
        y = np.array(acc_labels, dtype=float)
        what = fit_logistic_direction(X, y, w_init=what)

        # 4) operational diagnostics (this round's fresh pairs)
        agree = op_agree(opr.labels[opr.dup_index], opr.dup_labels)
        ti_std, ti_raw = op_temp_inv(opr.gaps, opr.labels, what)

        # 5) probe anchors -> panel readouts
        p_hat_t = probe_battery(battery, rngs["probe"], prob_fn=teacher.prob, n_probe=n_probe)
        Dd = d_dir(p_hat_t, p_hat_0, battery.dir_idx)
        Dn = d_norm(p_hat_t, p_hat_0, battery.z0, battery.norm_idx)

        v = teacher.v()
        ang = float(np.degrees(np.arccos(np.clip(
            (v @ v0) / (np.linalg.norm(v) * np.linalg.norm(v0)), -1.0, 1.0))))
        from ..streams import gap_scale
        c_k = gap_scale(k)

        for key, val in (("round", k), ("D_dir", Dd), ("D_norm", Dn), ("op_agree", agree),
                         ("op_flip", 1.0 - agree), ("op_temp_inv_std", ti_std),
                         ("op_temp_inv_raw", ti_raw), ("c_k", c_k),
                         ("norm_v", float(np.linalg.norm(v))), ("angle_v", ang)):
            out[key].append(val)

        if logger is not None:
            logger.log_row(round=k, seed=seed, schedule=schedule_name, v_true=v,
                           D_dir=Dd, D_norm=Dn, op_agree=agree, op_flip=1.0 - agree,
                           op_temp_inv_std=ti_std, op_temp_inv_raw=ti_raw,
                           n_labels_cum=len(acc_labels), event_flags=flags)

    if logger is not None:
        logger.close()
    return {k: np.array(v) for k, v in out.items()}
