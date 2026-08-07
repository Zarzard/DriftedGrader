"""E2 -- split-rule validation (spec v0.2 Section 4, E2).

A0, {S1p pure 25deg rotation, S2j pure x0.6 norm jump} x 20 seeds, event at t=30,
K=60. Probe-only (no operational stream needed): each round probes the frozen
battery; at round 30 two extra independent probes of the frozen post-event state
give the pure-sampling-noise null for cell (i).

Pre-registered predictions (amendment A2 applied):
(i)   D_dir under S2j = 0 in expectation -- equivalence vs the relabeling null;
(ii)  D_dir under S1p large;
(iii) D_norm under S2j matches per-anchor Delta z_j = (0.6-1) * zhat_j(0), with
      zhat_j(0) the round-0 probe MLE; per-anchor CIs cover for >= 7/8 anchors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import drift
from .._math import logit
from ..anchors import build_battery_a0
from ..probe import N_PROBE_DEFAULT, d_dir, d_norm, probe_battery
from ..seeding import spawn_generators
from ..teacher import Teacher

K = 60
EVENT_T = 30
_CLIP = (0.02, 0.98)


def run_e2(schedule_name: str, seed: int) -> dict:
    ctx = drift.DriftContext.a0()
    rngs = spawn_generators(seed)
    sched = drift.get_preset(schedule_name)
    w0 = np.zeros(ctx.d)
    w0[ctx.hist_idx] = rngs["teacher"].standard_normal(len(ctx.hist_idx))
    w0 /= np.linalg.norm(w0)
    teacher = Teacher(w=w0.copy(), s=0.0, beta=1.0)

    battery = build_battery_a0(teacher.v(), ctx, rngs["probe"])
    p0 = probe_battery(battery, rngs["probe"], prob_fn=teacher.prob)

    D_dir_t, D_norm_t = [], []
    logit_shift = None      # per-norm-anchor mean post-event logit shift accumulators
    shifts = []
    null_ddir = None
    for k in range(1, K + 1):
        v_new, _ = sched.apply_round(k, teacher.v(), ctx, rngs["drift"])
        teacher.w = v_new / teacher.beta
        p_t = probe_battery(battery, rngs["probe"], prob_fn=teacher.prob)
        D_dir_t.append(d_dir(p_t, p0, battery.dir_idx))
        D_norm_t.append(d_norm(p_t, p0, battery.z0, battery.norm_idx))
        if k > EVENT_T:
            lo, hi = _CLIP
            shifts.append(logit(np.clip(p_t[battery.norm_idx], lo, hi))
                          - logit(np.clip(p0[battery.norm_idx], lo, hi)))
        if k == EVENT_T:
            # relabeling null: two fresh probes of the SAME frozen post-event state
            pa = probe_battery(battery, rngs["probe"], prob_fn=teacher.prob)
            pb = probe_battery(battery, rngs["probe"], prob_fn=teacher.prob)
            null_ddir = float(np.mean(np.abs(pa[battery.dir_idx] - pb[battery.dir_idx])))

    lo, hi = _CLIP
    zhat0 = logit(np.clip(p0[battery.norm_idx], lo, hi))  # round-0 probe MLE of z_j(0)
    shifts = np.array(shifts)                              # (30, m_norm)
    return {
        "schedule": schedule_name, "seed": seed,
        "D_dir_post": float(np.mean(D_dir_t[EVENT_T:])),
        "D_norm_post": float(np.mean(D_norm_t[EVENT_T:])),
        "null_ddir": null_ddir,
        "zhat0_norm": zhat0.tolist(),
        "norm_shift_mean": shifts.mean(axis=0).tolist(),   # per-anchor observed dz
        "norm_shift_se": (shifts.std(axis=0, ddof=1) / np.sqrt(len(shifts))).tolist(),
    }


def analyze(rows: list[dict]) -> None:
    byc = lambda s: [r for r in rows if r["schedule"] == s]
    s2j, s1p = byc("S2j"), byc("S1p")
    n = len(s2j)

    # (i) equivalence: paired per-seed (observed D_dir - relabeling null)
    diff = np.array([r["D_dir_post"] - r["null_ddir"] for r in s2j])
    m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
    ci = (m - 1.645 * se, m + 1.645 * se)
    bound = 0.01
    ok1 = ci[0] > -bound and ci[1] < bound
    print(f"(i)  S2j D_dir vs sampling null: paired diff {m:+.4f} 90%CI [{ci[0]:+.4f},{ci[1]:+.4f}]"
          f" within +/-{bound} -> {'PASS' if ok1 else 'FAIL'}")

    # (ii) S1p D_dir large (CI above its null)
    d1 = np.array([r["D_dir_post"] for r in s1p]); n1 = np.array([r["null_ddir"] for r in s1p])
    print(f"(ii) S1p D_dir post-event {d1.mean():.3f} vs null {n1.mean():.3f} "
          f"-> {'PASS' if d1.mean() > n1.mean() + 5 * d1.std(ddof=1) / np.sqrt(len(d1)) else 'FAIL'}")

    # (iii) per-anchor prediction dz_j = (f-1) zhat_j(0) on S2j, f = 0.6.
    # CI must include the SHARED baseline error: obs - pred = logit(p_t_bar) -
    # f*logit(p_hat0), so the round-0 probe's logit noise (delta method:
    # 1/(n p0(1-p0))) enters with coefficient f and is invisible in the
    # round-to-round scatter -- omitting it makes the CI ~3x too narrow.
    f = 0.6
    cover, total = 0, 0
    for r in s2j:
        zhat0 = np.array(r["zhat0_norm"])
        pred = (f - 1.0) * zhat0
        obs = np.array(r["norm_shift_mean"]); se_rounds = np.array(r["norm_shift_se"])
        p0 = 1.0 / (1.0 + np.exp(-zhat0))
        se_base = f * np.sqrt(1.0 / (N_PROBE_DEFAULT * p0 * (1.0 - p0)))
        se_j = np.sqrt(se_rounds ** 2 + se_base ** 2)
        cover += int(np.sum(np.abs(obs - pred) <= 1.645 * se_j)); total += len(pred)
    frac = cover / total
    print(f"(iii) per-anchor dz coverage of (f-1)*zhat0: {cover}/{total} = {frac:.2f} "
          f"(gate >= 7/8 = 0.875) -> {'PASS' if frac >= 0.875 else 'FAIL'}")
    # cell means for the record
    print(f"cells: D_dir S2j={np.mean([r['D_dir_post'] for r in s2j]):.4f} "
          f"S1p={d1.mean():.4f} | D_norm S2j={np.mean([r['D_norm_post'] for r in s2j]):+.4f} "
          f"S1p={np.mean([r['D_norm_post'] for r in s1p]):+.4f}")


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sched in ("S2j", "S1p"):
        for seed in range(a.seeds):
            f = out / f"{sched}_{seed}.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
            else:
                r = run_e2(sched, seed)
                f.write_text(json.dumps(r))
                rows.append(r)
    analyze(rows)
