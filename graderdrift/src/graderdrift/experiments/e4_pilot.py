"""E4 pilot (spec v0.2 Section 4, E4): 5 schedules x 10 seeds + CRN controls.

Schedules {S1, S2, S5, S7a, S8t} + controls {S0, S7null}, all with the informed
tracker inline. Outputs: per-run predictors (net stats from m_k + Delta-align) and
outcomes (G_auc paired via A1_CONTROL, R_auc), the predictor-hierarchy adjusted
R^2 per tier with a seed-level bootstrap CI on H2, per-schedule harm table, A1
noise floors from the S0 cell, and the escalation-rule verdict (CI width > 0.2 ->
double seeds, cap 40).

CLI: ``python -m graderdrift.experiments.e4_pilot --out DIR [--seeds 0-9] [--nproc N]``
then ``--report_only`` to (re)analyze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..drift import A1_CONTROL
from ..envs.grid import TRUE_W
from .a1_run import run_a1

PILOT_SCHEDS = ["S1", "S2", "S5", "S7a", "S8t"]
CONTROLS = ["S0", "S7null"]


def _one(job):
    out_root, sched, seed = job
    run_a1(sched, seed, run_dir=Path(out_root) / sched / str(seed), estimator_mode="informed")
    return (sched, seed)


def run(out_root, seeds, nproc=1):
    jobs = [(out_root, s, seed) for s in CONTROLS + PILOT_SCHEDS for seed in seeds
            if not (Path(out_root) / s / str(seed) / "log.jsonl").exists()]
    print(f"{len(jobs)} jobs")
    if nproc > 1:
        import multiprocessing as mp
        with mp.Pool(nproc) as pool:
            for r in pool.imap_unordered(_one, jobs):
                print("done", r, flush=True)
    else:
        for j in jobs:
            print("done", _one(j), flush=True)


# ----------------------------- analysis ---------------------------------- #
def _load(out_root, sched, seed):
    rows = [json.loads(l) for l in
            (Path(out_root) / sched / str(seed) / "log.jsonl").read_text().splitlines()]
    return {
        "m_k": np.array([r["m_k"] for r in rows]),
        "gold_wstar": np.array([r["gold_wstar"] for r in rows]),
        "regret_wt": np.array([r["regret_wt"] for r in rows]),
        "e_shift": np.array([r["e_shift"] for r in rows]),
        "e_epi": np.array([r["e_epi"] for r in rows]),
    }


FIRST_EVENT = {"S1": 20, "S2": 20, "S5": 20, "S8t": 20, "S7a": 1}


def _angle(a, b):
    c = np.clip((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
    return float(np.degrees(np.arccos(c)))


def _features(run):
    M = run["m_k"]
    net_disp = float(np.linalg.norm(M[-1] - M[0]))
    net_ln = float(abs(np.log(np.linalg.norm(M[-1])) - np.log(np.linalg.norm(M[0]))))
    net_ang = _angle(M[-1], M[0])
    dalign = _angle(M[0], TRUE_W) - _angle(M[-1], TRUE_W)   # + = moved toward w*
    return net_disp, net_ln, net_ang, dalign


def _adj_r2(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n, p = len(y), X.shape[1]
    r2 = 1 - ss_res / ss_tot
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)


def analyze(out_root, seeds, n_boot=2000):
    rows = []
    for sched in PILOT_SCHEDS:
        ctrl_name = A1_CONTROL[sched]
        fe = FIRST_EVENT[sched]
        for seed in seeds:
            run_ = _load(out_root, sched, seed)
            ctrl = _load(out_root, ctrl_name, seed)
            g = float(np.mean(ctrl["gold_wstar"][fe - 1:] - run_["gold_wstar"][fe - 1:]))
            r_auc = float(np.mean(run_["regret_wt"][fe - 1:]))
            nd, nl, na, da = _features(run_)
            h3 = (float(np.mean(run_["e_shift"][fe - 1:])), float(np.mean(run_["e_epi"][fe - 1:])))
            rows.append(dict(sched=sched, seed=seed, G=g, R=r_auc, net_disp=nd,
                             net_lognorm=nl, net_angle=na, dalign=da,
                             e_shift_int=h3[0], e_epi_int=h3[1]))

    # per-schedule harm table
    print(f"{'sched':6s} {'G_auc mean':>10s} {'sd':>7s} {'R_auc':>8s} {'net_disp':>9s} {'dalign':>8s}")
    for sched in PILOT_SCHEDS:
        rs = [r for r in rows if r["sched"] == sched]
        print(f"{sched:6s} {np.mean([r['G'] for r in rs]):>10.3f} {np.std([r['G'] for r in rs]):>7.3f} "
              f"{np.mean([r['R'] for r in rs]):>8.3f} {np.mean([r['net_disp'] for r in rs]):>9.3f} "
              f"{np.mean([r['dalign'] for r in rs]):>8.2f}")

    # noise floors from the S0 cell
    s0_feats = [_features(_load(out_root, "S0", s)) for s in seeds]
    floors = dict(zip(("net_disp", "net_lognorm", "net_angle", "dalign"),
                      np.mean(np.array(s0_feats), axis=0)))
    print("\nA1 S0 floors:", {k: round(v, 3) for k, v in floors.items()})

    # predictor hierarchy on Y = G_auc
    y = np.array([r["G"] for r in rows])
    tiers = {
        "H0": ["net_disp"],
        "H1": ["net_disp", "net_lognorm", "net_angle"],
        "H2": ["net_disp", "net_lognorm", "net_angle", "dalign"],
        "H3": ["e_shift_int", "e_epi_int"],
    }
    Xall = {k: np.array([[r[c] for c in cols] for r in rows]) for k, cols in tiers.items()}
    r2 = {k: _adj_r2(X, y) for k, X in Xall.items()}
    print("\nadjusted R^2:", {k: round(v, 3) for k, v in r2.items()})

    # seed-level bootstrap CI on H2 R^2
    rng = np.random.default_rng(0)
    seeds_arr = np.array(sorted(seeds))
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(seeds_arr, size=len(seeds_arr), replace=True)
        idx = [i for s_ in pick for i, r in enumerate(rows) if r["seed"] == s_]
        yy = y[idx]
        if np.std(yy) < 1e-12:
            continue
        boots.append(_adj_r2(Xall["H2"][idx], yy))
    lo, hi = np.percentile(boots, [5, 95])
    width = hi - lo
    print(f"H2 R^2 90% CI [{lo:.3f}, {hi:.3f}] width {width:.3f} -> "
          f"{'OK (<=0.2)' if width <= 0.2 else 'ESCALATE: double seeds (cap 40)'}")
    print(f"gates preview: H2 R^2 >= 0.5: {'PASS' if r2['H2'] >= 0.5 else 'FAIL'}; "
          f"H2 >= 0.8*H3: {'PASS' if r2['H2'] >= 0.8 * r2['H3'] else 'FAIL'} "
          f"(H3={r2['H3']:.3f})")
    return rows


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    if not a.report_only:
        run(a.out, seeds, nproc=a.nproc)
    analyze(a.out, seeds)
