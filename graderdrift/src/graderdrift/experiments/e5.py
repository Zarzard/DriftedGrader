"""E5 -- matched-budget refresh baseline, dual-reference sign flip (spec v0.2 E5).

A1, arms {accumulate, window (W=10), refresh (f=0.7)} x {S1, S2, S5} x seeds, all
at budget B=100 pairs/round. Each drift run pairs with its S0 control at the same
seed (CRN) for the frozen-reference harm G_auc; R_auc is the current-standard harm.

Pre-registered prediction (descriptive, publishes either way): the arm RANKING
FLIPS between references -- accumulate-all best under the frozen reference (old
labels are the clean ones), refresh best under the current standard (tracking the
live rater is the point). This is the corruption-vs-renegotiation dichotomy made
quantitative -- the comparison any future correction method must beat.

CLI: ``python -m graderdrift.experiments.e5 --out DIR [--seeds 0-19] [--nproc N]``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .a1_run import run_a1

ARMS = ["accumulate", "window", "refresh"]
SCHEDS = ["S1", "S2", "S5"]
FIRST_EVENT = 20


def _one(job):
    out_root, arm, sched, seed = job
    run_a1(sched, seed, run_dir=Path(out_root) / arm / sched / str(seed),
           estimator_mode=None, arm=arm)
    return (arm, sched, seed)


def _dispatch(job):
    """job = (out_root, arm, sched, seed) for any schedule incl. the S0 control."""
    out_root, arm, sched, seed = job
    run_a1(sched, seed, run_dir=Path(out_root) / arm / sched / str(seed),
           estimator_mode=None, arm=arm)
    return (arm, sched, seed)


def run(out_root, seeds, nproc=1):
    jobs = [(out_root, a, s, seed) for a in ARMS for s in SCHEDS + ["S0"] for seed in seeds
            if not (Path(out_root) / a / s / str(seed) / "log.jsonl").exists()]
    print(f"{len(jobs)} jobs", flush=True)
    if nproc > 1:
        import multiprocessing as mp
        with mp.Pool(nproc) as pool:
            for r in pool.imap_unordered(_dispatch, jobs):
                print("done", r, flush=True)
    else:
        for j in jobs:
            print("done", _dispatch(j), flush=True)


def _gold(out_root, arm, sched, seed, key):
    rows = [json.loads(l) for l in
            (Path(out_root) / arm / sched / str(seed) / "log.jsonl").read_text().splitlines()]
    return np.array([r[key] for r in rows])


def analyze(out_root, seeds):
    print(f"{'arm':11s} {'sched':5s} {'G_auc (frozen)':>15s} {'R_auc (current)':>16s}")
    frozen, current = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    for arm in ARMS:
        for sched in SCHEDS:
            Gs, Rs = [], []
            for seed in seeds:
                g_ctrl = _gold(out_root, arm, "S0", seed, "gold_wstar")[FIRST_EVENT - 1:]
                g_run = _gold(out_root, arm, sched, seed, "gold_wstar")[FIRST_EVENT - 1:]
                Gs.append(float(np.mean(g_ctrl - g_run)))
                Rs.append(float(np.mean(_gold(out_root, arm, sched, seed, "regret_wt")[FIRST_EVENT - 1:])))
            frozen[arm].extend(Gs); current[arm].extend(Rs)
            print(f"{arm:11s} {sched:5s} {np.mean(Gs):>15.3f} {np.mean(Rs):>16.3f}")
    print("\npooled means (lower = less harm):")
    print(f"{'arm':11s} {'G_auc (frozen)':>15s} {'R_auc (current)':>16s}")
    for arm in ARMS:
        print(f"{arm:11s} {np.mean(frozen[arm]):>15.3f} {np.mean(current[arm]):>16.3f}")
    best_frozen = min(ARMS, key=lambda a: np.mean(frozen[a]))
    best_current = min(ARMS, key=lambda a: np.mean(current[a]))
    print(f"\nbest under FROZEN reference:  {best_frozen}")
    print(f"best under CURRENT reference: {best_current}")
    print("RANKING FLIP" if best_frozen != best_current else "no flip (same arm best both)")


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    if not a.report_only:
        run(a.out, seeds, nproc=a.nproc)
    analyze(a.out, seeds)
