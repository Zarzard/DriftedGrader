"""E6 -- gradient-coupled drift (spec v0.2 E6; gated, run on explicit go-ahead).

A1, 2x2: drift {coupled (grader rotates 2deg/round toward the policy's improvement
direction), random (2deg/round random rotation, magnitude-matched)} x persistence
{persistent, churn every 8 rounds}, x 20 seeds, norm held constant, axes in S_ach.
Churn = fresh raters from the original pool (reset to the baseline grader v0) -- the
"rotation-resets-capture" mechanism (amendment: single-teacher reset-to-v0 realizes
the spec's churn intent).

Pre-registered hypotheses (report whichever way they fall -- the differential is the
adversarial signature either way):
  H-A: coupled x persistent shows the largest G_auc.
  H-B: churn attenuates coupled-drift harm MORE than random-drift harm.
Outcomes: G_auc (CRN-paired vs S0); anchor detectability = recovered/true net drift.

CLI: python -m graderdrift.experiments.e6 --out DIR [--seeds 0-19] [--nproc N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import drift
from .a1_run import run_a1

def cells_for(tag=""):
    return [f"E6cp{tag}", f"E6cc{tag}", f"E6rp{tag}", f"E6rc{tag}"]


CELLS = cells_for("")
LABEL = {"cp": "coupled+persistent", "cc": "coupled+churn",
         "rp": "random+persistent", "rc": "random+churn"}


def _lbl(nm):
    return LABEL[nm[2:4]]


def _one(job):
    out_root, name, seed = job
    mode = None if name == "S0" else "informed"
    run_a1(name, seed, run_dir=Path(out_root) / name / str(seed),
           estimator_mode=mode, schedule=drift.get_preset_a1(name))
    return (name, seed)


def run(out_root, seeds, nproc=1, tag=""):
    jobs = [(out_root, nm, s) for nm in cells_for(tag) + ["S0"] for s in seeds
            if not (Path(out_root) / nm / str(s) / "log.jsonl").exists()]
    print(f"{len(jobs)} jobs", flush=True)
    if nproc > 1:
        import multiprocessing as mp
        with mp.Pool(nproc) as pool:
            for r in pool.imap_unordered(_one, jobs):
                print("done", r, flush=True)
    else:
        for j in jobs:
            print("done", _one(j), flush=True)


def _load(out_root, name, seed, key):
    rows = [json.loads(l) for l in
            (Path(out_root) / name / str(seed) / "log.jsonl").read_text().splitlines()]
    return np.array([r[key] for r in rows])


def _boot(vals, rng, n=10000):
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    m = vals[idx].mean(axis=1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def analyze(out_root, seeds, tag=""):
    rng = np.random.default_rng(0)
    cp_nm, cc_nm, rp_nm, rc_nm = cells_for(tag)
    G = {}   # cell -> per-seed G_auc (paired vs S0, CRN)
    for nm in cells_for(tag):
        gs = []
        for s in seeds:
            g = float(np.mean(_load(out_root, "S0", s, "gold_wstar") - _load(out_root, nm, s, "gold_wstar")))
            gs.append(g)
        G[nm] = np.array(gs)

    eta = {"": "2", "1": "1", "4": "4"}[tag]
    print(f"E6 gradient-coupled 2x2 @ eta={eta}deg (G_auc vs S0, paired; {len(list(seeds))} seeds):")
    print(f"{'cell':20s} {'G_auc':>8s} {'90% CI':>20s}")
    for nm in cells_for(tag):
        lo, hi = _boot(G[nm], rng)
        print(f"{_lbl(nm):20s} {G[nm].mean():>8.3f} [{lo:>+7.3f},{hi:>+7.3f}]")

    print("H-A (coupled+persistent largest):")
    cp = G[cp_nm].mean()
    print(f"  G(cp)={cp:.3f} largest? {'holds' if cp==max(G[n].mean() for n in cells_for(tag)) else 'no'}")
    d = G[cp_nm] - G[rp_nm]; lo, hi = _boot(d, rng)
    print(f"  coupled>random (persistent): {d.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}] "
          f"{'(excludes 0)' if lo>0 else '(includes 0)'}")

    print("H-B (churn attenuates coupled more than random):")
    coupled_atten = G[cp_nm] - G[cc_nm]
    random_atten = G[rp_nm] - G[rc_nm]
    did = coupled_atten - random_atten
    lo, hi = _boot(did, rng)
    print(f"  coupled attenuation={coupled_atten.mean():+.3f}, random attenuation={random_atten.mean():+.3f}")
    print(f"  difference-in-differences={did.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}] "
          f"{'(excludes 0 -> churn resets coupled capture)' if lo>0 else '(includes 0)'}")


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--eta", default="2", choices=["1", "2", "4"],
                    help="rotation rate deg/round; picks the E6*{tag} cell set")
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    tag = {"2": "", "1": "1", "4": "4"}[a.eta]
    if not a.report_only:
        run(a.out, seeds, nproc=a.nproc, tag=tag)
    analyze(a.out, seeds, tag=tag)
