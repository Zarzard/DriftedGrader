"""M6 DoD check (spec v0.2 Section 7): A1/S0 gold curve rises and plateaus;
A1/S1 shows G_auc > 0 -- else apply the sanctioned Risks knobs in order.

Runs {S0, S1, S5} x seeds x knob variants {default, tau002 (tau_pol=0.02),
np50 (n_pairs=50)}, computes paired G_auc per variant, and prints the decision
table. CLI: ``python -m graderdrift.experiments.m6_dod --out DIR [--seeds 0-2]``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .a1_run import run_a1

VARIANTS = {
    "default": {},
    "tau002": {"tau_pol": 0.02},
    "np50": {"n_pairs": 50},
}
SCHEDS = ["S0", "S1", "S5"]


def _one(job):
    out_root, variant, sched, seed = job
    kwargs = VARIANTS[variant]
    res = run_a1(sched, seed, **kwargs)
    payload = {
        "gold_wstar": res["gold_wstar"].tolist(),
        "regret_wt": res["regret_wt"].tolist(),
        "e_shift": res["e_shift"].tolist(),
        "e_epi": res["e_epi"].tolist(),
        "first_event": int(res["first_event"]) if res["first_event"] < 10**8 else 20,
    }
    p = Path(out_root) / variant / sched / f"{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    return (variant, sched, seed)


def run(out_root, seeds, nproc=1):
    jobs = [(out_root, v, s, seed) for v in VARIANTS for s in SCHEDS for seed in seeds
            if not (Path(out_root) / v / s / f"{seed}.json").exists()]
    print(f"{len(jobs)} jobs")
    if nproc > 1:
        import multiprocessing as mp
        with mp.Pool(nproc) as pool:
            for r in pool.imap_unordered(_one, jobs):
                print("done", r, flush=True)
    else:
        for j in jobs:
            print("done", _one(j), flush=True)


def report(out_root, seeds):
    print(f"{'variant':9s} {'sched':5s} {'G_auc mean':>10s} {'per-seed':>30s} {'R_auc':>8s}")
    for v in VARIANTS:
        base = {}
        for seed in seeds:
            base[seed] = json.loads((Path(out_root) / v / "S0" / f"{seed}.json").read_text())
        for s in SCHEDS:
            if s == "S0":
                continue
            gs, rs = [], []
            for seed in seeds:
                run_j = json.loads((Path(out_root) / v / s / f"{seed}.json").read_text())
                fe = run_j["first_event"]
                g0 = np.array(base[seed]["gold_wstar"])[fe - 1:]
                g1 = np.array(run_j["gold_wstar"])[fe - 1:]
                gs.append(float(np.mean(g0 - g1)))
                rs.append(float(np.mean(np.array(run_j["regret_wt"])[fe - 1:])))
            print(f"{v:9s} {s:5s} {np.mean(gs):>10.4f} {np.array2string(np.array(gs), precision=3):>30s} "
                  f"{np.mean(rs):>8.4f}")
    # S0 rises-and-plateaus check (default variant, seed set)
    for seed in seeds:
        g = np.array(json.loads((Path(out_root) / "default" / "S0" / f"{seed}.json").read_text())["gold_wstar"])
        print(f"S0 seed {seed}: r1={g[0]:.2f} r5={g[4]:.2f} r20={g[19]:.2f} r40={g[-1]:.2f} "
              f"(uniform baseline ~ -8.5)")


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    lo, hi = a.seeds.split("-") if "-" in a.seeds else (a.seeds, a.seeds)
    seeds = list(range(int(lo), int(hi) + 1))
    if not a.report_only:
        run(a.out, seeds, nproc=a.nproc)
    report(a.out, seeds)
