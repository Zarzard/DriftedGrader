"""Stage B driver: detection blindness (TP2 / H4) with the MLP reward model.

Claim: under ordering-changing drift, regret R_t rises while the field-standard
monitor -- held-out preference accuracy D1 on fresh current-grader comparisons --
stays high. So the standard practice is BLIND to the degradation.

H4 metric: per-seed Spearman rho between D1 and R_t.
  drift  (SAT): median |rho| < 0.3     (decorrelated -> blind)
  no-drift(NONE): median rho < -0.6     (accuracy up <-> regret down; monitor works)

Deliverables (from parquet logs):
  Fig 2  : R_t, J_proxy, D1, VR over rounds -- SAT vs NONE, MLP RM.
  Fig 2b : scatter of (D1, R_t) per round with fitted Spearman, drift vs no-drift.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from drifted_grader.runner import Config, run


PRIMARY = dict(beta=0.05, d=6, k=6, rm="mlp", T=100, betaT=10.0, q=64,
               q_eval=128, hidden=64, steps=60, schema="full")


def run_cell(drift, lam, seeds):
    rows_by_seed = []
    for s in seeds:
        cfg = Config(stage="B", drift=drift, lam=lam, **PRIMARY)
        rows_by_seed.append(run(cfg, s))
    return rows_by_seed


def h4_spearman(rows_by_seed, warmup=20):
    """Per-seed Spearman(D1, R_t) over rounds >= warmup."""
    rhos = []
    for rows in rows_by_seed:
        d1 = np.array([r["D1_heldout_acc"] for r in rows][warmup:])
        rt = np.array([r["R_t"] for r in rows][warmup:])
        ok = np.isfinite(d1) & np.isfinite(rt)
        if ok.sum() > 5 and np.std(d1[ok]) > 1e-9:
            rhos.append(spearmanr(d1[ok], rt[ok]).correlation)
    return np.array(rhos)


def summarize(rows_by_seed, label):
    R = np.array([[r["R_t"] for r in rows] for rows in rows_by_seed])
    D1 = np.array([[r["D1_heldout_acc"] for r in rows] for rows in rows_by_seed])
    half = R.shape[1] // 2
    print(f"  {label}: R_bar={R.mean():.3f}  R_2ndhalf={R[:, half:].mean():.3f}  "
          f"D1_2ndhalf={D1[:, half:].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.1)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    print(f"Stage B blindness (MLP RM), lam={args.lam}, {len(seeds)} seeds")
    sat = run_cell("SAT", args.lam, seeds)
    none = run_cell("NONE", 0.0, seeds)
    summarize(sat, "SAT ")
    summarize(none, "NONE")

    rho_sat = h4_spearman(sat)
    rho_none = h4_spearman(none)
    print(f"\nH4 Spearman(D1, R_t):")
    print(f"  SAT  median rho = {np.median(rho_sat):+.3f}  "
          f"(H4 supports if |rho|<0.3: {'YES' if abs(np.median(rho_sat))<0.3 else 'no'})")
    print(f"  NONE median rho = {np.median(rho_none):+.3f}  "
          f"(monitor works if rho<-0.6: {'YES' if np.median(rho_none)<-0.6 else 'no'})")
    return sat, none


if __name__ == "__main__":
    main()
