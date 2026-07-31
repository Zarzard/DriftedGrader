"""Stage P0 (HG0) -- the precondition that gates Plan 02.

Does ensemble disagreement fall while true regret rises under drift? If not, the
LLM-relevance framing (Section 1.4: conservative penalty lambda*std shrinks as the
reward becomes more wrong) collapses.

Reuses Plan 01's Stage B: DRIFT-SAT, misspecified linear RM (zeta=P phi, k<d),
5-member bootstrap ensemble. The runner logs D2_ens_disagree (on-policy ensemble
disagreement) and R_t per round.

HG0 metric: per-seed Spearman rho between disagreement and R_t under DRIFT-SAT.
Support if median rho < -0.3 with CI excluding 0. Also report the NONE contrast.

Run:  python -m analysis.run_p0 --seeds 40
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from drifted_grader.runner import Config, run
from analysis import figures  # noqa: F401  (imported so figure helpers are available)


PRIMARY = dict(stage="B", beta=0.05, d=6, k=3, rm="ensemble", K=5, T=100,
               betaT=10.0, q=64, buffer_regime="all")


def per_seed_spearman(drift, lam, seeds, warmup=20):
    rhos, dis_traj, R_traj = [], [], []
    for s in seeds:
        cfg = Config(drift=drift, lam=lam, **PRIMARY)
        rows = run(cfg, s)
        dis = np.array([r["D2_ens_disagree"] for r in rows])
        R = np.array([r["R_t"] for r in rows])
        dis_traj.append(dis); R_traj.append(R)
        d, r = dis[warmup:], R[warmup:]
        ok = np.isfinite(d) & np.isfinite(r)
        if ok.sum() > 5 and np.std(d[ok]) > 1e-9 and np.std(r[ok]) > 1e-9:
            rhos.append(spearmanr(d[ok], r[ok]).correlation)
    return np.array(rhos), np.array(dis_traj), np.array(R_traj)


def boot_ci(x, n=10000, rng=None):
    rng = rng or np.random.default_rng(0)
    b = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--figdir", default="figs")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    rho_sat, dis_sat, R_sat = per_seed_spearman("SAT", args.lam, seeds)
    rho_none, dis_none, R_none = per_seed_spearman("NONE", 0.0, seeds)

    m, lo, hi = boot_ci(rho_sat)
    mn, lon, hin = boot_ci(rho_none)
    print(f"Stage P0 / HG0  (lam={args.lam}, {len(seeds)} seeds)")
    print(f"  SAT  Spearman(disagreement, R_t): median={m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"  NONE Spearman(disagreement, R_t): median={mn:+.3f} [{lon:+.3f},{hin:+.3f}]")
    support = (m < -0.3) and (hi < 0)
    print(f"  HG0 {'SUPPORTED' if support else 'NOT supported'} "
          f"(need SAT median < -0.3, CI excludes 0)")
    print(f"  disagreement SAT {dis_sat[:, 20:].mean():.4f}  R_t SAT 2nd-half {R_sat[:, 50:].mean():.3f}")

    # Figure 0: disagreement + R_t over rounds, twin axes, drift vs no-drift
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import os
    fig, ax = plt.subplots(figsize=(6, 4)); ax2 = ax.twinx()
    t = np.arange(dis_sat.shape[1])
    ax.plot(t, dis_sat.mean(0), "b-", label="disagreement (SAT)")
    ax.plot(t, dis_none.mean(0), "b--", alpha=0.6, label="disagreement (NONE)")
    ax2.plot(t, R_sat.mean(0), "r-", label="R_t (SAT)")
    ax2.plot(t, R_none.mean(0), "r--", alpha=0.6, label="R_t (NONE)")
    ax.set_xlabel("round t"); ax.set_ylabel("ensemble disagreement", color="b")
    ax2.set_ylabel("regret R_t", color="r")
    ax.set_title(f"Fig 0 (HG0): disagreement falls while regret rises  (rho={m:+.2f})")
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    os.makedirs(args.figdir, exist_ok=True)
    fig.tight_layout(); fig.savefig(f"{args.figdir}/fig0_hg0.png", dpi=130); plt.close(fig)
    print(f"  wrote {args.figdir}/fig0_hg0.png")


if __name__ == "__main__":
    main()
