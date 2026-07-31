"""Stage P1 -- posterior VaR coverage under drift (HG1, HG2, HG3).

Closed loop with the reward posterior in the loop:
  each round: deploy pi_t = Gibbs(posterior-mean reward) -> sample comparisons
  (on-policy OR broad, the coverage-breadth axis) -> label with the CURRENT grader
  w_t -> accumulate -> refit posterior -> measure coverage / width / concentration
  against the current grader -> grader.step.

Coverage (Section 4.1): V_true = w_t . f_{pi_t} (full d-dim current grader);
LB = alpha-quantile of { w_i . (P f_{pi_t}) }, w_i ~ posterior over the RM's
(possibly projected, k<=d) reward. hit = V_true >= LB.

Factorial (HG3): {well-spec k=d, misspec k<d} x {lam=0, lam>0} x breadth x alpha.
The dangerous-quadrant prediction: on-policy narrow queries -> posterior concentrates
on a stale compromise -> tight AND wrong; broad queries -> contradictions -> widens
-> loose but honest.

Run:  python -m analysis.run_p1 --seeds 20 --workers 8
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader
from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import (policy_from_scores, rangenorm, regret_Rt,
                                     bound_width, rank_correlation)
from drifted_grader.policies import expected_features, sample_pairs
from drifted_grader.posterior import RewardPosterior

ALPHAS = (0.05, 0.10, 0.25)
N, D, BETA, BETAT, Q, PRIOR = 256, 6, 0.05, 10.0, 64, 2.0
NS, BUF_CAP = 4000, 1800   # posterior samples; buffer subsample cap for the fit


def _eval_pols(arms):
    Phi, w0 = arms.Phi, arms.w0
    pols = [policy_from_scores(Phi @ w0, b) for b in (2.0, 0.2, 0.05)]
    pols.append(np.full(N, 1.0 / N))
    o = np.zeros(N); o[(Phi @ w0).argmax()] = 1.0; pols.append(o)   # good under w0, bad under w_T
    return pols


def run_coverage(seed, drift, lam, k, breadth, T=100):
    ss = np.random.SeedSequence(seed)
    prng = np.random.default_rng(ss.spawn(1)[0])
    arng = np.random.default_rng(np.random.SeedSequence(seed + 987654).spawn(1)[0])
    arms = make_arms(N, D, prng)
    Phi, w0 = arms.Phi, arms.w0
    P = make_projection(D, k, prng)                    # identity if k>=d
    grader = make_grader(drift, arms, BETAT, prng, lam=lam, T=T)
    evalp = _eval_pols(arms)
    evalf = [expected_features(pe, Phi) for pe in evalp]

    buf_d, buf_y = [], []
    post = RewardPosterior(P.shape[0])
    m = np.zeros(D)
    rows = []
    for t in range(T):
        w_t = grader.w.copy()
        # deployed policy = Gibbs on posterior-mean reward (uniform until data)
        if buf_y:
            rmean = (Phi @ P.T) @ post.posterior_mean()
            pi = policy_from_scores(rangenorm(rmean), BETA)
        else:
            pi = np.full(N, 1.0 / N)

        pairs = (sample_pairs(pi, Q, arng) if breadth == "onpolicy"
                 else _uniform_pairs(arng))
        y = grader.label(pairs, arng)
        buf_d.append(Phi[pairs[:, 0]] - Phi[pairs[:, 1]]); buf_y.append(y)

        alld = np.concatenate(buf_d); ally = np.concatenate(buf_y).astype(float)
        if len(ally) > BUF_CAP:                        # subsample for speed (CG4)
            idx = arng.choice(len(ally), BUF_CAP, replace=False)
            alld, ally = alld[idx], ally[idx]
        dz = alld @ P.T
        post.fit(dz, ally, BETAT, prior_scale=PRIOR, n_samples=NS, rng=arng)

        f = expected_features(pi, Phi)
        fz = P @ f
        range_t = float((Phi @ w_t).ptp())
        v_true = float(w_t @ f)
        lbs = {a: post.lower_bound(fz, a) for a in ALPHAS}
        vmean = post.value_mean(fz)
        # rank validity over the fixed eval set
        lb_eval = [post.lower_bound(P @ fe, 0.05) for fe in evalf]
        v_eval = [float(w_t @ fe) for fe in evalf]

        R_t, _ = regret_Rt(pi, Phi, w_t, BETA)
        row = dict(t=t, seed=int(seed), drift=drift, lam=lam, k=k, breadth=breadth,
                   R_t=R_t, range_t=range_t, K_t=float(np.trace(post.cov)),
                   ess=post.ess, W_t=bound_width(vmean, lbs[0.05], range_t),
                   rank_corr=rank_correlation(lb_eval, v_eval),
                   w_dist=float(np.linalg.norm(w_t - w0)))
        for a in ALPHAS:
            row[f"hit_{a}"] = int(v_true >= lbs[a])
        rows.append(row)

        m = 0.5 * m + 0.5 * f
        grader.step(m)
    return rows


def _uniform_pairs(rng):
    i = rng.integers(0, N, Q); j = rng.integers(0, N, Q); mk = i != j
    return np.stack([i[mk], j[mk]], 1)


def _worker(args):
    (drift, lam, k, breadth), seed = args
    try:
        return run_coverage(seed, drift, lam, k, breadth), None
    except Exception as e:
        return None, f"{drift} lam={lam} k={k} {breadth} seed={seed}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--ks", type=int, nargs="+", default=[6, 3])
    ap.add_argument("--out", default="runs/p1_rows.parquet")
    args = ap.parse_args()

    cells = []
    for k in args.ks:
        for lam in args.lams:
            drift = "SAT" if lam > 0 else "NONE"
            for breadth in ("onpolicy", "broad"):
                cells.append((drift, lam, k, breadth))
    jobs = [(c, s) for c in cells for s in range(args.seeds)]
    print(f"Stage P1: {len(cells)} cells x {args.seeds} seeds = {len(jobs)} runs")

    import pandas as pd
    all_rows, crashes = [], []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for rows, err in pool.imap_unordered(_worker, jobs, chunksize=1):
                (all_rows.extend(rows) if rows else crashes.append(err))
    else:
        for job in jobs:
            rows, err = _worker(job)
            (all_rows.extend(rows) if rows else crashes.append(err))
    for c in crashes:
        print("  CRASH:", c)

    df = pd.DataFrame(all_rows)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)

    # ---- HG3 table: final-round coverage per cell ----
    T = df["t"].max()
    fin = df[df["t"] >= T - 5]
    print(f"\nFinal-round coverage (alpha=0.05), width W_t, by cell:")
    print(f"{'k':>2} {'lam':>4} {'breadth':>9} | {'cov@.05':>7} {'W_t':>6} {'K_t':>7} {'R_t':>6} {'rankcorr':>8}")
    cellstats = {}
    for (k, lam, br), g in fin.groupby(["k", "lam", "breadth"]):
        cov = g["hit_0.05"].mean(); W = g["W_t"].mean(); K = g["K_t"].mean()
        R = g["R_t"].mean(); rc = g["rank_corr"].mean()
        cellstats[(k, lam, br)] = (cov, W)
        print(f"{k:>2} {lam:>4} {br:>9} | {cov:>7.3f} {W:>6.3f} {K:>7.4f} {R:>6.3f} {rc:>8.3f}")

    # HG3 interaction: does drift drop coverage BEYOND misspec-only?
    print("\nHG3 (drift vs misspec-alone), on-policy, alpha=0.05:")
    for k in args.ks:
        c0 = cellstats.get((k, 0.0, "onpolicy"), (np.nan,))[0]
        c1 = cellstats.get((k, max(args.lams), "onpolicy"), (np.nan,))[0]
        print(f"  k={k}: cov(lam=0)={c0:.3f}  cov(lam>0)={c1:.3f}  drift drop={c0-c1:+.3f}")
    print("  HG3 needs: drift drop at misspec k<d to exceed the k=d (well-spec) drift drop.")


if __name__ == "__main__":
    main()
