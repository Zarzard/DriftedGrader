"""MLP-ensemble conservative bound under drift (the neural-RLHF regime).

Where the linear posterior (Stage P1) had robust marginal coverage but broken
RANKING, the MLP ensemble can over-optimise -> the conservative bound mean_k - lam*std_k
may be confidently wrong. Because an MLP reward has a free additive constant, absolute
coverage is ill-posed; we measure the affine-invariant / well-posed signals:

  rank_corr : Spearman(LB_ens(pi^j), V_true(pi^j)) over a fixed eval set -- the
              Bayesian-REX use case, directly comparable to Stage P1.
  pen_t     : lam * std_k(pi_t . r_k) -- the conservative safety margin (HG4 inversion:
              does it shrink as regret grows?).
  R_cons vs R_mean : regret of the conservative policy vs the mean policy -- does the
              mean-lam*std penalty actually PROTECT against over-optimisation under drift?

Deployed policy = Gibbs on the conservative reward. MLP ensemble on full phi (k=d);
over-optimisation comes from non-linearity + on-policy coverage gaps (Plan 01).

Run:  python -m analysis.run_p2_mlp --seeds 10 --workers 8
"""
from __future__ import annotations

import argparse

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader
from drifted_grader.reward_models import EnsembleBT
from drifted_grader.metrics import (policy_from_scores, rangenorm, regret_Rt,
                                     rank_correlation)
from drifted_grader.policies import expected_features, sample_pairs

N, D, BETA, BETAT, Q = 256, 6, 0.05, 10.0, 64
LAM_CONS = 1.0
HIDDEN, STEPS, K = 48, 40, 5


def _eval_pols(arms):
    Phi, w0 = arms.Phi, arms.w0
    pols = [policy_from_scores(Phi @ w0, b) for b in (2.0, 0.2, 0.05)]
    pols.append(np.full(N, 1.0 / N))
    o = np.zeros(N); o[(Phi @ w0).argmax()] = 1.0; pols.append(o)
    return pols


def run_mlp(seed, drift, lam, breadth, T=100):
    prng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    arng = np.random.default_rng(np.random.SeedSequence(seed + 555).spawn(1)[0])
    arms = make_arms(N, D, prng); Phi = arms.Phi
    grader = make_grader(drift, arms, BETAT, prng, lam=lam, T=T)
    eval_pols = _eval_pols(arms)
    evalf = [expected_features(pe, Phi) for pe in eval_pols]
    ens = EnsembleBT(D, P=None, K=K, base="mlp", hidden=HIDDEN, steps=STEPS)
    bp, by = [], []
    rows = []
    m = np.zeros(D)
    for t in range(T):
        w_t = grader.w.copy()
        if by:
            preds = ens.predict_all(Phi)                 # (K, N)
            mean_r = preds.mean(0); std_r = preds.std(0)
            r_cons = mean_r - LAM_CONS * std_r
            pi = policy_from_scores(rangenorm(r_cons), BETA)
            pi_mean = policy_from_scores(rangenorm(mean_r), BETA)
        else:
            preds = np.zeros((K, N)); pi = np.full(N, 1.0 / N); pi_mean = pi

        pairs = (sample_pairs(pi, Q, arng) if breadth == "onpolicy"
                 else _uniform(arng))
        y = grader.label(pairs, arng)
        bp.append(pairs); by.append(y)
        allp = np.concatenate(bp); ally = np.concatenate(by).astype(float)
        ens.fit_buffer(Phi, allp, ally, arng)

        preds = ens.predict_all(Phi)                      # (K, N) per-arm head rewards
        # head value of a policy pol = pol . r_k  ->  preds @ pol : (K,)
        def lb_ens(pol):
            vals = preds @ pol
            return float(vals.mean() - LAM_CONS * vals.std())
        pen = LAM_CONS * float((preds @ pi).std())        # conservative margin, deployed pi
        R_cons, _ = regret_Rt(pi, Phi, w_t, BETA)
        R_mean, _ = regret_Rt(pi_mean, Phi, w_t, BETA)
        disagree = float((preds.std(0) * pi).sum())       # on-policy ensemble disagreement
        lb_eval = [lb_ens(pe) for pe in eval_pols]
        v_eval = [float(w_t @ fe) for fe in evalf]
        rows.append(dict(t=t, seed=int(seed), drift=drift, lam=lam, breadth=breadth,
                         pen_t=pen, R_cons=R_cons, R_mean=R_mean, disagree=disagree,
                         rank_corr=rank_correlation(lb_eval, v_eval),
                         w_dist=float(np.linalg.norm(w_t - arms.w0))))
        m = 0.5 * m + 0.5 * expected_features(pi, Phi)
        grader.step(m)
    return rows


def _uniform(rng):
    i = rng.integers(0, N, Q); j = rng.integers(0, N, Q); mk = i != j
    return np.stack([i[mk], j[mk]], 1)


def _worker(a):
    (drift, lam, breadth), seed = a
    try:
        return run_mlp(seed, drift, lam, breadth), None
    except Exception as e:
        return None, f"{drift} lam={lam} {breadth} s={seed}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cells = [("NONE", 0.0, "onpolicy"), ("SAT", 0.4, "onpolicy"),
             ("NONE", 0.0, "broad"), ("SAT", 0.4, "broad")]
    jobs = [(c, s) for c in cells for s in range(args.seeds)]
    print(f"P2-MLP: {len(cells)} cells x {args.seeds} seeds = {len(jobs)} runs")
    import pandas as pd
    rows, crashes = [], []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for r, e in pool.imap_unordered(_worker, jobs, chunksize=1):
                (rows.extend(r) if r else crashes.append(e))
    else:
        for j in jobs:
            r, e = _worker(j); (rows.extend(r) if r else crashes.append(e))
    for c in crashes:
        print("  CRASH:", c)
    df = pd.DataFrame(rows)
    import os
    os.makedirs("runs", exist_ok=True); df.to_parquet("runs/p2_mlp.parquet", index=False)

    from scipy.stats import spearmanr
    print(f"\n{'drift':>5} {'breadth':>9} | {'rank_final':>10} {'R_cons':>7} {'R_mean':>7} "
          f"{'pen_final':>9} {'pen-R corr':>10}")
    for (drift, br), g in df.groupby(["drift", "breadth"]):
        T = g.t.max(); fin = g[g.t >= T - 5]
        rc = fin.rank_corr.mean(); Rc = fin.R_cons.mean(); Rm = fin.R_mean.mean()
        pn = fin.pen_t.mean()
        # HG4 inversion: per-seed corr of pen_t vs R_cons over rounds
        corrs = []
        for s, gs in g.groupby("seed"):
            gs = gs[gs.t > 20]
            if gs.pen_t.std() > 1e-9 and gs.R_cons.std() > 1e-9:
                corrs.append(spearmanr(gs.pen_t, gs.R_cons).correlation)
        print(f"{drift:>5} {br:>9} | {rc:>10.3f} {Rc:>7.3f} {Rm:>7.3f} {pn:>9.4f} "
              f"{np.median(corrs):>10.3f}")
    print("\nrank_final: Spearman(LB_ens, V_true) over eval set (compare to P1 linear).")
    print("pen-R corr < 0 => HG4 inversion (penalty shrinks as regret grows).")
    print("R_cons vs R_mean: does the conservative penalty reduce regret under drift?")


if __name__ == "__main__":
    main()
