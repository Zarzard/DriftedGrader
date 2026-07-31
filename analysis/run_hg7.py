"""Stage P6 / HG7 -- does CAPACITY amplify the coverage failure? (updated plan)

The scale mechanism: a higher-capacity reward model fits the pooled grader MIXTURE
better (lower held-out loss on the pooled buffer, tighter/more confident) while
covering the CURRENT grader worse. Figure-9 claim, two axes reported together.

Instruments (each with its OWN lambda=0 reference row, Section 1.6):
  - linear         : RewardPosterior on phi.
  - mlp (refit)    : train MLPBT on the pooled buffer each round -> trunk features
                     H = tanh(W1 phi + b1) -> last-layer Laplace (RewardPosterior on
                     H, fit with the known betaT so the scale is pinned).
Capacity axis = MLP hidden width. Drift-attributable drop = C(lam=0) - C(lam>0),
paired by seed, per instrument. Pooled held-out log-loss (lower = fits mixture
better) is the second axis.

Run:  python -m analysis.run_hg7 --seeds 8 --workers 8
"""
from __future__ import annotations

import argparse

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader
from drifted_grader.reward_models import MLPBT
from drifted_grader.metrics import policy_from_scores, rangenorm, regret_Rt, rank_correlation
from drifted_grader.policies import expected_features, sample_pairs
from drifted_grader.posterior import RewardPosterior

N, D, BETA, BETAT, Q, PRIOR, NS = 256, 6, 0.05, 10.0, 64, 2.0, 4000
BUF_CAP = 1800


def _eval_pols(arms):
    Phi, w0 = arms.Phi, arms.w0
    return [policy_from_scores(Phi @ w0, b) for b in (2.0, 0.2, 0.05)]


def _trunk_feats(mlp, Phi):
    return np.tanh(Phi @ mlp.W1.T + mlp.b1)          # (N, h)


def run_hg7(seed, drift, lam, cap, T=100):
    prng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    arng = np.random.default_rng(np.random.SeedSequence(seed + 42).spawn(1)[0])
    arms = make_arms(N, D, prng); Phi, w0 = arms.Phi, arms.w0
    grader = make_grader(drift, arms, BETAT, prng, lam=lam, T=T)
    evalp = _eval_pols(arms); evalf = [expected_features(pe, Phi) for pe in evalp]
    mlp = None if cap == "linear" else MLPBT(D, hidden=int(cap.split("_")[1]),
                                             steps=50, seed=seed)
    bp, by = [], []; m = np.zeros(D); rows = []
    post = RewardPosterior(D if cap == "linear" else int(cap.split("_")[1]))
    for t in range(T):
        w_t = grader.w.copy()
        if by:
            if cap == "linear":
                Hf = Phi; r = Hf @ post.posterior_mean()
            else:
                Hf = _trunk_feats(mlp, Phi); r = Hf @ post.posterior_mean()
            pi = policy_from_scores(rangenorm(r), BETA)
        else:
            Hf = Phi; pi = np.full(N, 1.0 / N)
        pairs = sample_pairs(pi, Q, arng); y = grader.label(pairs, arng)
        bp.append(pairs); by.append(y)
        allp = np.concatenate(bp); ally = np.concatenate(by).astype(float)
        if len(ally) > BUF_CAP:
            idx = arng.choice(len(ally), BUF_CAP, replace=False); allp, ally = allp[idx], ally[idx]
        if cap == "linear":
            H = Phi
        else:
            mlp.fit_buffer(Phi, allp, ally, arng)       # refit trunk on pooled buffer
            H = _trunk_feats(mlp, Phi)
        dH = H[allp[:, 0]] - H[allp[:, 1]]
        # held-out pooled log-loss (train/test split) under posterior mean
        ntr = int(0.8 * len(ally)); tr = slice(0, ntr); te = slice(ntr, None)
        post.fit(dH[tr], ally[tr], BETAT, prior_scale=PRIOR, n_samples=NS, rng=arng)
        wm = post.posterior_mean()
        z = BETAT * (dH[te] @ wm)
        ll = float(np.mean(ally[te] * np.logaddexp(0, -z) + (1 - ally[te]) * np.logaddexp(0, z)))

        f = expected_features(pi, Phi); fH = pi @ H
        v_true = float(w_t @ f)
        lb = post.lower_bound(fH, 0.05)
        lb_eval = [post.lower_bound(pe @ H, 0.05) for pe in evalp]
        v_eval = [float(w_t @ fe) for fe in evalf]
        R_t, _ = regret_Rt(pi, Phi, w_t, BETA)
        rows.append(dict(t=t, seed=int(seed), drift=drift, lam=lam, cap=cap,
                         hit=int(v_true >= lb), pooled_ll=ll, R_t=R_t,
                         rank_corr=rank_correlation(lb_eval, v_eval),
                         K_t=float(np.trace(post.cov))))
        m = 0.5 * m + 0.5 * f; grader.step(m)
    return rows


def _worker(a):
    (drift, lam, cap), s = a
    try:
        return run_hg7(s, drift, lam, cap), None
    except Exception as e:
        return None, f"{cap} {drift} lam={lam} s={s}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--caps", nargs="+", default=["linear", "mlp_32", "mlp_128"])
    args = ap.parse_args()
    cells = []
    for cap in args.caps:
        cells.append(("NONE", 0.0, cap)); cells.append(("SAT", 0.4, cap))
    jobs = [(c, s) for c in cells for s in range(args.seeds)]
    print(f"HG7: {len(cells)} cells x {args.seeds} seeds = {len(jobs)} runs")
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
    df = pd.DataFrame(rows); import os
    os.makedirs("runs", exist_ok=True); df.to_parquet("runs/hg7.parquet", index=False)
    T = df.t.max(); fin = df[df.t >= T - 5]
    print(f"\n{'cap':>8} | {'C(lam=0)':>8} {'C(lam=.4)':>9} {'drop':>6} | "
          f"{'pooledLL_0':>10} {'pooledLL_.4':>11} | {'rank_.4':>7} {'K_.4':>7}")
    for cap in args.caps:
        c0 = fin[(fin.cap == cap) & (fin.lam == 0.0)]
        c1 = fin[(fin.cap == cap) & (fin.lam == 0.4)]
        C0, C1 = c0.hit.mean(), c1.hit.mean()
        print(f"{cap:>8} | {C0:>8.3f} {C1:>9.3f} {C0 - C1:>+6.3f} | "
              f"{c0.pooled_ll.mean():>10.4f} {c1.pooled_ll.mean():>11.4f} | "
              f"{c1.rank_corr.mean():>7.3f} {c1.K_t.mean():>7.4f}")
    print("\nHG7 supports if drift drop INCREASES with capacity while pooled LL FALLS")
    print("(higher capacity fits the mixture better but covers the current grader worse).")


if __name__ == "__main__":
    main()
