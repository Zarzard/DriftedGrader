"""Bayesian-REX decision demo: does the VaR bound, used EXACTLY as Bayesian-REX
intends -- to SELECT the policy with the best high-confidence (alpha-VaR) value and
to flag reward hacking -- make the wrong decision under grader drift + misspec?

Bayesian-REX (Brown et al. 2020) fits a reward posterior from preferences and uses
alpha-quantile performance lower bounds to (a) rank/select evaluation policies by
worst-case value and (b) detect reward hacking. It assumes a STATIC preference
distribution. We stress it under a drifting grader.

Per round t (current grader w_t), with the reward posterior fit on comparisons:
  pool  : a fixed, diverse set of candidate policies (specialised to each feature
          direction + Gibbs(w0) + uniform), so the true-best CHANGES as w_t drifts.
  v_true_j = w_t . f_j                          (true value under the CURRENT grader)
  lb_j     = alpha-quantile of { w_i . (P f_j) }   (B-REX high-confidence bound)
  vmean_j  = posterior-mean value

Decisions:
  j_brex   = argmax_j lb_j        (B-REX: pick the safest-by-VaR policy)
  j_naive  = argmax_j vmean_j     (naive: pick best point estimate -> over-opt target)
  j_oracle = argmax_j v_true_j
Metrics (normalised selection regret in [0,1], and a safety-violation rate):
  regret_brex  = (max v_true - v_true[j_brex]) / (max v_true - min v_true)
  regret_naive = ... [j_naive]
  violation    = 1[ v_true[j_brex] < lb[j_brex] ]  (bound dishonest about its OWN pick)

Run:  python -m analysis.brex_demo --seeds 20 --workers 8
"""
from __future__ import annotations

import argparse

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader
from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import policy_from_scores, rangenorm
from drifted_grader.policies import expected_features, sample_pairs
from drifted_grader.posterior import RewardPosterior

N, D, BETA, BETAT, Q, PRIOR = 256, 6, 0.05, 10.0, 64, 2.0
NS, BUF_CAP, ALPHA = 4000, 1800, 0.05


def _pool(arms):
    """Diverse candidate policies; true-best shifts as the grader drifts."""
    Phi, w0 = arms.Phi, arms.w0
    pols = [policy_from_scores(Phi[:, i], 1.0) for i in range(D)]     # specialise per feature
    pols += [policy_from_scores(Phi @ w0, b) for b in (2.0, 0.5)]
    pols.append(np.full(N, 1.0 / N))
    return pols


def run_brex(seed, drift, lam, k, T=100):
    prng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    arng = np.random.default_rng(np.random.SeedSequence(seed + 987654).spawn(1)[0])
    arms = make_arms(N, D, prng); Phi, w0 = arms.Phi, arms.w0
    P = make_projection(D, k, prng)
    grader = make_grader(drift, arms, BETAT, prng, lam=lam, T=T)
    pool = _pool(arms); poolf = [expected_features(pj, Phi) for pj in pool]

    buf_d, buf_y = [], []; post = RewardPosterior(P.shape[0]); m = np.zeros(D); rows = []
    for t in range(T):
        w_t = grader.w.copy()
        if buf_y:
            rmean = (Phi @ P.T) @ post.posterior_mean()
            pi = policy_from_scores(rangenorm(rmean), BETA)
        else:
            pi = np.full(N, 1.0 / N)
        pairs = sample_pairs(pi, Q, arng); y = grader.label(pairs, arng)
        buf_d.append(Phi[pairs[:, 0]] - Phi[pairs[:, 1]]); buf_y.append(y)
        alld = np.concatenate(buf_d); ally = np.concatenate(buf_y).astype(float)
        if len(ally) > BUF_CAP:
            idx = arng.choice(len(ally), BUF_CAP, replace=False); alld, ally = alld[idx], ally[idx]
        post.fit(alld @ P.T, ally, BETAT, prior_scale=PRIOR, n_samples=NS, rng=arng)

        v_true = np.array([float(w_t @ f) for f in poolf])
        lb = np.array([post.lower_bound(P @ f, ALPHA) for f in poolf])
        vmean = np.array([post.value_mean(P @ f) for f in poolf])
        j_brex, j_naive = int(lb.argmax()), int(vmean.argmax())
        rng_t = float(v_true.max() - v_true.min()) + 1e-9
        rows.append(dict(t=t, seed=int(seed), drift=drift, lam=lam, k=k,
                         regret_brex=(v_true.max() - v_true[j_brex]) / rng_t,
                         regret_naive=(v_true.max() - v_true[j_naive]) / rng_t,
                         violation=int(v_true[j_brex] < lb[j_brex]),
                         picked_wrong=int(j_brex != int(v_true.argmax())),
                         w_dist=float(np.linalg.norm(w_t - w0))))
        m = 0.5 * m + 0.5 * expected_features(pi, Phi); grader.step(m)
    return rows


def _worker(a):
    (drift, lam, k), s = a
    try:
        return run_brex(s, drift, lam, k), None
    except Exception as e:
        return None, f"{drift} lam={lam} k={k} s={s}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cells = [("NONE", 0.0, 6), ("SAT", 0.6, 6), ("NONE", 0.0, 3), ("SAT", 0.6, 3)]
    jobs = [(c, s) for c in cells for s in range(args.seeds)]
    print(f"B-REX demo: {len(cells)} cells x {args.seeds} seeds = {len(jobs)} runs")
    import pandas as pd, multiprocessing as mp, os
    rows, crashes = [], []
    with mp.Pool(args.workers) as pool:
        for r, e in pool.imap_unordered(_worker, jobs, chunksize=1):
            (rows.extend(r) if r else crashes.append(e))
    for c in crashes:
        print("  CRASH:", c)
    df = pd.DataFrame(rows); os.makedirs("runs", exist_ok=True)
    df.to_parquet("runs/brex_demo.parquet", index=False)
    T = df.t.max(); fin = df[df.t >= T - 10]
    print(f"\n{'spec':>10} {'drift':>5} | {'regret_brex':>11} {'regret_naive':>12} "
          f"{'wrong_pick':>10} {'violation':>9}")
    for k in (6, 3):
        for drift, lam in (("NONE", 0.0), ("SAT", 0.6)):
            g = fin[(fin.k == k) & (fin.drift == drift)]
            spec = "well-spec" if k == 6 else "misspec"
            print(f"{spec:>10} {drift:>5} | {g.regret_brex.mean():>11.3f} "
                  f"{g.regret_naive.mean():>12.3f} {g.picked_wrong.mean():>10.3f} "
                  f"{g.violation.mean():>9.3f}")
    print("\nB-REX failure = regret_brex spikes (bad selection) and/or violation>alpha")
    print("(bound dishonest about its OWN pick) specifically under misspec+SAT drift.")


if __name__ == "__main__":
    main()
