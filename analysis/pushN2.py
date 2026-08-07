"""Paper B, second experiment: does the endogenous null-space PUSH (confirmed by
pushN.py) actually CAUSE the selection damage? pushN found the push is real but
its across-seed correlation with damage ran backwards -- because that damage
metric's eval policies were defined by the INITIAL grader w0 and barely varied
along N, so they could not feel a push into N.

Fix: measure damage on eval pools that DO vary along the reward model's null
space N. As ||Pi_N w_t|| grows, the blind bound (which sees only S = row(P))
should increasingly mis-rank / mis-select policies whose true-value differences
load on N. Prediction (the causal link):
    corr(dfrac, rank on the N-pool)  < 0     push hurts N-ranking
    corr(dfrac, regret on the N-pool) > 0     push hurts N-selection
    corr(dfrac, rank on an S-pool)   ~ 0      push does NOT hurt what the RM sees
and for the closed_FULL control (representation sees N), NO dependence on dfrac.

N-pool  = one-hot policies on the arms with the LARGEST ||Pi_N phi(arm)|| (true-
          value differences load on N -> only a model that reads N can rank them)
S-pool  = one-hot policies on the arms with the largest ||P phi(arm)|| (loads on
          the readable subspace -> even a blind-to-N model should rank them)
Both pools are FIXED per seed (depend on arms + P only), evaluated at the final
round against w_T and the final posterior.

Reuses pushN.run_arm verbatim for the trajectory + push statistic; adds the pool
damage inside a thin re-run so the two are perfectly seed-paired.

Run: python -m analysis.pushN2 --out <dir> --seeds 48 --workers 16
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader
from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import policy_from_scores, rangenorm, rank_correlation
from drifted_grader.posterior import RewardPosterior
from drifted_grader.policies import expected_features, sample_pairs

from analysis.pushN import (N, D, K, BETA, BETAT, Q, PRIOR, NS, BUF_CAP, T, LAST)

POOL = 15               # policies per eval pool
ARMS2 = ("closed_blind", "closed_full")


def _null_basis(P):
    """Orthonormal basis (d x (d-k)) for null(P); columns span S^perp = N."""
    d = P.shape[1]; k = P.shape[0]
    _, _, Vt = np.linalg.svd(P, full_matrices=True)
    return Vt[k:].T


def _npool(Phi, P):
    """N-loaded one-hot eval pool: the arms whose features project most onto the
    RM null space N. Their true-value differences load on N, so a model that
    cannot read N mis-ranks them -- and mis-ranks them MORE as w moves into N.
    (Control against dfrac is the closed_FULL model, which reads N, on the SAME
    pool -- cleaner than a hand-built S-pool.)"""
    Nb = _null_basis(P)                       # d x (d-k)
    n_mag = np.linalg.norm(Phi @ Nb, axis=1)  # each arm's magnitude in N
    return np.argsort(-n_mag)[:POOL]


def _damage(arms_idx, Phi, Pfit, w_T, post, alpha=0.05):
    """rank_corr(LB, V_true) and normalized argmax-LB regret over the one-hot
    policies on `arms_idx`. Pfit = the model's projection (None for full)."""
    F = Phi[arms_idx]                          # (pool, d); one-hot => f_pi = phi(arm)
    V = F @ w_T
    LB = np.array([post.lower_bound(f if Pfit is None else Pfit @ f, alpha) for f in F])
    rng = float(V.max() - V.min())
    if rng < 1e-9:
        return 1.0, 0.0
    reg = float((V.max() - V[int(np.argmax(LB))]) / rng)
    return rank_correlation(LB.tolist(), V.tolist()), reg


def run_arm2(seed, lam, arm):
    """pushN.run_arm trajectory (verbatim loop) + final-round N/S-pool damage."""
    ss = np.random.SeedSequence(seed)
    prng = np.random.default_rng(ss.spawn(1)[0])
    arng = np.random.default_rng(np.random.SeedSequence(seed + 987654).spawn(1)[0])
    arms = make_arms(N, D, prng)
    Phi, w0 = arms.Phi, arms.w0
    P = make_projection(D, K, np.random.default_rng(seed + 555))
    PiN = np.eye(D) - P.T @ P
    grader = make_grader("SAT", arms, BETAT, prng, lam=lam, T=T)
    n_pool = _npool(Phi, P)

    Pfit = P if arm == "closed_blind" else None
    post = RewardPosterior(K if arm == "closed_blind" else D)
    buf_d, buf_y = [], []
    m = np.zeros(D)
    frac = []
    dmg = {}
    for t in range(T):
        w_t = grader.w.copy()
        frac.append(float(np.linalg.norm(PiN @ w_t) / (np.linalg.norm(w_t) + 1e-12)))
        if buf_y:
            Z = Phi @ Pfit.T if Pfit is not None else Phi
            pi = policy_from_scores(rangenorm(Z @ post.posterior_mean()), BETA)
        else:
            pi = np.full(N, 1.0 / N)
        pairs = sample_pairs(pi, Q, arng)
        y = grader.label(pairs, arng)
        buf_d.append(Phi[pairs[:, 0]] - Phi[pairs[:, 1]]); buf_y.append(y)
        alld = np.concatenate(buf_d); ally = np.concatenate(buf_y).astype(float)
        if len(ally) > BUF_CAP:
            idx = arng.choice(len(ally), BUF_CAP, replace=False)
            alld, ally = alld[idx], ally[idx]
        dz = alld @ Pfit.T if Pfit is not None else alld
        post.fit(dz, ally, BETAT, prior_scale=PRIOR, n_samples=NS, rng=arng)

        if t == T - 1:
            rn, gn = _damage(n_pool, Phi, Pfit, w_t, post)
            dmg = dict(rank_N=rn, reg_N=gn)
        m = 0.5 * m + 0.5 * expected_features(pi, Phi)
        grader.step(m)

    return dict(seed=int(seed), lam=lam, arm=arm,
                dfrac=float(np.mean(frac[-LAST:]) - frac[0]), **dmg)


def _worker(args):
    (lam, arm), seed = args
    try:
        return run_arm2(seed, lam, arm), None
    except Exception as e:
        return None, f"lam={lam} arm={arm} seed={seed}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [((lam, arm), s) for lam in args.lams for arm in ARMS2
            for s in range(args.seeds)]
    print(f"{len(jobs)} runs ({len(args.lams)} lams x {len(ARMS2)} arms x "
          f"{args.seeds} seeds)", flush=True)
    rows, crashes = [], []
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        for r, err in pool.imap_unordered(_worker, jobs, chunksize=1):
            (rows.append(r) if r else crashes.append(err))
    for c in crashes:
        print("CRASH:", c)
    json.dump(rows, open(os.path.join(args.out, "pushN2.json"), "w"), indent=1)

    lines = ["# Paper B causal link: does the null-space push cause selection damage?\n",
             f"N-pool = {POOL} one-hot policies on the arms most extreme along the RM "
             "null space N; their true-value differences load on N. Damage at the final "
             "round vs w_T and the final posterior. If the push is CAUSAL, dfrac "
             "correlates NEGATIVELY with N-pool rank (POSITIVELY with N-pool regret) for "
             "the BLIND arm, and NOT for the FULL arm (which reads N) on the same pool.\n"]
    for lam in args.lams:
        lines.append(f"\n## lam={lam:g}\n")
        lines.append("| arm | mean rank_N | mean reg_N | Spearman(dfrac,rank_N) "
                     "| Spearman(dfrac,reg_N) | n |")
        lines.append("|---|---|---|---|---|---|")
        for arm in ARMS2:
            sel = [r for r in rows if r["lam"] == lam and r["arm"] == arm]
            df = [r["dfrac"] for r in sel]
            def sp(key):
                return rank_correlation(df, [r[key] for r in sel]) if len(sel) >= 5 else float("nan")
            lines.append(f"| {arm} | {np.mean([r['rank_N'] for r in sel]):+.2f} "
                         f"| {np.mean([r['reg_N'] for r in sel]):.2f} "
                         f"| {sp('rank_N'):+.2f} | {sp('reg_N'):+.2f} | {len(sel)} |")
    lines.append("\n## Verdict\n"
                 "- closed_blind: Spearman(dfrac,rank_N)<0 AND Spearman(dfrac,reg_N)>0 "
                 "-> the push CAUSES N-specific selection damage (Paper B consequence "
                 "established); closed_full showing no such dependence isolates it to the "
                 "blind representation, not drift magnitude per se.\n"
                 "- If the blind N-pool correlations are ~0, the push is real but its "
                 "harm is not demonstrated even on an N-aligned metric -> report the push "
                 "as a mechanism without a demonstrated selection consequence.\n")
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "results.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
