"""Confirm the hammer mechanism in the controlled simulator. The neural forensics
said the CORRECTLY-SPECIFIED model degrades where the eval pool is geometrically
WIDE (high effective rank), and argued the cause is finite-sample posterior error
amplified by narrow (on-policy) comparison data. Direct causal test:

  full (k=d) posterior, SAT drift, cross
    collection breadth in {onpolicy, broad}
    eval pool      in {narrow (low effrank), wide (high effrank)}
  measure rank_corr(LB, true value) over each pool.

Prediction if the mechanism is right:
  wide pool  : onpolicy rank LOW, broad rank HIGH   (broad collection RESCUES it)
  narrow pool: rank HIGH under both breadths
  i.e. an interaction: degradation needs BOTH a wide eval pool AND narrow data.
If instead broad does not rescue the wide pool, the cause is not data breadth
(reject the mechanism; the wide-pool degradation is intrinsic).

narrow pool = policies defined by the initial grader w0 (as in run_p1): low
              dispersion, differ mostly along one direction.
wide pool   = one-hot policies on the per-dimension argmax/argmin arms: maximal
              dispersion, differ along all d directions.

Run: python -m analysis.hammer_mechanism --out <dir> --seeds 40 --workers 16
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

from analysis.run_p1 import N, D, BETA, BETAT, Q, PRIOR, NS, BUF_CAP, _eval_pols, _uniform_pairs

ALPHA = 0.05
LAST = 10


def _effrank(F):
    Fc = F - F.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(np.cov(Fc.T)), 0, None)
    s = ev.sum()
    return float(s * s / ((ev ** 2).sum() + 1e-18)) if s > 1e-18 else 0.0


def _wide_pool_feats(arms):
    """One-hot policies on the per-dimension extreme arms -> features span all d
    directions (high effective rank)."""
    Phi = arms.Phi
    idx = []
    for j in range(D):
        idx.append(int(Phi[:, j].argmax()))
        idx.append(int(Phi[:, j].argmin()))
    idx = sorted(set(idx))
    return Phi[idx]                      # one-hot => expected features = arm features


def run_cell(seed, breadth, lam, T=100):
    ss = np.random.SeedSequence(seed)
    prng = np.random.default_rng(ss.spawn(1)[0])
    arng = np.random.default_rng(np.random.SeedSequence(seed + 987654).spawn(1)[0])
    arms = make_arms(N, D, prng)
    Phi, w0 = arms.Phi, arms.w0
    P = make_projection(D, D, prng)                 # k=d -> identity (correctly specified)
    grader = make_grader("SAT" if lam > 0 else "NONE", arms, BETAT, prng, lam=lam, T=T)

    narrow_f = np.stack([expected_features(pe, Phi) for pe in _eval_pols(arms)])
    wide_f = _wide_pool_feats(arms)
    eff_narrow, eff_wide = _effrank(narrow_f), _effrank(wide_f)

    buf_d, buf_y = [], []
    post = RewardPosterior(D)
    m = np.zeros(D)
    rk_narrow, rk_wide = [], []
    for t in range(T):
        w_t = grader.w.copy()
        if buf_y:
            pi = policy_from_scores(rangenorm(Phi @ post.posterior_mean()), BETA)
        else:
            pi = np.full(N, 1.0 / N)
        pairs = sample_pairs(pi, Q, arng) if breadth == "onpolicy" else _uniform_pairs(arng)
        y = grader.label(pairs, arng)
        buf_d.append(Phi[pairs[:, 0]] - Phi[pairs[:, 1]]); buf_y.append(y)
        alld = np.concatenate(buf_d); ally = np.concatenate(buf_y).astype(float)
        if len(ally) > BUF_CAP:
            idx = arng.choice(len(ally), BUF_CAP, replace=False)
            alld, ally = alld[idx], ally[idx]
        post.fit(alld, ally, BETAT, prior_scale=PRIOR, n_samples=NS, rng=arng)

        if t >= T - LAST:
            for feats, store in ((narrow_f, rk_narrow), (wide_f, rk_wide)):
                LB = [post.lower_bound(f, ALPHA) for f in feats]
                V = (feats @ w_t).tolist()
                store.append(rank_correlation(LB, V))
        m = 0.5 * m + 0.5 * expected_features(pi, Phi)
        grader.step(m)

    return dict(seed=int(seed), breadth=breadth, lam=lam,
                rank_narrow=float(np.mean(rk_narrow)),
                rank_wide=float(np.mean(rk_wide)),
                eff_narrow=eff_narrow, eff_wide=eff_wide)


def _worker(args):
    (breadth, lam), seed = args
    try:
        return run_cell(seed, breadth, lam), None
    except Exception as e:
        return None, f"{breadth} lam={lam} seed={seed}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.1])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cells = [(b, lam) for b in ("onpolicy", "broad") for lam in args.lams]
    jobs = [(c, s) for c in cells for s in range(args.seeds)]
    print(f"{len(jobs)} runs", flush=True)
    rows, crashes = [], []
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        for r, err in pool.imap_unordered(_worker, jobs, chunksize=1):
            (rows.append(r) if r else crashes.append(err))
    for c in crashes:
        print("CRASH:", c)
    json.dump(rows, open(os.path.join(args.out, "hammer_mech.json"), "w"), indent=1)

    rng = np.random.default_rng(0)
    def ci(v):
        v = np.asarray(v, float)
        if len(v) < 2:
            return (float(np.mean(v)) if len(v) else np.nan, np.nan, np.nan)
        bs = np.array([np.mean(rng.choice(v, len(v))) for _ in range(2000)])
        return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    eN = np.mean([r["eff_narrow"] for r in rows])
    eW = np.mean([r["eff_wide"] for r in rows])
    lines = ["# Hammer mechanism (controlled sim): does broad collection rescue the "
             "correct model's ranking on a WIDE eval pool?\n",
             f"full (k=d) posterior; eval-pool effective rank: narrow={eN:.2f}, "
             f"wide={eW:.2f} (of d={D}); {args.seeds} seeds/cell; mean [95% CI].\n",
             "| breadth | lam | rank (narrow pool) | rank (WIDE pool) |",
             "|---|---|---|---|"]
    for b in ("onpolicy", "broad"):
        for lam in args.lams:
            sel = [r for r in rows if r["breadth"] == b and r["lam"] == lam]
            mn, ln, hn = ci([r["rank_narrow"] for r in sel])
            mw, lw, hw = ci([r["rank_wide"] for r in sel])
            lines.append(f"| {b} | {lam:g} | {mn:+.2f} [{ln:+.2f},{hn:+.2f}] "
                         f"| {mw:+.2f} [{lw:+.2f},{hw:+.2f}] |")
    # the decisive interaction: wide-pool rank, broad minus onpolicy, at max drift
    lam_hi = max(args.lams)
    onp = [r["rank_wide"] for r in rows if r["breadth"] == "onpolicy" and r["lam"] == lam_hi]
    brd = [r["rank_wide"] for r in rows if r["breadth"] == "broad" and r["lam"] == lam_hi]
    d = float(np.mean(brd) - np.mean(onp))
    bs = np.array([np.mean(rng.choice(brd, len(brd))) - np.mean(rng.choice(onp, len(onp)))
                   for _ in range(4000)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    lines.append(f"\n**Rescue effect (wide pool, lam={lam_hi:g}): broad - onpolicy = "
                 f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]** "
                 f"{'-> broad RESCUES (mechanism confirmed)' if lo > 0 else '-> no rescue (mechanism rejected)'}\n")
    lines.append("Reading: mechanism CONFIRMED iff the wide-pool rank is low under "
                 "onpolicy and materially higher under broad (rescue CI > 0), while the "
                 "narrow pool stays high under both. That isolates 'wide eval pool x "
                 "narrow data' as the cause of the correct-model degradation (the "
                 "neural hammer being the extreme-dispersion case).\n")
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "results.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
