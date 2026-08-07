"""Held-out certification: does the VaR bound's single-policy guarantee hold for
policies NEVER deployed and absent from the comparison distribution?

The paper's "certification is robust to drift" (Section 4.4) was measured on the
DEPLOYED policy only -- a value the collected comparisons surround. The reviewer's
falsifier: certify held-out policies OFF the comparison support. If held-out
certification degrades wherever ranking degrades, the certification/ranking split
is ordinary on-support/off-support covariate shift and the framing must change.

Protocol = run_p1.run_coverage verbatim (same constants, on-policy collection,
posterior refit each round, SAT drift), plus a held-out battery certified each
round: hit = 1[ w_t . f_e >= LB(P f_e, alpha=0.05) ].

Battery (f_e = expected features):
  deployed      the deployed policy (reproduces the paper's metric; ON-support)
  uniform       uniform over arms (mild off-support)
  stale_best    one-hot argmax(Phi w0)   -- good under the OLD grader
  stale_sharp   Gibbs(Phi w0, beta=0.05) -- concentrated stale mixture
  current_best  one-hot argmax(Phi w_t)  -- good NOW, never deployed
  worst_current one-hot argmin(Phi w_t)  -- genuinely bad now
  anti_drift    one-hot argmin(Phi u_t), u_t = w_t - w0 -- the trap-flavored
                policy: worst specifically along the drift direction
  random_arm    one-hot on a fixed random arm

Cells: k in {6, 3} x lam in {0.0, 0.05, 0.1} (SAT; 0 = NONE), on-policy breadth.
Also records rank_corr on run_p1's eval set so the ranking collapse is verified
in the SAME runs the certification numbers come from.

Run: python -m analysis.heldout_cert --out <dir> --seeds 20 --workers 16
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader
from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import policy_from_scores, rangenorm, regret_Rt, rank_correlation
from drifted_grader.policies import expected_features, sample_pairs
from drifted_grader.posterior import RewardPosterior

ALPHA = 0.05
N, D, BETA, BETAT, Q, PRIOR = 256, 6, 0.05, 10.0, 64, 2.0
NS, BUF_CAP = 4000, 1800
LAST = 10          # rounds at the end over which certification is scored

BATTERY = ("deployed", "uniform", "stale_best", "stale_sharp", "current_best",
           "worst_current", "anti_drift", "random_arm")


def _eval_pols(arms):
    """run_p1's ranking eval set (kept identical for the rank_corr continuity check)."""
    Phi, w0 = arms.Phi, arms.w0
    pols = [policy_from_scores(Phi @ w0, b) for b in (2.0, 0.2, 0.05)]
    pols.append(np.full(N, 1.0 / N))
    o = np.zeros(N); o[(Phi @ w0).argmax()] = 1.0; pols.append(o)
    return pols


def _battery_feats(arms, w_t, pi, rand_arm):
    """Expected-feature vector for each battery policy at the current round."""
    Phi, w0 = arms.Phi, arms.w0
    u = w_t - w0
    def onehot(i):
        p = np.zeros(N); p[i] = 1.0; return p
    pols = {
        "deployed": pi,
        "uniform": np.full(N, 1.0 / N),
        "stale_best": onehot(int((Phi @ w0).argmax())),
        "stale_sharp": policy_from_scores(Phi @ w0, 0.05),
        "current_best": onehot(int((Phi @ w_t).argmax())),
        "worst_current": onehot(int((Phi @ w_t).argmin())),
        "anti_drift": (onehot(int((Phi @ u).argmin()))
                       if np.linalg.norm(u) > 1e-6 else None),
        "random_arm": onehot(rand_arm),
    }
    return {k: (expected_features(p, Phi) if p is not None else None)
            for k, p in pols.items()}


def _uniform_pairs(rng):
    i = rng.integers(0, N, Q); j = rng.integers(0, N, Q); mk = i != j
    return np.stack([i[mk], j[mk]], 1)


def run_cell(seed, lam, k, T=100):
    drift = "SAT" if lam > 0 else "NONE"
    ss = np.random.SeedSequence(seed)
    prng = np.random.default_rng(ss.spawn(1)[0])
    arng = np.random.default_rng(np.random.SeedSequence(seed + 987654).spawn(1)[0])
    arms = make_arms(N, D, prng)
    Phi, w0 = arms.Phi, arms.w0
    P = make_projection(D, k, prng)
    grader = make_grader(drift, arms, BETAT, prng, lam=lam, T=T)
    evalp = _eval_pols(arms)
    evalf = [expected_features(pe, Phi) for pe in evalp]
    rand_arm = int(np.random.default_rng(seed + 31337).integers(0, N))

    buf_d, buf_y = [], []
    post = RewardPosterior(P.shape[0])
    m = np.zeros(D)
    recs = []
    for t in range(T):
        w_t = grader.w.copy()
        if buf_y:
            rmean = (Phi @ P.T) @ post.posterior_mean()
            pi = policy_from_scores(rangenorm(rmean), BETA)
        else:
            pi = np.full(N, 1.0 / N)

        pairs = sample_pairs(pi, Q, arng)
        y = grader.label(pairs, arng)
        buf_d.append(Phi[pairs[:, 0]] - Phi[pairs[:, 1]]); buf_y.append(y)
        alld = np.concatenate(buf_d); ally = np.concatenate(buf_y).astype(float)
        if len(ally) > BUF_CAP:
            idx = arng.choice(len(ally), BUF_CAP, replace=False)
            alld, ally = alld[idx], ally[idx]
        post.fit(alld @ P.T, ally, BETAT, prior_scale=PRIOR, n_samples=NS, rng=arng)

        if t >= T - LAST:
            range_t = float((Phi @ w_t).ptp())
            feats_b = _battery_feats(arms, w_t, pi, rand_arm)
            rec = dict(t=t)
            for name, fe in feats_b.items():
                if fe is None:
                    continue
                v = float(w_t @ fe)
                lb = post.lower_bound(P @ fe, ALPHA)
                rec[f"hit_{name}"] = int(v >= lb)
                rec[f"mar_{name}"] = float((v - lb) / (range_t + 1e-12))
            lb_eval = [post.lower_bound(P @ fe, ALPHA) for fe in evalf]
            v_eval = [float(w_t @ fe) for fe in evalf]
            rec["rank_corr"] = rank_correlation(lb_eval, v_eval)
            rec["R_t"], _ = regret_Rt(pi, Phi, w_t, BETA)
            recs.append(rec)

        m = 0.5 * m + 0.5 * expected_features(pi, Phi)
        grader.step(m)

    out = dict(seed=int(seed), lam=lam, k=k)
    for name in BATTERY:
        hits = [r[f"hit_{name}"] for r in recs if f"hit_{name}" in r]
        mars = [r[f"mar_{name}"] for r in recs if f"mar_{name}" in r]
        out[f"cov_{name}"] = float(np.mean(hits)) if hits else np.nan
        out[f"mar_{name}"] = float(np.mean(mars)) if mars else np.nan
    out["rank_corr"] = float(np.mean([r["rank_corr"] for r in recs]))
    out["R_t"] = float(np.mean([r["R_t"] for r in recs]))
    return out


def _worker(args):
    (lam, k), seed = args
    try:
        return run_cell(seed, lam, k), None
    except Exception as e:
        return None, f"lam={lam} k={k} seed={seed}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    ap.add_argument("--ks", type=int, nargs="+", default=[6, 3])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cells = [(lam, k) for k in args.ks for lam in args.lams]
    jobs = [(c, s) for c in cells for s in range(args.seeds)]
    print(f"{len(cells)} cells x {args.seeds} seeds = {len(jobs)} runs", flush=True)

    rows, crashes = [], []
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        for r, err in pool.imap_unordered(_worker, jobs, chunksize=1):
            (rows.append(r) if r else crashes.append(err))
    for c in crashes:
        print("CRASH:", c)

    json.dump(rows, open(os.path.join(args.out, "heldout.json"), "w"), indent=1)

    # ---- aggregate: coverage per (k, lam, policy) mean +/- bootstrap CI over seeds ----
    rng = np.random.default_rng(0)
    def ci(vals):
        v = np.asarray([x for x in vals if not np.isnan(x)], float)
        if len(v) < 2:
            return (float(np.mean(v)) if len(v) else np.nan, np.nan, np.nan)
        bs = np.array([np.mean(rng.choice(v, len(v))) for _ in range(2000)])
        return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    lines = ["# Held-out certification under drift\n",
             f"alpha={ALPHA}; coverage scored over the last {LAST} rounds, "
             f"{args.seeds} seeds/cell; mean [95% bootstrap CI over seeds]. "
             "Target >= 0.95. `deployed` reproduces the paper's marginal-coverage "
             "metric; every other row is a held-out policy the loop never deployed.\n"]
    for k in args.ks:
        lines.append(f"\n## k={k} ({'well-specified' if k >= D else 'blind (k<d)'})\n")
        header = "| policy | " + " | ".join(f"lam={l:g}" for l in args.lams) + " |"
        lines.append(header); lines.append("|" + "---|" * (1 + len(args.lams)))
        for name in BATTERY:
            cells_txt = []
            for lam in args.lams:
                sel = [r for r in rows if r["k"] == k and r["lam"] == lam]
                mcv, lo, hi = ci([r[f"cov_{name}"] for r in sel])
                cells_txt.append(f"{mcv:.2f} [{lo:.2f},{hi:.2f}]" if not np.isnan(mcv) else "n/a")
            lines.append(f"| {name} | " + " | ".join(cells_txt) + " |")
        # continuity: ranking + regret in the same runs
        rk, rg = [], []
        for lam in args.lams:
            sel = [r for r in rows if r["k"] == k and r["lam"] == lam]
            rk.append(f"{np.mean([r['rank_corr'] for r in sel]):+.2f}")
            rg.append(f"{np.mean([r['R_t'] for r in sel]):.2f}")
        lines.append(f"| *(rank_corr)* | " + " | ".join(rk) + " |")
        lines.append(f"| *(regret)* | " + " | ".join(rg) + " |")

    lines.append(
        "\n## Reading the verdict\n"
        "- ALL rows >= ~0.95 in both k blocks -> certification genuinely robust "
        "off-support; the certification/ranking dissociation survives as stated.\n"
        "- `deployed` ~1.0 but held-out rows (esp. anti_drift / stale_best / "
        "current_best) collapse in the k<d blocks where rank_corr collapses -> "
        "certification robustness is DEPLOYMENT-PROXIMITY; the split is "
        "on-support vs off-support (covariate shift), and the paper's framing "
        "must be revised accordingly.\n"
        "- held-out collapse ONLY under drift (lam>0) but not lam=0 -> drift-"
        "specific off-support failure: intermediate framing.\n")
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "results.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
