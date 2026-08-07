"""Two verifications of the representation-coverage mechanism (post-hoc on the drift runs).

(A) FROZEN PRE-DRIFT ENCODER: does a reward encoder trained on EARLY preferences and then
    frozen fail to rank policies under the LATE (drifted) grader -- even though the raw
    features phi still contain the needed info? For bottleneck width b:
      - enc_early = MLP encoder (hidden=b) fit on early-third comparisons, then FROZEN.
      - enc_fresh = same architecture fit on all comparisons (control: info IS recoverable).
      - Bayesian last layer (RewardPosterior over standardized b-dim features) fit on ALL
        comparisons for BOTH, ranked vs the late grader w_T.
      - cov_r2 = R^2 of predicting phi.u (the drift-direction value) from the frozen early
        features -> LOW = the drift direction is not retained in the frozen representation.
    Prediction: under drift, small-b frozen << fresh (freezing on pre-drift data loses the
    later-needed direction); the gap shrinks with capacity b and vanishes under NONE.

(B) DRIFT-TOWARD-BLIND-SPOTS geometry: variance ratio VR = u^T C u / (tr C / d) of the drift
    direction u vs the comparison-feature covariance C (the reward posterior is constrained
    ~ betaT^2 * C, so low-variance directions of C are the blind / high-posterior-variance
    ones). VR < 1 and a low percentile vs random directions => drift sits in blind directions.

Run: python -m analysis.stageF_repr_verify --root <root> --out <dir> --workers 16
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from drifted_grader.reward_models import MLPBT
from drifted_grader.posterior import RewardPosterior
from drifted_grader.metrics import rank_correlation
from analysis.stageF_ranking import _load, _pool, BETAT, ALPHA, PRIOR, NS
from analysis.stageF_scaleup import _parse_path
from analysis.stageF_hack import _box_corners, _regret

BS = (2, 3, 6)   # encoder bottleneck widths (input phi is 6-dim)


def _scale(rounds):
    a1 = np.concatenate([r["sphi1"] for r in rounds])
    a2 = np.concatenate([r["sphi2"] for r in rounds])
    return 1.0 / (np.sqrt(((a1 - a2) ** 2).sum(1).mean()) + 1e-8)


def _enc_fit(s1, s2, y, scale, b, rng):
    Phi = np.vstack([s1 * scale, s2 * scale]); n = len(y)
    pairs = np.stack([np.arange(n), np.arange(n) + n], axis=1)
    mlp = MLPBT(s1.shape[1], P=None, hidden=b, reg=1e-3, steps=150, seed=0)
    mlp.fit_buffer(Phi, pairs, np.asarray(y, float), rng)
    return mlp


def _z(mlp, X, scale):
    return np.tanh((np.asarray(X) * scale) @ mlp.W1.T + mlp.b1)


def _bll_eval(enc, s1, s2, y, feats, corners, wt, scale, rng):
    """Fit the Bayesian last layer on the frozen features, then report BOTH the natural-pool
    rank corr and the adversarial box-corner regret (does the bound deploy a truly-bad OOD
    corner it can't see through?)."""
    raw = lambda X: _z(enc, X, scale)
    H = np.vstack([raw(s1), raw(s2)]); mu = H.mean(0); sd = H.std(0) + 1e-8
    h = lambda X: (raw(X) - mu) / sd
    post = RewardPosterior(enc.h)
    post.fit(h(s1) - h(s2), np.asarray(y, float), BETAT, prior_scale=PRIOR, n_samples=NS, rng=rng)
    ev = np.vstack([feats, corners])
    lb = np.array([post.lower_bound(f, ALPHA) for f in h(ev)])
    lbf, lbc = lb[:len(feats)], lb[len(feats):]
    rank = rank_correlation(lbf.tolist(), (feats @ wt).tolist())
    vc = corners @ wt
    box_regret = _regret(vc[int(np.argmax(lbc))], vc)
    return rank, box_regret


def _r2(z, target):
    Z = np.hstack([z, np.ones((len(z), 1))])
    coef, *_ = np.linalg.lstsq(Z, target, rcond=None)
    pred = Z @ coef
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum() + 1e-12
    return float(1 - ss_res / ss_tot)


def analyze_frozen(path):
    rounds = _load(path)
    if len(rounds) < 6:
        return None
    feats = np.stack([r["m"] for r in rounds]); scale = _scale(rounds)
    w0 = rounds[0]["w"]; wT = rounds[-1]["w"]; u = wT - w0
    E = len(rounds) // 3
    s1e, s2e, ye = _pool(rounds, E - 1)                    # early-third comparisons
    s1a, s2a, ya = _pool(rounds, len(rounds) - 1)          # all comparisons
    if s1e is None or s1a is None or len(ye) < 20 or len(ya) < 20:
        return None
    Xseg = np.vstack([s1a, s2a]); tgt = Xseg @ u           # drift-direction value per segment
    corners = _box_corners(feats)                          # adversarial OOD candidate policies
    out = {}
    for b in BS:
        enc_early = _enc_fit(s1e, s2e, ye, scale, b, np.random.default_rng(1))
        enc_fresh = _enc_fit(s1a, s2a, ya, scale, b, np.random.default_rng(2))
        r_frozen, box_frozen = _bll_eval(enc_early, s1a, s2a, ya, feats, corners, wT, scale, np.random.default_rng(3))
        r_fresh, box_fresh = _bll_eval(enc_fresh, s1a, s2a, ya, feats, corners, wT, scale, np.random.default_rng(4))
        cov = _r2(_z(enc_early, Xseg, scale), tgt)
        out[b] = dict(frozen=r_frozen, fresh=r_fresh, box_frozen=box_frozen,
                      box_fresh=box_fresh, cov_r2=cov)
    return out


def analyze_geometry(path):
    rounds = _load(path)
    u = rounds[-1]["w"] - rounds[0]["w"]; nu = np.linalg.norm(u)
    if nu < 1e-9:
        return None                                        # NONE: no drift direction
    dph = np.concatenate([r["sphi1"] - r["sphi2"] for r in rounds])
    C = np.cov(dph.T)
    d = C.shape[0]; trd = np.trace(C) / d + 1e-12
    uu = u / nu
    vr = float(uu @ C @ uu / trd)
    rng = np.random.default_rng(0)
    R = rng.standard_normal((4000, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    vr_rand = np.einsum("ij,jk,ik->i", R, C, R) / trd
    return dict(vr=vr, pct_vs_random=float((vr_rand < vr).mean()))


def _worker(path):
    task, arm, seed = _parse_path(path)
    return dict(task=task, arm=arm, seed=seed,
                frozen=analyze_frozen(path), geom=analyze_geometry(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_repr_verify")
    os.makedirs(out_dir, exist_ok=True)
    runs = sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
    runs = [p for p in runs if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    print(f"found {len(runs)} runs")
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_worker, runs):
                results.append(r)
    else:
        results = [_worker(p) for p in runs]

    arms = ["NONE", "SAT_lam0.5", "SAT_lam1.0"]
    # ---- (A) frozen-encoder aggregation ----
    fa = defaultdict(lambda: defaultdict(list))   # (arm,b)->metric->[vals]
    rows = []
    for r in results:
        if not r["frozen"]:
            continue
        for b in BS:
            for m in ("frozen", "fresh", "box_frozen", "box_fresh", "cov_r2"):
                fa[(r["arm"], b)][m].append(r["frozen"][b][m])
            rows.append(dict(task=r["task"], arm=r["arm"], seed=r["seed"], bottleneck=b,
                             **{k: r["frozen"][b][k] for k in
                                ("frozen", "fresh", "box_frozen", "box_fresh", "cov_r2")}))
    with open(os.path.join(out_dir, "frozen_per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "arm", "seed", "bottleneck", "frozen",
                           "fresh", "box_frozen", "box_fresh", "cov_r2"])
        w.writeheader(); w.writerows(rows)

    L = ["# Representation-coverage verifications\n",
         "\n## (A) Frozen pre-drift encoder — rank corr under the LATE grader w_T\n",
         "`frozen` = encoder trained on EARLY prefs then frozen; `fresh` = same width trained "
         "on all prefs; `cov_r2` = R^2 recovering the drift-direction value phi.u from the "
         "frozen early features (LOW = drift dir not retained). Mean over tasks/seeds.\n",
         "| arm | b | frozen rank | fresh rank | frozen−fresh | cov_r2(u|z_early) |",
         "|---|---|---|---|---|---|"]
    for arm in arms:
        for b in BS:
            fr = fa[(arm, b)]["frozen"]; fe = fa[(arm, b)]["fresh"]; cr = fa[(arm, b)]["cov_r2"]
            if not fr:
                continue
            L.append(f"| {arm} | {b} | {np.mean(fr):+.2f} | {np.mean(fe):+.2f} | "
                     f"{np.mean(fr) - np.mean(fe):+.2f} | {np.mean(cr):.2f} |")

    L.append("\n## (A2) Frozen pre-drift encoder on ADVERSARIAL box-corner policies\n")
    L.append("box_regret = normalized true-value regret of the corner the bound deploys "
             "(argmax-VaR over box corners; higher = the bound is hacked). Does emergent "
             "omission bite once candidates EXERCISE the dropped direction?\n")
    L.append("| arm | b | box_regret frozen | box_regret fresh | frozen−fresh |")
    L.append("|---|---|---|---|---|")
    for arm in arms:
        for b in BS:
            bf = fa[(arm, b)]["box_frozen"]; bh = fa[(arm, b)]["box_fresh"]
            if not bf:
                continue
            L.append(f"| {arm} | {b} | {np.mean(bf):.2f} | {np.mean(bh):.2f} | "
                     f"{np.mean(bf) - np.mean(bh):+.2f} |")

    # ---- (B) blind-spot geometry ----
    ga = defaultdict(list)
    for r in results:
        if r["geom"]:
            ga[r["arm"]].append((r["geom"]["vr"], r["geom"]["pct_vs_random"]))
    L.append("\n## (B) Drift toward blind spots — variance ratio of u vs comparison covariance C\n")
    L.append("VR = u^T C u / (tr C / d).  VR<1 and pct_vs_random<0.5 => drift lies in "
             "below-average-variance (blind / high-posterior-variance) directions.\n")
    L.append("| arm | mean VR | mean pct_vs_random | n |")
    L.append("|---|---|---|---|")
    for arm in arms:
        if ga[arm]:
            v = np.array(ga[arm]); L.append(f"| {arm} | {v[:,0].mean():.2f} | {v[:,1].mean():.2f} | {len(v)} |")

    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
