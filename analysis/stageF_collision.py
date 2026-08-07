"""Collision-vs-OOD verification (does imposed adv HIDE the bad corner while full flags it?).

For the archetypal drift-hack u-corner (loads the satiated dims -> bad under w_T), we measure
each bound's UNCERTAINTY = posterior-mean value - alpha-VaR. If the drift-blind `adv` sees the
u-corner as in-distribution it assigns LOW uncertainty (ratio vs natural pool ~1 -> collision,
can't flag); a full-representation bound assigns HIGH uncertainty (ratio >>1 -> OOD defense).
Also report the u-corner's true-value percentile (LOW = it really is a bad policy).

Run: python -m analysis.stageF_collision --root <root> --out <dir> --workers 16
"""
from __future__ import annotations
import argparse, glob, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from drifted_grader.reward_models import make_projection, _apply_P, MLPBT
from drifted_grader.posterior import RewardPosterior
from analysis.stageF_ranking import _load, _pool, _adv_projection, BETAT, ALPHA, PRIOR, NS
from analysis.stageF_scaleup import _parse_path

CONDS = ("full", "adv", "mlp_full", "mlp_adv")


def _scale(rounds):
    a1 = np.concatenate([r["sphi1"] for r in rounds]); a2 = np.concatenate([r["sphi2"] for r in rounds])
    return 1.0 / (np.sqrt(((a1 - a2) ** 2).sum(1).mean()) + 1e-8)


def _lin_meanunc(dphi, y, X, scale, P, rng):
    Xp = dphi if P is None else dphi @ P.T
    Fp = X if P is None else X @ P.T
    post = RewardPosterior(Xp.shape[1])
    post.fit(Xp * scale, y, BETAT, prior_scale=PRIOR, n_samples=NS, rng=rng)
    pm = post.posterior_mean()
    mean = np.array([pm @ (f * scale) for f in Fp])
    lb = np.array([post.lower_bound(f * scale, ALPHA) for f in Fp])
    return mean - lb                                   # uncertainty per eval point


def _mlp_meanunc(s1, s2, y, X, scale, P, rng, hidden=6):
    n = len(y); Phi = np.vstack([s1 * scale, s2 * scale])
    pairs = np.stack([np.arange(n), np.arange(n) + n], axis=1)
    mlp = MLPBT(s1.shape[1], P=P, hidden=hidden, reg=1e-3, steps=150, seed=0)
    mlp.fit_buffer(Phi, pairs, np.asarray(y, float), rng)
    raw = lambda Z: np.tanh(_apply_P(np.asarray(Z) * scale, P) @ mlp.W1.T + mlp.b1)
    H = np.vstack([raw(s1), raw(s2)]); mu = H.mean(0); sd = H.std(0) + 1e-8
    h = lambda Z: (raw(Z) - mu) / sd
    post = RewardPosterior(hidden)
    post.fit(h(s1) - h(s2), np.asarray(y, float), BETAT, prior_scale=PRIOR, n_samples=NS, rng=rng)
    pm = post.posterior_mean(); Fh = h(X)
    mean = np.array([pm @ f for f in Fh]); lb = np.array([post.lower_bound(f, ALPHA) for f in Fh])
    return mean - lb


def analyze_run(path, k=3, stages=10):
    rounds = _load(path)
    if len(rounds) < 4:
        return None
    feats = np.stack([r["m"] for r in rounds]); scale = _scale(rounds)
    u = rounds[-1]["w"] - rounds[0]["w"]; nu = np.linalg.norm(u)
    if nu < 1e-9:
        return None                                     # NONE: no u-corner
    rng = np.random.default_rng(0)
    P_adv = _adv_projection(feats.shape[1], k, u, rng)
    idx = np.unique(np.linspace(len(rounds) // 3, len(rounds) - 1, stages).astype(int))
    out = {c: {"ratio": [], "true_pct": []} for c in CONDS}
    for t in idx:
        s1, s2, y = _pool(rounds, t)
        if s1 is None or len(y) < 20:
            continue
        dphi = s1 - s2; wt = rounds[t]["w"]; pool = feats[: t + 1]
        lo = pool.min(0); hi = pool.max(0)
        ucorner = np.where(u < 0, hi, lo)               # bad-under-w_T stale-loading corner
        ev = np.vstack([pool, ucorner]); ci = len(ev) - 1
        vt = ev @ wt
        rng2 = np.random.default_rng(int(t) + 1)
        specs = (("full", "lin", None), ("adv", "lin", P_adv),
                 ("mlp_full", "mlp", None), ("mlp_adv", "mlp", P_adv))
        for name, kind, P in specs:
            unc = (_lin_meanunc(dphi, y, ev, scale, P, rng2) if kind == "lin"
                   else _mlp_meanunc(s1, s2, y, ev, scale, P, rng2))
            pool_unc = np.mean(unc[:-1]) + 1e-12
            out[name]["ratio"].append(float(unc[ci] / pool_unc))     # >1 flagged, ~1 collision
            out[name]["true_pct"].append(float((vt < vt[ci]).mean()))
    return out


def _worker(p):
    task, arm, seed = _parse_path(p)
    return dict(arm=arm, res=analyze_run(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_collision"); os.makedirs(out_dir, exist_ok=True)
    runs = [p for p in sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
            if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    print(f"found {len(runs)} runs")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, runs):
            results.append(r)
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r["res"]:
            continue
        for c in CONDS:
            if r["res"][c]["ratio"]:
                agg[(r["arm"], c)]["ratio"].append(np.mean(r["res"][c]["ratio"]))
                agg[(r["arm"], c)]["true_pct"].append(np.mean(r["res"][c]["true_pct"]))
    arms = ["SAT_lam0.5", "SAT_lam1.0"]
    L = ["# Collision-vs-OOD verification\n",
         "uncertainty ratio = (mean−VaR at the u-corner) / (mean−VaR over the natural pool). "
         "~1 => the bound treats the bad u-corner as in-distribution (COLLISION, can't flag); "
         ">>1 => flagged as OOD (uncertainty defense active). true_pct = u-corner's true-value "
         "percentile (LOW = genuinely bad).\n",
         "| arm | condition | uncertainty ratio | u-corner true_pct |", "|---|---|---|---|"]
    for arm in arms:
        for c in CONDS:
            a = agg[(arm, c)]
            if a["ratio"]:
                L.append(f"| {arm} | {c} | {np.mean(a['ratio']):.2f} | {np.mean(a['true_pct']):.2f} |")
    open(os.path.join(out_dir, "results.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
