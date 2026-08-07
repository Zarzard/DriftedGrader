"""Frozen-encoder ablation: does the OOD defense depend on tanh saturation, and does a
genuinely-collapsed (tiny-bottleneck) frozen encoder get hacked?

Sweep bottleneck b in {1,2,3,6} x activation {tanh, relu}. Train a 1-hidden-layer BT reward
encoder on EARLY-third prefs, freeze it, Bayesian-last-layer on all prefs, then measure
box_regret on the adversarial box-corner policies (hacked if high) and cov_r2(phi.u | z_early).
Prediction: relu and/or tiny b (genuine collision, no saturation) -> HACKED, unlike tanh/large b.

Run: python -m analysis.stageF_frozen_ablate --root <root> --out <dir> --workers 16
"""
from __future__ import annotations
import argparse, glob, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import expit
from drifted_grader.posterior import RewardPosterior
from analysis.stageF_ranking import _load, _pool, BETAT, ALPHA, PRIOR, NS
from analysis.stageF_scaleup import _parse_path
from analysis.stageF_hack import _box_corners, _regret

BS = (1, 2, 3, 6)
ACTS = ("tanh", "relu")


def _act(x, k):
    return np.tanh(x) if k == "tanh" else np.maximum(x, 0.0)


def _dact(z, k):
    return (1.0 - np.tanh(z) ** 2) if k == "tanh" else (z > 0).astype(float)


def _scale(rounds):
    a1 = np.concatenate([r["sphi1"] for r in rounds]); a2 = np.concatenate([r["sphi2"] for r in rounds])
    return 1.0 / (np.sqrt(((a1 - a2) ** 2).sum(1).mean()) + 1e-8)


def _enc_fit(X1, X2, y, b, kind, steps=200, lr=5e-2, reg=1e-3, seed=0):
    """1-hidden BT reward r=w2.act(W1 x+b1); return (W1,b1). X1,X2 already scaled."""
    d = X1.shape[1]; rng = np.random.default_rng(seed); n = len(y)
    W1 = rng.standard_normal((b, d)) * np.sqrt(1.0 / d); b1 = np.zeros(b)
    w2 = rng.standard_normal(b) * np.sqrt(1.0 / b)
    params = [W1, b1, w2]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    b1_, b2_ = 0.9, 0.999
    for t in range(1, steps + 1):
        Z1 = X1 @ W1.T + b1; H1 = _act(Z1, kind); Z2 = X2 @ W1.T + b1; H2 = _act(Z2, kind)
        r = H1 @ w2 - H2 @ w2
        g = expit(r) - y
        gw2 = (H1 - H2).T @ g / n + reg * w2
        gZ1 = (g[:, None] * w2) * _dact(Z1, kind); gZ2 = (-g[:, None] * w2) * _dact(Z2, kind)
        gW1 = (gZ1.T @ X1 + gZ2.T @ X2) / n + reg * W1
        gb1 = (gZ1 + gZ2).mean(0)
        grads = [gW1, gb1, gw2]
        for i, gg in enumerate(grads):
            m[i] = b1_ * m[i] + (1 - b1_) * gg
            v[i] = b2_ * v[i] + (1 - b2_) * gg ** 2
            params[i] -= lr * (m[i] / (1 - b1_ ** t)) / (np.sqrt(v[i] / (1 - b2_ ** t)) + 1e-8)
    return params[0], params[1]


def _bll_boxregret(W1, b1, kind, s1, s2, y, feats, corners, wt, scale, rng):
    raw = lambda X: _act((np.asarray(X) * scale) @ W1.T + b1, kind)
    H = np.vstack([raw(s1), raw(s2)]); mu = H.mean(0); sd = H.std(0) + 1e-8
    h = lambda X: (raw(X) - mu) / sd
    post = RewardPosterior(W1.shape[0])
    post.fit(h(s1) - h(s2), np.asarray(y, float), BETAT, prior_scale=PRIOR, n_samples=NS, rng=rng)
    lbc = np.array([post.lower_bound(f, ALPHA) for f in h(corners)])
    vc = corners @ wt
    return _regret(vc[int(np.argmax(lbc))], vc)


def _r2(z, tgt):
    Z = np.hstack([z, np.ones((len(z), 1))]); coef, *_ = np.linalg.lstsq(Z, tgt, rcond=None)
    p = Z @ coef; return float(1 - ((tgt - p) ** 2).sum() / (((tgt - tgt.mean()) ** 2).sum() + 1e-12))


def analyze_run(path):
    rounds = _load(path)
    if len(rounds) < 6:
        return None
    feats = np.stack([r["m"] for r in rounds]); scale = _scale(rounds)
    u = rounds[-1]["w"] - rounds[0]["w"]; wT = rounds[-1]["w"]
    E = len(rounds) // 3
    s1e, s2e, ye = _pool(rounds, E - 1); s1a, s2a, ya = _pool(rounds, len(rounds) - 1)
    if s1e is None or s1a is None or len(ye) < 20 or len(ya) < 20:
        return None
    corners = _box_corners(feats); Xseg = np.vstack([s1a, s2a]); tgt = Xseg @ u
    out = {}
    for kind in ACTS:
        for b in BS:
            W1e, b1e = _enc_fit(s1e * scale, s2e * scale, ye, b, kind, seed=1)
            W1f, b1f = _enc_fit(s1a * scale, s2a * scale, ya, b, kind, seed=2)
            bz = _bll_boxregret(W1e, b1e, kind, s1a, s2a, ya, feats, corners, wT, scale, np.random.default_rng(3))
            bh = _bll_boxregret(W1f, b1f, kind, s1a, s2a, ya, feats, corners, wT, scale, np.random.default_rng(4))
            cov = _r2(_act((Xseg * scale) @ W1e.T + b1e, kind), tgt)
            out[(kind, b)] = dict(box_frozen=bz, box_fresh=bh, cov_r2=cov)
    return out


def _worker(p):
    task, arm, seed = _parse_path(p)
    return dict(arm=arm, res=analyze_run(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_frozen_ablate"); os.makedirs(out_dir, exist_ok=True)
    runs = [p for p in sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
            if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    print(f"found {len(runs)} runs")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, runs):
            results.append(r)
    arms = ["SAT_lam0.5", "SAT_lam1.0"]
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r["res"]:
            continue
        for key, d in r["res"].items():
            for m in ("box_frozen", "box_fresh", "cov_r2"):
                agg[(r["arm"],) + key][m].append(d[m])
    L = ["# Frozen-encoder ablation (activation x bottleneck) on box-corner hacking\n",
         "box_regret frozen = does the frozen pre-drift encoder deploy a bad corner (hacked)? "
         "Compare tanh vs relu and small vs large bottleneck b. cov_r2 = drift-dir retention.\n",
         "| arm | act | b | box_regret frozen | box_regret fresh | cov_r2 |", "|---|---|---|---|---|---|"]
    for arm in arms:
        for kind in ACTS:
            for b in BS:
                a = agg[(arm, kind, b)]
                if a["box_frozen"]:
                    L.append(f"| {arm} | {kind} | {b} | {np.mean(a['box_frozen']):.2f} | "
                             f"{np.mean(a['box_fresh']):.2f} | {np.mean(a['cov_r2']):.2f} |")
    open(os.path.join(out_dir, "results.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
