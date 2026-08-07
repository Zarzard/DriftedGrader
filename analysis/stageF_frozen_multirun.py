"""Frozen pre-drift encoder on the MULTI-RUN pool — the direct test of "natural feature drop
hurts realistic (diverse) selection".

For each (task, arm) and each reference run: train a reward encoder on that run's EARLY-third
prefs (FROZEN) vs ALL prefs (fresh), fit the Bayesian last layer on all prefs, and rank BOTH the
run's OWN pool (single-run) and the all-seed pool (multi-run) under the run's w_T. If frozen ≪
fresh on the multi-run pool (but ≈ on single-run), natural early-freezing feature drop bites once
the candidates are diverse along u. `full` linear = well-spec anchor.

Run: python -m analysis.stageF_frozen_multirun --root <root> --out <dir> --workers 16
"""
from __future__ import annotations
import argparse, glob, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from drifted_grader.posterior import RewardPosterior
from drifted_grader.metrics import rank_correlation
from analysis.stageF_ranking import _load, _pool, BETAT, ALPHA, PRIOR, NS
from analysis.stageF_repr_verify import _enc_fit, _z            # MLPBT (tanh) encoder + feature map
from analysis.stageF_scaleup import _fit_linear

BS = (2, 6)


def _scale(rounds):
    a1 = np.concatenate([r["sphi1"] for r in rounds]); a2 = np.concatenate([r["sphi2"] for r in rounds])
    return 1.0 / (np.sqrt(((a1 - a2) ** 2).sum(1).mean()) + 1e-8)


def _sel_regret(lb, v):
    lb = np.asarray(lb, float); v = np.asarray(v, float)
    rng = v.max() - v.min()
    return 0.0 if rng < 1e-9 else float((v.max() - v[int(np.argmax(lb))]) / rng)


def _bll_lb(enc, s1, s2, y, X, scale, rng):
    raw = lambda Z: _z(enc, Z, scale)
    H = np.vstack([raw(s1), raw(s2)]); mu = H.mean(0); sd = H.std(0) + 1e-8
    h = lambda Z: (raw(Z) - mu) / sd
    post = RewardPosterior(enc.h)
    post.fit(h(s1) - h(s2), np.asarray(y, float), BETAT, prior_scale=PRIOR, n_samples=NS, rng=rng)
    return np.array([post.lower_bound(f, ALPHA) for f in h(X)])


def analyze_cell(paths):
    runs = [r for r in (_load(p) for p in paths) if len(r) >= 6]
    if len(runs) < 2:
        return None
    Mmulti = np.vstack([np.stack([x["m"] for x in r])[len(r) // 2:] for r in runs])
    conds = ["full"] + [f"frozen_b{b}" for b in BS] + [f"fresh_b{b}" for b in BS]
    out = {c: {"single_reg": [], "multi_reg": [], "single_rank": [], "multi_rank": []} for c in conds}
    for r in runs:
        feats = np.stack([x["m"] for x in r]); scale = _scale(r); wT = r[-1]["w"]
        Msingle = feats[len(feats) // 2:]
        E = len(r) // 3
        s1e, s2e, ye = _pool(r, E - 1); s1a, s2a, ya = _pool(r, len(r) - 1)
        if s1e is None or s1a is None or len(ye) < 20 or len(ya) < 20:
            continue
        dphi = s1a - s2a
        def _log(name, lb_s, lb_m):
            out[name]["single_reg"].append(_sel_regret(lb_s, Msingle @ wT))
            out[name]["multi_reg"].append(_sel_regret(lb_m, Mmulti @ wT))
            out[name]["single_rank"].append(rank_correlation(lb_s.tolist(), (Msingle @ wT).tolist()))
            out[name]["multi_rank"].append(rank_correlation(lb_m.tolist(), (Mmulti @ wT).tolist()))
        rng = np.random.default_rng(1)
        _log("full",
             _fit_linear(dphi, ya, Msingle, "full", 3, scale, rng, Pmat=None),
             _fit_linear(dphi, ya, Mmulti, "full", 3, scale, rng, Pmat=None))
        for b in BS:
            ee = _enc_fit(s1e, s2e, ye, scale, b, np.random.default_rng(1))
            ef = _enc_fit(s1a, s2a, ya, scale, b, np.random.default_rng(2))
            _log(f"frozen_b{b}",
                 _bll_lb(ee, s1a, s2a, ya, Msingle, scale, np.random.default_rng(3)),
                 _bll_lb(ee, s1a, s2a, ya, Mmulti, scale, np.random.default_rng(3)))
            _log(f"fresh_b{b}",
                 _bll_lb(ef, s1a, s2a, ya, Msingle, scale, np.random.default_rng(4)),
                 _bll_lb(ef, s1a, s2a, ya, Mmulti, scale, np.random.default_rng(4)))
    return out


def _worker(args):
    (task, arm), paths = args
    return dict(arm=arm, res=analyze_cell(paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_frozen_multirun"); os.makedirs(out_dir, exist_ok=True)
    runs = [p for p in sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
            if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    cells = defaultdict(list)
    for p in runs:
        parts = p.rstrip("/").split(os.sep); cells[(parts[-3], parts[-2])].append(p)
    print(f"found {len(runs)} runs in {len(cells)} cells")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, list(cells.items())):
            results.append(r)
    arms = ["NONE", "SAT_lam0.5", "SAT_lam1.0"]
    conds = ["full"] + [f"frozen_b{b}" for b in BS] + [f"fresh_b{b}" for b in BS]
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r["res"]:
            continue
        for c in conds:
            for m in ("single_reg", "multi_reg", "single_rank", "multi_rank"):
                if r["res"][c][m]:
                    agg[(r["arm"], c)][m].append(np.mean(r["res"][c][m]))
    L = ["# Frozen pre-drift encoder on single-run vs MULTI-run pools\n",
         "Selection regret ranking the run's OWN pool (single) vs the all-seed pool (multi) under "
         "the run's w_T. Does natural early-freezing feature drop (frozen vs fresh) bite on the "
         "diverse multi-run pool?\n",
         "| arm | condition | single_reg | multi_reg | single_rank | multi_rank |",
         "|---|---|---|---|---|---|"]
    for arm in arms:
        for c in conds:
            a = agg[(arm, c)]
            if a["multi_reg"]:
                L.append(f"| {arm} | {c} | {np.mean(a['single_reg']):.3f} | {np.mean(a['multi_reg']):.3f} "
                         f"| {np.mean(a['single_rank']):+.2f} | {np.mean(a['multi_rank']):+.2f} |")
    open(os.path.join(out_dir, "results.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
