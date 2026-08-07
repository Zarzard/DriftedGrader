"""Multi-run policy pool: does selecting among policies from MANY runs (more diverse,
spanning more directions incl. the drift axis u) expose the feature-drop harm that a
single run's clustered checkpoints did NOT?

For each (task, arm): pool the last-half achieved profiles across all seeds into one eval
set M. For each reference seed s (its reward model + current grader w_T), rank M and
report selection regret / rank corr on the MULTI-RUN pool. Also report the pool's spread
along the drift axis u vs a single run's, to confirm the multi-run pool is more diverse
along u. Compare conditions full / adv (linear) and mlp_full / mlp_adv (nonlinear BLL).

Run: python -m analysis.stageF_multirun --root <root> --out <dir> --workers 16
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import rank_correlation
from analysis.stageF_ranking import _load, _pool, _adv_projection, _fit_linear
from analysis.stageF_scaleup import _fit_mlp_bll

CONDS = ("full", "adv", "mlp_full", "mlp_adv")


def _scale(rounds):
    a1 = np.concatenate([r["sphi1"] for r in rounds])
    a2 = np.concatenate([r["sphi2"] for r in rounds])
    return 1.0 / (np.sqrt(((a1 - a2) ** 2).sum(1).mean()) + 1e-8)


def _sel_regret(lb, v):
    lb = np.asarray(lb, float); v = np.asarray(v, float)
    rng = v.max() - v.min()
    return 0.0 if rng < 1e-9 else float((v.max() - v[int(np.argmax(lb))]) / rng)


def analyze_cell(paths):
    """paths = all seeds of one (task, arm)."""
    runs = [r for r in (_load(p) for p in paths) if len(r) >= 4]
    if len(runs) < 2:
        return None
    # multi-run pool = last-half achieved profiles across all runs
    M = np.vstack([np.stack([x["m"] for x in r])[len(r) // 2:] for r in runs])
    out = {c: {"rank": [], "reg": []} for c in CONDS}
    spread = {"multi_over_single": []}
    for r in runs:                                   # each run supplies its RM + grader
        feats = np.stack([x["m"] for x in r]); scale = _scale(r)
        w0 = r[0]["w"]; wT = r[-1]["w"]; u = wT - w0
        s1, s2, y = _pool(r, len(r) - 1)
        if s1 is None or len(y) < 20:
            continue
        dphi = s1 - s2
        nu = np.linalg.norm(u)
        rng = np.random.default_rng(0)
        P_adv = (_adv_projection(6, 3, u, rng) if nu > 1e-6 else make_projection(6, 3, rng))
        vt = M @ wT
        if nu > 1e-6:
            uu = u / nu
            spread["multi_over_single"].append(
                float((M @ uu).std() / ((feats[len(feats) // 2:] @ uu).std() + 1e-9)))
        rng2 = np.random.default_rng(1)
        for kind in ("full", "adv"):
            lb = _fit_linear(dphi, y, M, kind, 3, scale, rng2, Pmat=P_adv)
            out[kind]["rank"].append(rank_correlation(lb.tolist(), vt.tolist()))
            out[kind]["reg"].append(_sel_regret(lb, vt))
        for name, Pm in (("mlp_full", None), ("mlp_adv", P_adv)):
            lb = _fit_mlp_bll(s1, s2, y, M, scale, rng2, P=Pm)
            out[name]["rank"].append(rank_correlation(lb.tolist(), vt.tolist()))
            out[name]["reg"].append(_sel_regret(lb, vt))
    return out, spread


def _worker(args):
    (task, arm), paths = args
    return dict(task=task, arm=arm, res=analyze_cell(paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_multirun")
    os.makedirs(out_dir, exist_ok=True)
    runs = sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
    runs = [p for p in runs if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    cells = defaultdict(list)
    for p in runs:
        parts = p.rstrip("/").split(os.sep)
        cells[(parts[-3], parts[-2])].append(p)
    items = list(cells.items())
    print(f"found {len(runs)} runs in {len(items)} (task,arm) cells")
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_worker, items):
                results.append(r)
    else:
        results = [_worker(a) for a in items]

    arms = ["NONE", "SAT_lam0.5", "SAT_lam1.0"]
    agg = defaultdict(lambda: defaultdict(list))     # (arm,cond)->metric->[per-cell means]
    sp = defaultdict(list)
    for r in results:
        if not r["res"]:
            continue
        out, spread = r["res"]
        if spread["multi_over_single"]:
            sp[r["arm"]].append(np.mean(spread["multi_over_single"]))
        for c in CONDS:
            if out[c]["reg"]:
                agg[(r["arm"], c)]["reg"].append(float(np.mean(out[c]["reg"])))
                agg[(r["arm"], c)]["rank"].append(float(np.mean(out[c]["rank"])))

    L = ["# Multi-run policy pool — does a diverse (multi-seed) pool expose the feature drop?\n",
         "Selection regret (lower=better) and rank corr (higher=better) ranking the pooled "
         "profiles of ALL seeds under each reference run's grader. Compare to the single-run "
         "numbers in results/.\n"]
    for metric, title in (("reg", "Selection regret on the multi-run pool (lower better)"),
                          ("rank", "Rank correlation on the multi-run pool (higher better)")):
        L.append(f"\n## {title}\n| condition | " + " | ".join(arms) + " |")
        L.append("|" + "---|" * (1 + len(arms)))
        for c in CONDS:
            cells_ = [(f"{np.mean(agg[(arm, c)][metric]):+.3f}" if agg[(arm, c)][metric] else "n/a")
                      for arm in arms]
            L.append(f"| {c} | " + " | ".join(cells_) + " |")
    L.append("\n## Pool diversity along u (multi-run spread / single-run spread)\n")
    L.append("| arm | spread ratio |\n|---|---|")
    for arm in arms:
        if sp[arm]:
            L.append(f"| {arm} | {np.mean(sp[arm]):.2f}x |")
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
