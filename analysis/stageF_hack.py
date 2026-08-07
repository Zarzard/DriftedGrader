"""Adversarial-to-the-bound hacking probe (B-REX reward-hacking detection under drift).

Instead of hand-building one impostor, we let EACH bound pick its own most-tempting
policy from the feasible feature box and check whether that policy is actually good.
This is the tightest "reward hacking" test: the practitioner deploys argmax-VaR over the
whole feasible set; if the bound is exploitable, its favourite is truly bad.

Per stage t:
  - feasible set = corners of the observed feature box [min_j, max_j]^d (2^d candidates;
    the OOD adversarial combinations, analogous to B-REX's out-of-distribution No-Op).
  - fit each reward model on the pooled comparisons up to t (real data).
  - c* = argmax_corner VaR_condition(corner)   (the policy the bound would deploy)
  - box_regret = (max_c v_t(c) - v_t(c*)) / (max_c v_t(c) - min_c v_t(c)),  v_t = corner . w_t
    (0 = deploys the truly-best feasible policy; 1 = deploys the truly-worst)
  - hack_true_pct = true-value percentile of c* among corners (LOW = deployed a bad one)

Conditions: linear {full, kproj, adv} and nonlinear BLL {mlp_full, mlp_kproj, mlp_adv},
the same 2x3 (model class) x (full / generic-k / drift-blind-k) grid as the ranking
analysis. Prediction: adv and mlp_adv (blind to the drift axis) are exploitable under
drift; full/kproj and mlp_full/mlp_kproj are not.

Run: python -m analysis.stageF_hack --root <root> --out <dir> --workers 16
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from drifted_grader.reward_models import make_projection
from analysis.stageF_ranking import _load, _pool, _adv_projection, _fit_linear
from analysis.stageF_scaleup import _fit_mlp_bll, _parse_path

LIN = ("full", "kproj", "adv")
MLP = ("mlp_full", "mlp_kproj", "mlp_adv")
CONDS = LIN + MLP
_MLP_P = {"mlp_full": "none", "mlp_kproj": "kproj", "mlp_adv": "adv"}


def _pct(x, arr):
    arr = np.asarray(arr, float)
    return float((arr < x).mean())


def _box_corners(pool, cap=64):
    lo = pool.min(0); hi = pool.max(0)
    d = len(lo)
    if 2 ** d <= cap:
        return np.array([[hi[j] if b else lo[j] for j, b in enumerate(bits)]
                         for bits in itertools.product([0, 1], repeat=d)])
    # too many dims: random subset of corners
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(cap, d))
    return np.where(bits.astype(bool), hi, lo)


def _regret(v_at_pick, v_all):
    v_all = np.asarray(v_all, float)
    rng = v_all.max() - v_all.min()
    return 0.0 if rng < 1e-12 else float((v_all.max() - v_at_pick) / rng)


def analyze_run(path, k=3, stages=12):
    rounds = _load(path)
    if len(rounds) < 4:
        return None
    feats = np.stack([r["m"] for r in rounds])
    a1 = np.concatenate([r["sphi1"] for r in rounds])
    a2 = np.concatenate([r["sphi2"] for r in rounds])
    scale = 1.0 / (np.sqrt(((a1 - a2) ** 2).sum(1).mean()) + 1e-8)
    u = rounds[-1]["w"] - rounds[0]["w"]
    rng = np.random.default_rng(0)
    P_adv = (_adv_projection(feats.shape[1], k, u, rng) if np.linalg.norm(u) > 1e-6
             else make_projection(feats.shape[1], k, rng))
    P_kproj = make_projection(feats.shape[1], k, np.random.default_rng(777))
    Pmap = {"mlp_full": None, "mlp_kproj": P_kproj, "mlp_adv": P_adv}
    idx = np.unique(np.linspace(len(rounds) // 3, len(rounds) - 1, stages).astype(int))
    out = {c: {"regret": [], "true_pct": []} for c in CONDS}
    for t in idx:
        s1, s2, y = _pool(rounds, t)
        if s1 is None or len(y) < 20:
            continue
        dphi = s1 - s2
        wt = rounds[t]["w"]; pool = feats[: t + 1]
        C = _box_corners(pool)
        v = C @ wt                                  # true value of each feasible corner
        rng2 = np.random.default_rng(int(t) + 1)
        for kind in LIN:
            lb = _fit_linear(dphi, y, C, kind, k, scale, rng2, Pmat=P_adv)
            star = int(np.argmax(lb))
            out[kind]["regret"].append(_regret(v[star], v))
            out[kind]["true_pct"].append(_pct(v[star], v))
        for name in MLP:
            lb = _fit_mlp_bll(s1, s2, y, C, scale, rng2, P=Pmap[name])
            star = int(np.argmax(lb))
            out[name]["regret"].append(_regret(v[star], v))
            out[name]["true_pct"].append(_pct(v[star], v))
    return out


def _worker(args):
    path, k = args
    task, arm, seed = _parse_path(path)
    return dict(task=task, arm=arm, seed=seed, res=analyze_run(path, k=k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_hack")
    os.makedirs(out_dir, exist_ok=True)
    runs = sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
    runs = [p for p in runs if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    print(f"found {len(runs)} runs")
    targs = [(p, args.k) for p in runs]
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_worker, targs):
                results.append(r)
    else:
        results = [_worker(a) for a in targs]

    agg = defaultdict(lambda: defaultdict(list))   # (task,arm,cond)->metric->[per-seed mean]
    rows = []
    for r in results:
        if r["res"] is None:
            continue
        for c in CONDS:
            reg = r["res"][c]["regret"]
            if reg:
                agg[(r["task"], r["arm"], c)]["regret"].append(float(np.mean(reg)))
                agg[(r["task"], r["arm"], c)]["true_pct"].append(float(np.mean(r["res"][c]["true_pct"])))
                rows.append(dict(task=r["task"], arm=r["arm"], seed=r["seed"], condition=c,
                                 box_regret=float(np.mean(reg)),
                                 pick_true_pct=float(np.mean(r["res"][c]["true_pct"]))))
    with open(os.path.join(out_dir, "per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "arm", "seed", "condition", "box_regret", "pick_true_pct"])
        w.writeheader(); w.writerows(rows)

    tasks = sorted({r["task"] for r in results})
    arms = ["NONE", "SAT_lam0.5", "SAT_lam1.0"]
    L = ["# Stage F — adversarial-to-the-bound hacking probe (2x3 representation grid)\n",
         "Each bound deploys its favourite policy from the feasible feature box; "
         "box_regret = normalized true-value regret of that deployed policy (0=best, 1=worst). "
         "Higher = the bound is exploitable / gets hacked under drift.\n"]
    for metric, title in (("regret", "Box regret of the bound-optimal policy (higher = hacked)"),
                          ("true_pct", "True-value percentile of the deployed policy (lower = worse)")):
        L.append(f"\n## {title}\n")
        L.append("| task | condition | " + " | ".join(arms) + " |")
        L.append("|" + "---|" * (2 + len(arms)))
        for task in tasks:
            for c in CONDS:
                cells = [(" %.2f" % np.mean(agg[(task, arm, c)][metric])) if agg[(task, arm, c)][metric] else "n/a"
                         for arm in arms]
                L.append(f"| {task} | {c} | " + " | ".join(cells) + " |")
        L.append("| **POOLED** | | | | |")
        for c in CONDS:
            cells = []
            for arm in arms:
                pooled = [x for (tk, am, cc), d in agg.items() if am == arm and cc == c for x in d[metric]]
                cells.append(("%.2f" % np.mean(pooled)) if pooled else "n/a")
            L.append(f"| POOLED | {c} | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out_dir}/results.md and per_run.csv")


if __name__ == "__main__":
    main()
