"""Pooled-RM multi-run control: removes the cross-run OOD confound.

Fit ONE reward model on the POOLED comparisons of ALL seeds (so every candidate policy is
in-distribution), then rank the multi-seed pool under each reference seed's grader w_T. If the
NONE regret now drops toward 0 (cross-run OOD removed) while SAT regret stays high, the residual
is the DRIFT effect. Conditions full / adv / mlp_full / mlp_adv.

Run: python -m analysis.stageF_multirun_pooled --root <root> --out <dir> --workers 16
"""
from __future__ import annotations
import argparse, glob, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import rank_correlation
from analysis.stageF_ranking import _load, _pool, _adv_projection
from analysis.stageF_scaleup import _fit_mlp_bll, _fit_linear

CONDS = ("full", "adv", "mlp_full", "mlp_adv")
N_CAP = 2500   # subsample budget for the pooled comparison set (speed)


def _sel_regret(lb, v):
    lb = np.asarray(lb, float); v = np.asarray(v, float)
    rng = v.max() - v.min()
    return 0.0 if rng < 1e-9 else float((v.max() - v[int(np.argmax(lb))]) / rng)


def analyze_cell(paths):
    runs = [r for r in (_load(p) for p in paths) if len(r) >= 4]
    if len(runs) < 2:
        return None
    S1, S2, Y = [], [], []
    for r in runs:
        s1, s2, y = _pool(r, len(r) - 1)
        if s1 is not None:
            S1.append(s1); S2.append(s2); Y.append(y)
    if not S1:
        return None
    S1 = np.concatenate(S1); S2 = np.concatenate(S2); Y = np.concatenate(Y)
    if len(Y) > N_CAP:
        sel = np.random.default_rng(0).choice(len(Y), N_CAP, replace=False)
        S1, S2, Y = S1[sel], S2[sel], Y[sel]
    dphi = S1 - S2
    scale = 1.0 / (np.sqrt((dphi ** 2).sum(1).mean()) + 1e-8)
    M = np.vstack([np.stack([x["m"] for x in r])[len(r) // 2:] for r in runs])
    out = {c: {"reg": [], "rank": []} for c in CONDS}
    for r in runs:
        wT = r[-1]["w"]; u = wT - r[0]["w"]; nu = np.linalg.norm(u)
        P_adv = _adv_projection(6, 3, u, np.random.default_rng(0)) if nu > 1e-6 else make_projection(6, 3, np.random.default_rng(0))
        vt = M @ wT; rng2 = np.random.default_rng(1)
        for kind in ("full", "adv"):
            lb = _fit_linear(dphi, Y, M, kind, 3, scale, rng2, Pmat=P_adv)
            out[kind]["reg"].append(_sel_regret(lb, vt)); out[kind]["rank"].append(rank_correlation(lb.tolist(), vt.tolist()))
        for name, Pm in (("mlp_full", None), ("mlp_adv", P_adv)):
            lb = _fit_mlp_bll(S1, S2, Y, M, scale, rng2, P=Pm)
            out[name]["reg"].append(_sel_regret(lb, vt)); out[name]["rank"].append(rank_correlation(lb.tolist(), vt.tolist()))
    return out


def _worker(args):
    (task, arm), paths = args
    return dict(arm=arm, res=analyze_cell(paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_multirun_pooled"); os.makedirs(out_dir, exist_ok=True)
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
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r["res"]:
            continue
        for c in CONDS:
            if r["res"][c]["reg"]:
                agg[(r["arm"], c)]["reg"].append(np.mean(r["res"][c]["reg"]))
                agg[(r["arm"], c)]["rank"].append(np.mean(r["res"][c]["rank"]))
    L = ["# Pooled-RM multi-run (cross-run OOD removed) — selection regret / rank\n",
         "One RM fit on ALL seeds' pooled prefs; rank the multi-seed pool under each seed's w_T. "
         "NONE ~0 would confirm the earlier multi-run NONE regret was cross-run OOD, not drift.\n"]
    for metric, title in (("reg", "Selection regret (lower better)"), ("rank", "Rank corr (higher better)")):
        L.append(f"\n## {title}\n| condition | " + " | ".join(arms) + " |")
        L.append("|" + "---|" * (1 + len(arms)))
        for c in CONDS:
            cells_ = [(f"{np.mean(agg[(a, c)][metric]):+.3f}" if agg[(a, c)][metric] else "n/a") for a in arms]
            L.append(f"| {c} | " + " | ".join(cells_) + " |")
    open(os.path.join(out_dir, "results.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
