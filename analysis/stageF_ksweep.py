"""How narrow must the linear head be to enable drift-driven VaR-selection failure? Sweep the
head dimension k for a RANDOM projection (kproj) and a drift-BLIND projection (adv), per task,
and report selection regret vs k -- and the drift effect (SAT - NONE) vs k. Finds the threshold
k* below which drift-driven failure appears, and whether it is task-dependent.

Run: python -m analysis.stageF_ksweep --root <root> --out <dir> --workers 16
"""
from __future__ import annotations
import argparse, glob, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from drifted_grader.reward_models import make_projection
from analysis.stageF_ranking import _load, _pool, _adv_projection, _fit_linear
from analysis.stageF_scaleup import _sel_regret, _parse_path

KS = (1, 2, 3, 4, 5)


def analyze_run(path, stages=10):
    rounds = _load(path)
    if len(rounds) < 4:
        return None
    feats = np.stack([r["m"] for r in rounds]); d = feats.shape[1]
    a1 = np.concatenate([r["sphi1"] for r in rounds]); a2 = np.concatenate([r["sphi2"] for r in rounds])
    scale = 1.0 / (np.sqrt(((a1 - a2) ** 2).sum(1).mean()) + 1e-8)
    u = rounds[-1]["w"] - rounds[0]["w"]
    rng = np.random.default_rng(0)
    Padv = {k: (_adv_projection(d, k, u, rng) if np.linalg.norm(u) > 1e-6 else make_projection(d, k, rng)) for k in KS}
    idx = np.unique(np.linspace(len(rounds) // 3, len(rounds) - 1, stages).astype(int))
    out = {f"{c}_k{k}": [] for k in KS for c in ("kproj", "adv")}
    for t in idx:
        s1, s2, y = _pool(rounds, t)
        if s1 is None or len(y) < 20:
            continue
        dphi = s1 - s2; wt = rounds[t]["w"]; ev = feats[: t + 1]; vt = ev @ wt
        rr = np.random.default_rng(int(t) + 1)
        for k in KS:
            lb = _fit_linear(dphi, y, ev, "kproj", k, scale, rr)
            out[f"kproj_k{k}"].append(_sel_regret(lb, vt))
            lb = _fit_linear(dphi, y, ev, "adv", k, scale, rr, Pmat=Padv[k])
            out[f"adv_k{k}"].append(_sel_regret(lb, vt))
    return out


def _worker(p):
    task, arm, seed = _parse_path(p)
    return dict(task=task, arm=arm, res=analyze_run(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_ksweep"); os.makedirs(out_dir, exist_ok=True)
    runs = [p for p in sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
            if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    print(f"found {len(runs)} runs")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, runs):
            results.append(r)
    tasks = sorted({r["task"] for r in results})
    arms = ["NONE", "SAT_lam0.5", "SAT_lam1.0"]
    agg = defaultdict(list)                              # (task,arm,cond_k)->[per-run means]
    for r in results:
        if not r["res"]:
            continue
        for key, v in r["res"].items():
            if v:
                agg[(r["task"], r["arm"], key)].append(np.mean(v))

    def m(task, arm, cond, k):
        v = agg[(task, arm, f"{cond}_k{k}")]
        return float(np.mean(v)) if v else float("nan")

    L = ["# How narrow must the head be? Selection regret vs head dimension k (d=6 features)\n",
         "kproj = random k-dim head; adv = k-dim head blind to the drift direction u.\n"]
    for cond in ("kproj", "adv"):
        L.append(f"\n## {cond}: selection regret by k (pooled over tasks/seeds)\n")
        L.append("| arm | " + " | ".join(f"k={k}" for k in KS) + " |")
        L.append("|" + "---|" * (1 + len(KS)))
        for arm in arms:
            cells = []
            for k in KS:
                vals = [x for (t, a, key), lst in agg.items() if a == arm and key == f"{cond}_k{k}" for x in lst]
                cells.append(f"{np.mean(vals):.3f}" if vals else "n/a")
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        # drift effect SAT_lam0.5 - NONE vs k
        L.append(f"\n{cond} DRIFT effect (SAT_lam0.5 regret - NONE regret) vs k:")
        de = []
        for k in KS:
            sat = [x for (t, a, key), lst in agg.items() if a == "SAT_lam0.5" and key == f"{cond}_k{k}" for x in lst]
            non = [x for (t, a, key), lst in agg.items() if a == "NONE" and key == f"{cond}_k{k}" for x in lst]
            de.append(f"k={k}:{(np.mean(sat) - np.mean(non)):+.3f}")
        L.append("  " + "  ".join(de) + "\n")

    # per-task threshold k* for kproj (drift effect SAT0.5-NONE first drops below 0.05)
    L.append("\n## Per-task kproj drift effect (SAT_lam0.5 - NONE) vs k -> threshold k*\n")
    L.append("| task | " + " | ".join(f"k={k}" for k in KS) + " | k* (drift<0.05) |")
    L.append("|" + "---|" * (2 + len(KS)))
    for task in tasks:
        row = []; kstar = ">5"
        for k in KS:
            de = m(task, "SAT_lam0.5", "kproj", k) - m(task, "NONE", "kproj", k)
            row.append(f"{de:+.3f}")
            if kstar == ">5" and de < 0.05:
                kstar = str(k)
        L.append(f"| {task.replace('metaworld_','')} | " + " | ".join(row) + f" | {kstar} |")
    open(os.path.join(out_dir, "results.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
