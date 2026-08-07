"""§8: gold-set sizing — how many labeled examples do you need to AUDIT coverage?

The coverage audit is only a recipe if you can estimate a criterion's coverage from a
small labeled set. For each criterion we compute the coverage estimate (held-out ridge
R^2 from the frozen 3B judge embedding) on random gold sets of increasing size n, over
many resamples, and ask:
  - CONVERGENCE: how does the estimate's mean/spread stabilize with n? What n* gives it
    to +-0.05 of the full-data value?
  - RANK STABILITY: at each n, does the ORDER of criteria by estimated coverage match the
    full-data order (Spearman)? -- i.e. how few labels to tell WHICH criteria are low-
    coverage (the decision the audit actually drives), which is easier than pinning each R^2.

Criteria = factual correctness (TruthfulQA is_correct) + the 6 interpretable surface/text
attributes -- spanning the coverage range (hedging ~0 .. specificity ~0.75).

Usage: HF_HOME=... python -m analysis.goldset --out <dir> [--model ...]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats as sps

from analysis.llm_judge_demo import embed, _pca, _std, attrs, ATTR_NAMES
from analysis.llm_judge_refine import ridge_r2
from analysis.semantic_blindspot import load_items

SIZES = [50, 100, 200, 500, 1000, 2000, 4000]
REPS = 40


def _cov(E, t, seed=0):
    """Coverage estimate, clipped to [-1,1]: an R² far below 0 (near-singular fit on a
    tiny gold set) is 'unusable', and its exact magnitude is meaningless noise."""
    return float(np.clip(ridge_r2(E, t, seed=seed), -1.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--pca", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    items = load_items(cap=args.cap)
    texts = [t for t, a, c in items]
    iscorr = np.array([c for t, a, c in items], float)
    Attr = np.stack([attrs(t) for t in texts])
    # PCA-reduce the frozen embedding first (fit unsupervised on the full pool -- no label
    # leakage): this is the practical audit representation and makes the gold-set
    # requirement modest (raw ~thousands-of-dim embeddings need far more labels).
    E = _std(_pca(_std(embed(texts, args.model)), args.pca))
    N = len(texts)
    print(f"items {N}  PCA-{args.pca} embeddings {E.shape}", flush=True)

    targets = {"is_correct": iscorr}
    for j, nm in enumerate(ATTR_NAMES):
        targets[nm] = Attr[:, j]
    names = list(targets)

    # full-data reference coverage (the value each estimate should converge to)
    full = {nm: _cov(E, t) for nm, t in targets.items()}
    print("full-data coverage:", {k: round(v, 3) for k, v in full.items()}, flush=True)

    # coverage estimate vs gold-set size
    curves = {nm: {} for nm in names}
    for nm, t in targets.items():
        for n in SIZES:
            if n > N:
                continue
            ests = []
            for rep in range(REPS):
                rng = np.random.default_rng(1000 * SIZES.index(n) + rep)
                idx = rng.choice(N, n, replace=False)
                ests.append(_cov(E[idx], t[idx], seed=rep))
            ests = np.array(ests)
            curves[nm][n] = dict(mean=float(ests.mean()), sd=float(ests.std()),
                                 lo=float(np.percentile(ests, 5)), hi=float(np.percentile(ests, 95)))
        print(nm, {n: round(curves[nm][n]["mean"], 2) for n in curves[nm]}, flush=True)

    # rank stability: order criteria by estimated coverage at each n vs full order
    full_order = [full[nm] for nm in names]
    rank_stab = {}
    for n in SIZES:
        if n > N:
            continue
        est_order = [curves[nm][n]["mean"] for nm in names]
        rank_stab[n] = float(sps.spearmanr(est_order, full_order)[0])

    # n* per criterion: smallest n with |mean-full|<=0.05 AND sd<=0.05
    def nstar(nm):
        for n in SIZES:
            if n in curves[nm] and abs(curves[nm][n]["mean"] - full[nm]) <= 0.05 and curves[nm][n]["sd"] <= 0.05:
                return n
        return None

    out = dict(model=args.model, n_items=N, full_coverage=full,
               curves=curves, rank_stability=rank_stab,
               nstar={nm: nstar(nm) for nm in names})
    json.dump(out, open(os.path.join(args.out, "goldset.json"), "w"), indent=1)

    L = [f"# §8 Gold-set sizing ({args.model}, {N} items)\n",
         "How many labeled examples to estimate a criterion's coverage? Coverage = "
         "held-out ridge R² from the frozen embedding; estimate computed on random gold "
         f"sets of size n, {REPS} resamples each (mean ± sd).\n",
         "## Coverage estimate (mean ± sd) vs gold-set size n\n",
         "| criterion | full | " + " | ".join(f"n={n}" for n in SIZES) + " | n\\* |",
         "|---|---|" + "---|" * (len(SIZES) + 1)]
    for nm in sorted(names, key=lambda z: -full[z]):
        cells = []
        for n in SIZES:
            c = curves[nm].get(n)
            cells.append(f"{c['mean']:.2f}±{c['sd']:.2f}" if c else "-")
        ns = nstar(nm)
        L.append(f"| {nm} | {full[nm]:.2f} | " + " | ".join(cells) + f" | {ns or '>max'} |")
    L.append("\n## Rank stability (Spearman of criterion coverage-order vs full-data order)\n")
    L.append("| n | " + " | ".join(str(n) for n in SIZES if n <= N) + " |")
    L.append("|---|" + "---|" * len([n for n in SIZES if n <= N]))
    L.append("| Spearman | " + " | ".join(f"{rank_stab[n]:+.2f}" for n in SIZES if n <= N) + " |")
    L.append("\n**Reading:** *n\\** = smallest gold set whose coverage estimate is within "
             "±0.05 of the full-data value with sd ≤ 0.05. RANK stability reaching ≈1 at a "
             "small n means you can tell WHICH criteria are low-coverage (the audit's actual "
             "decision) with far fewer labels than needed to pin each R² precisely.\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
        for nm in names:
            ns = [n for n in SIZES if n in curves[nm]]
            mean = [curves[nm][n]["mean"] for n in ns]
            lo = [curves[nm][n]["lo"] for n in ns]; hi = [curves[nm][n]["hi"] for n in ns]
            lab = nm + (" (correctness)" if nm == "is_correct" else "")
            line, = axA.plot(ns, mean, "-o", label=lab, lw=1.6, ms=4)
            axA.fill_between(ns, lo, hi, alpha=0.12, color=line.get_color())
            axA.axhline(full[nm], color=line.get_color(), lw=0.6, ls=":")
        axA.set_xscale("log"); axA.set_xlabel("gold-set size n (log)")
        axA.set_ylabel("estimated coverage R²  (band = 5–95%)"); axA.grid(alpha=0.3)
        axA.legend(fontsize=7, ncol=2); axA.set_title("Coverage estimate converges with n")
        nn = [n for n in SIZES if n <= N]
        axB.plot(nn, [rank_stab[n] for n in nn], "-o", c="#0072B2")
        axB.set_xscale("log"); axB.set_ylim(0, 1.05); axB.set_xlabel("gold-set size n (log)")
        axB.set_ylabel("Spearman(coverage order vs full)"); axB.grid(alpha=0.3)
        axB.set_title("Which criteria are low-coverage: stable early")
        fig.suptitle("§8 Gold-set sizing: how few labels to audit coverage", fontsize=11)
        fig.tight_layout(); fp = os.path.join(args.out, "fig_goldset.png")
        fig.savefig(fp, dpi=150); print("wrote", fp)
    except Exception as e:
        print("skip figure:", e)


if __name__ == "__main__":
    main()
