"""Verify the graded-coverage LAW: across many criteria, does the frozen judge's coverage R^2
predict how well it can SELECT on that criterion? Also verifies R^2 itself (multi-seed CV +
shuffled-label null). 26 criteria = 6 pure attributes + 20 random weightings.

For each criterion c (unit weight over the 6 attributes): target t = a(x).c.
  coverage R^2   : held-out ridge probe of t from the embedding, mean+/-sd over seeds
  null R^2       : same with SHUFFLED labels (should be ~0 -> probe isn't overfitting)
  judge_rank     : Spearman(judge VaR-lb, t) when the standard = c (can the judge select on it?)
  oracle_rank    : same using the true attributes (~1, sanity)
Report the Spearman correlation between coverage R^2 and judge_rank across criteria.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
from drifted_grader.metrics import rank_correlation
from analysis.llm_judge_demo import attrs, load_corpus, embed, _std, _pca, _fit_lb
from analysis.llm_judge_refine import ridge_r2


def cov_cv(E, t, seeds=5):
    r = [ridge_r2(E, t, seed=s) for s in range(seeds)]
    return float(np.mean(r)), float(np.std(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--ncrit", type=int, default=20)
    ap.add_argument("--pca", type=int, default=128); ap.add_argument("--npairs", type=int, default=3000)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(0)
    texts = load_corpus(args.n)
    A = _std(np.stack([attrs(t) for t in texts])); d = A.shape[1]
    E = _std(embed(texts, args.model)); Epca = _std(_pca(E, args.pca))
    print(f"A {A.shape}  E {E.shape}", flush=True)
    scaleA = 1.0 / (np.sqrt((np.diff(A[rng.integers(0, len(A), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    scaleE = 1.0 / (np.sqrt((np.diff(Epca[rng.integers(0, len(Epca), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    crits = [np.eye(d)[i] for i in range(d)] + \
            [(lambda v: v / np.linalg.norm(v))(rng.standard_normal(d)) for _ in range(args.ncrit)]
    rows = []
    for c in crits:
        t = A @ c
        covm, covs = cov_cv(E, t)
        nullr = ridge_r2(E, rng.permutation(t.copy()))
        ij = rng.integers(0, len(A), (args.npairs, 2)); ij = ij[ij[:, 0] != ij[:, 1]]
        y = (rng.random(len(ij)) < 1.0 / (1.0 + np.exp(-10.0 * ((A[ij[:, 0]] - A[ij[:, 1]]) @ c)))).astype(float)
        rr = np.random.default_rng(1)
        lb_j = _fit_lb(Epca[ij[:, 0]] - Epca[ij[:, 1]], y, Epca, scaleE, rr)
        lb_o = _fit_lb(A[ij[:, 0]] - A[ij[:, 1]], y, A, scaleA, rr)
        rows.append(dict(covR2=round(covm, 3), covSD=round(covs, 3), null=round(nullr, 3),
                         judge_rank=round(rank_correlation(lb_j.tolist(), t.tolist()), 3),
                         oracle_rank=round(rank_correlation(lb_o.tolist(), t.tolist()), 3)))
        print(rows[-1], flush=True)
    cov = [r["covR2"] for r in rows]; jr = [r["judge_rank"] for r in rows]
    spear = rank_correlation(cov, jr)
    pear = float(np.corrcoef(cov, jr)[0, 1])
    json.dump({"rows": rows, "spearman_covR2_vs_judgerank": spear, "pearson": pear,
               "mean_null_r2": float(np.mean([r["null"] for r in rows]))}, open(os.path.join(args.out, "law.json"), "w"), indent=2)
    L = [f"# Graded-coverage LAW verification ({args.model}, {len(crits)} criteria)\n",
         f"**Coverage R² predicts the judge's selection rank: Spearman = {spear:+.2f}, Pearson = {pear:+.2f}.**",
         f"Null (shuffled-label) R² mean = {np.mean([r['null'] for r in rows]):.3f} (~0 => probe not overfitting). "
         f"Oracle rank mean = {np.mean([r['oracle_rank'] for r in rows]):.2f} (sanity).\n",
         "| coverage R² (mean±sd) | null R² | judge selection rank |", "|---|---|---|"]
    for r in sorted(rows, key=lambda z: z["covR2"]):
        L.append(f"| {r['covR2']:.3f} ± {r['covSD']:.3f} | {r['null']:+.3f} | {r['judge_rank']:+.3f} |")
    L.append("\nHigher coverage R² of a criterion => the frozen judge can select on it; low coverage "
             "=> selection fails. The monotone relationship IS the graded-coverage law.\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
