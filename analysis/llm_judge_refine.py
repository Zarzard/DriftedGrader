"""Clean refinement of the LLM-as-judge drift demo — isolate DRIFT-coverage from baseline
representation mismatch.

(1) Full-embedding ridge probe (held-out R^2): how well does the frozen LLM embedding e(x)
    linearly encode each attribute axis, the initial standard w0, the final standard w_end, the
    DRIFT direction u=w_end-w0, and RANDOM attribute directions? If cov(u) ~ cov(random), there
    is no drift-SPECIFIC blind spot; the issue is general mismatch, not drift.
(2) Selection regret / rank PER drift stage (full trend) for oracle / llm_frozen(PCA-128) /
    llm_kproj — does regret RISE as the standard drifts (drift into a blind spot) or not?

Usage: HF_HOME=... python -m analysis.llm_judge_refine --out <dir> [--model ...] [--n 1500]
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
from drifted_grader.metrics import rank_correlation
from analysis.llm_judge_demo import (attrs, load_corpus, embed, _std, _pca, _fit_lb,
                                     ATTR_NAMES, make_projection)


def ridge_r2(E, t, seed=0, lams=(1.0, 10.0, 100.0, 1000.0)):
    """Held-out R^2 of a ridge linear probe predicting scalar target t from features E."""
    rng = np.random.default_rng(seed); n = len(E); idx = rng.permutation(n)
    tr, te = idx[:int(0.7 * n)], idx[int(0.7 * n):]
    Xtr = np.hstack([E[tr], np.ones((len(tr), 1))]); Xte = np.hstack([E[te], np.ones((len(te), 1))])
    ttr, tte = t[tr], t[te]; G = Xtr.T @ Xtr; I = np.eye(G.shape[0]); best = -1e9
    for lam in lams:
        w = np.linalg.solve(G + lam * I, Xtr.T @ ttr)
        pred = Xte @ w
        r2 = 1 - ((tte - pred) ** 2).sum() / (((tte - tte.mean()) ** 2).sum() + 1e-12)
        best = max(best, float(r2))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--stages", type=int, default=9)
    ap.add_argument("--npairs", type=int, default=3000)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--kproj", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(0)

    texts = load_corpus(args.n)
    A = _std(np.stack([attrs(t) for t in texts]))
    E = _std(embed(texts, args.model))                       # full 2048-d, standardized
    d = A.shape[1]
    print(f"corpus {len(texts)}  A {A.shape}  E {E.shape}", flush=True)

    w0 = np.array([+1.0, 0.0, +1.0, -0.5, +0.4, +1.0]); w0 /= np.linalg.norm(w0)
    wE = np.array([-1.0, +1.5, 0.0, -1.0, 0.0, +0.4]); wE /= np.linalg.norm(wE)
    u = wE - w0; u /= np.linalg.norm(u)

    # ---- (1) full-embedding coverage: is the DRIFT axis specifically uncovered? ----
    cov = {}
    for name, vec in [("w0", w0), ("w_end", wE), ("drift_u", u)]:
        cov[name] = ridge_r2(E, A @ vec)
    for i, an in enumerate(ATTR_NAMES):
        cov[f"attr:{an}"] = ridge_r2(E, A[:, i])
    randR2 = [ridge_r2(E, A @ (lambda r: r / np.linalg.norm(r))(rng.standard_normal(d)), seed=i)
              for i in range(20)]
    cov["random(mean)"] = float(np.mean(randR2)); cov["random(sd)"] = float(np.std(randR2))

    # ---- (2) regret / rank per drift stage ----
    Epca = _std(_pca(E, args.pca))
    scaleA = 1.0 / (np.sqrt((np.diff(A[rng.integers(0, len(A), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    scaleE = 1.0 / (np.sqrt((np.diff(Epca[rng.integers(0, len(Epca), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    P_kproj = make_projection(Epca.shape[1], args.kproj, rng)
    per_stage = []
    for s in range(args.stages):
        frac = s / (args.stages - 1)
        wt = (1 - frac) * w0 + frac * wE; wt /= np.linalg.norm(wt)
        vt = A @ wt
        ij = rng.integers(0, len(A), (args.npairs, 2)); ij = ij[ij[:, 0] != ij[:, 1]]
        dA = A[ij[:, 0]] - A[ij[:, 1]]; dE = Epca[ij[:, 0]] - Epca[ij[:, 1]]
        pref = 1.0 / (1.0 + np.exp(-10.0 * (dA @ wt)))
        y = (rng.random(len(pref)) < pref).astype(float)
        rr = np.random.default_rng(s + 1)
        lbs = {"oracle_attr": _fit_lb(dA, y, A, scaleA, rr),
               "llm_frozen": _fit_lb(dE, y, Epca, scaleE, rr),
               "llm_kproj": _fit_lb(dE, y, Epca, scaleE, rr, P=P_kproj)}
        hi, lo = vt.max(), vt.min()
        row = {"stage": s, "frac": round(frac, 2)}
        for c, lb in lbs.items():
            row[f"{c}_reg"] = round(float((hi - vt[int(np.argmax(lb))]) / (hi - lo + 1e-9)), 3)
            row[f"{c}_rank"] = round(rank_correlation(lb.tolist(), vt.tolist()), 3)
        per_stage.append(row)

    json.dump({"coverage": cov, "per_stage": per_stage}, open(os.path.join(args.out, "refine.json"), "w"), indent=2)
    L = ["# LLM-as-judge drift — clean refinement\n",
         f"model={args.model}  corpus={len(texts)}  full-embedding ridge probe (held-out R^2)\n",
         "## (1) Does the frozen LLM embedding cover the DRIFT axis specifically?\n",
         "| direction | held-out R^2 |", "|---|---|"]
    for k in ["w0", "w_end", "drift_u"] + [f"attr:{a}" for a in ATTR_NAMES]:
        L.append(f"| {k} | {cov[k]:.3f} |")
    L.append(f"| random dir (mean±sd) | {cov['random(mean)']:.3f} ± {cov['random(sd)']:.3f} |")
    L.append("\nIf drift_u R^2 ~ random => NO drift-specific blind spot (general mismatch, not drift). "
             "If drift_u << random => a real drift-specific blind spot.\n")
    L.append("## (2) Selection regret & rank per drift stage (does regret RISE with drift?)\n")
    L.append("| stage | frac | oracle_reg | llm_frozen_reg | llm_kproj_reg | llm_frozen_rank | llm_kproj_rank |")
    L.append("|---|---|---|---|---|---|---|")
    for r in per_stage:
        L.append(f"| {r['stage']} | {r['frac']} | {r['oracle_attr_reg']} | {r['llm_frozen_reg']} "
                 f"| {r['llm_kproj_reg']} | {r['llm_frozen_rank']:+.2f} | {r['llm_kproj_rank']:+.2f} |")
    # trend: slope of llm_frozen regret vs stage
    xs = np.array([r["stage"] for r in per_stage]); ys = np.array([r["llm_frozen_reg"] for r in per_stage])
    slope = float(np.polyfit(xs, ys, 1)[0])
    L.append(f"\nllm_frozen regret slope vs stage = {slope:+.4f}/stage "
             f"({'RISES with drift' if slope > 0.01 else 'flat/falls -> NOT a drift effect'}).\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
