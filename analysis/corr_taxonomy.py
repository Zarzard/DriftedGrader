"""Correlation-sign taxonomy: when a judge optimizes/selects on a READABLE
criterion, does the UNREADABLE one (factual correctness, coverage R^2~0.20) get
dragged UP (positively correlated), stay flat (uncorrelated), or get actively
destroyed (negatively correlated -- classical Goodhart)?

For a linear head the policy/selection depends only on the readable coordinates
s(x); the unreadable criterion moves only through the regression E[n | s]. The
dangerous regime is blindness PLUS a negative sign. This measures that sign on
TruthfulQA with the frozen Qwen judge, for each readable surface attribute
(length/verbosity is the headline pair).

Three levels, per readable attribute s:
  (1) TRUE corr(s, is_correct)          -- the world's correlation
  (2) corr(s_hat, is_correct), s_hat = question-grouped cross-fitted ridge
      readout of s from the embedding   -- the sign along what the judge READS
  (3) selection drag: mean is_correct of the top-q picks under (a) true s,
      (b) s_hat, (c) the judge's preference-fit VaR bound for an s-only standard
      (the deployed object; refit over NSEEDS independent preference draws)

STATISTICAL GUARDS (per adversarial review): TruthfulQA items cluster ~6.3
answers/question and the question text is inside every embedding, so ALL
bootstrap CIs are CLUSTER bootstraps over questions (iid item bootstraps are
anticonservative exactly where the headline effect is borderline); CV folds are
grouped by question (no same-question train/test leakage); top-q sets under the
integer-valued TRUE attributes are tie-broken by averaging over jitter draws;
the regime verdict must hold across all NSEEDS judge refits, not one draw.

Usage: HF_HOME=... python -m analysis.corr_taxonomy --out <dir> [--model ...]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from drifted_grader.metrics import rank_correlation
from analysis.llm_judge_demo import embed, _std, _pca, _fit_lb
from analysis.semantic_blindspot import _surface

READABLE = ["length", "avg_word_len", "digit_frac"]   # order matches _surface
QS = (0.01, 0.05, 0.10, 0.20)
NBOOT = 400
NJIT = 10
NSEEDS = 5


def load_items_qid(cap=6000):
    """semantic_blindspot.load_items with a question id per item (identical
    construction and rng sequence, tuples extended with qid)."""
    import datasets
    d = datasets.load_dataset("truthful_qa", "generation", split="validation")
    pos, neg = [], []
    for qi, r in enumerate(d):
        q = r["question"]
        for a in r["correct_answers"]:
            if a and a.strip():
                pos.append((q + " " + a, a, 1, qi))
        for a in r["incorrect_answers"]:
            if a and a.strip():
                neg.append((q + " " + a, a, 0, qi))
    rng = np.random.default_rng(0)
    m = min(len(pos), len(neg), cap // 2)
    items = [pos[i] for i in rng.choice(len(pos), m, replace=False)] + \
            [neg[i] for i in rng.choice(len(neg), m, replace=False)]
    rng.shuffle(items)
    return items


def cv_ridge_predict(X, t, qid, folds=5, alphas=(1e0, 1e1, 1e2, 1e3, 1e4), seed=0):
    """Question-GROUPED cross-fitted ridge: whole questions assigned to folds, so
    no answer shares a question with its predictor's training set. Standardization
    and alpha selection strictly inside the training part."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    uq = rng.permutation(np.unique(qid))
    fold_of_q = {q: i % folds for i, q in enumerate(uq)}
    fold = np.array([fold_of_q[q] for q in qid])
    pred = np.empty(n)
    for f in range(folds):
        test = np.where(fold == f)[0]
        train = np.where(fold != f)[0]
        # inner split by question as well
        tr_q = rng.permutation(np.unique(qid[train]))
        va_qs = set(tr_q[: max(1, len(tr_q) // 5)].tolist())
        va = train[np.isin(qid[train], list(va_qs))]
        tr = train[~np.isin(qid[train], list(va_qs))]
        Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-8
        tm = t[tr].mean()
        def solve(idx, a):
            Z = (X[idx] - Xm) / Xs
            A = Z.T @ Z + a * np.eye(d)
            return np.linalg.solve(A, Z.T @ (t[idx] - tm))
        best_a, best_err = alphas[0], np.inf
        for a in alphas:
            w = solve(tr, a)
            err = np.mean((((X[va] - Xm) / Xs) @ w + tm - t[va]) ** 2)
            if err < best_err:
                best_a, best_err = a, err
        w = solve(train, best_a)
        pred[test] = ((X[test] - Xm) / Xs) @ w + tm
    return pred


_MEMBERS = {}


def _members(qid):
    """Per-question item-index arrays, built ONCE per qid array (cached by id)."""
    key = id(qid)
    if key not in _MEMBERS:
        uq = np.unique(qid)
        _MEMBERS[key] = (uq, [np.where(qid == q)[0] for q in uq])
    return _MEMBERS[key]


def _cluster_idx(qid, rng):
    """One cluster-bootstrap resample: draw questions with replacement, return
    the concatenated item indices (index precomputed, so O(#questions))."""
    uq, mem = _members(qid)
    draw = rng.integers(0, len(uq), len(uq))
    return np.concatenate([mem[i] for i in draw])


def boot_corr(x, y, qid, rng, nb=NBOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    bs = [np.corrcoef(x[i], y[i])[0, 1]
          for i in (_cluster_idx(qid, rng) for _ in range(nb))]
    return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def topq(scores, iscorr, q, qid, rng, nb=NBOOT, jitter=False):
    """Mean correctness of the top-q fraction, cluster-bootstrap CI over
    questions. jitter=True averages the POINT estimate over NJIT tiny-noise
    tie-breaks (integer scores have heavy ties); bootstrap resamples use a
    single tie-break each (CI width is dominated by resampling, not ties)."""
    scores = np.asarray(scores, float); iscorr = np.asarray(iscorr, float)
    eps = 1e-9 * (np.std(scores) + 1e-12)

    def stat(s, c, r, nj):
        kq = max(int(q * len(s)), 5)
        if nj <= 1:
            sc = s + (eps * r.standard_normal(len(s)) if jitter else 0.0)
            return float(c[np.argsort(-sc)[:kq]].mean())
        return float(np.mean([c[np.argsort(-(s + eps * r.standard_normal(len(s))))[:kq]].mean()
                              for _ in range(nj)]))

    m = stat(scores, iscorr, rng, NJIT if jitter else 1)
    if nb <= 0:                     # point estimate only (per-seed judge points)
        return m, float("nan"), float("nan")
    bs = [stat(scores[i], iscorr[i], rng, 1)
          for i in (_cluster_idx(qid, rng) for _ in range(nb))]
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--npairs", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    items = load_items_qid()
    texts = [t for t, a, c, qi in items]
    qid = np.array([qi for t, a, c, qi in items])
    A_raw = np.array([[c] + _surface(a) for t, a, c, qi in items], float)
    iscorr = A_raw[:, 0]
    base = float(iscorr.mean())
    print(f"items {len(items)}  questions {len(np.unique(qid))}  "
          f"base correct-rate {base:.3f}", flush=True)
    E = _std(embed(texts, args.model))
    Epca = _std(_pca(E, args.pca))
    A = _std(A_raw)
    # NOTE: kept byte-identical to semantic_blindspot.py:71 (np.diff(...,1) --
    # coordinate diffs, not pairwise-item distances) for stage-0 comparability.
    scaleE = 1.0 / (np.sqrt((np.diff(Epca[rng.integers(0, len(Epca), (5000, 2))], 1)[:, 0] ** 2)
                            .sum(1).mean()) + 1e-8)

    out = dict(base_rate=base, model=args.model, n=len(items),
               n_questions=int(len(np.unique(qid))), attrs={})
    lines = [f"# Correlation-sign taxonomy ({args.model}, TruthfulQA, n={len(items)}, "
             f"{len(np.unique(qid))} questions, base correct-rate {base:.2f})\n",
             "All CIs are CLUSTER bootstraps over questions; CV grouped by question; "
             f"judge column refit over {NSEEDS} independent preference draws.\n",
             "For each READABLE surface attribute s: does selecting on s (as the "
             "world defines it / as the judge reads it / as the judge's "
             "preference-fit bound acts on it) drag factual correctness up, "
             "nowhere, or down?\n"]

    for ai, name in enumerate(READABLE, start=1):
        s_true = A_raw[:, ai]
        print(f"== {name} ==", flush=True)
        # (2) the judge's readout of s (full E + Epca control for the PCA channel)
        s_hat = cv_ridge_predict(E, s_true, qid, seed=ai)
        s_hat_pca = cv_ridge_predict(Epca, s_true, qid, seed=ai + 50)
        r2 = lambda p: 1.0 - np.mean((p - s_true) ** 2) / (np.var(s_true) + 1e-12)
        read_r2, read_r2_pca = r2(s_hat), r2(s_hat_pca)
        r_true = boot_corr(s_true, iscorr, qid, rng)
        r_hat = boot_corr(s_hat, iscorr, qid, rng)
        r_hat_pca = boot_corr(s_hat_pca, iscorr, qid, rng)

        # (3c) the deployed object, refit over NSEEDS independent draws
        wt = np.zeros(A.shape[1]); wt[ai] = 1.0
        vt = A @ wt
        lbs, ranks = [], []
        for r_i in range(NSEEDS):
            prng = np.random.default_rng(ai * 1000 + r_i)
            ij = prng.integers(0, len(A), (args.npairs, 2)); ij = ij[ij[:, 0] != ij[:, 1]]
            y = (prng.random(len(ij)) <
                 1.0 / (1.0 + np.exp(-10.0 * ((A[ij[:, 0]] - A[ij[:, 1]]) @ wt)))).astype(float)
            rr = np.random.default_rng(ai + 100 + r_i)
            lb = _fit_lb(Epca[ij[:, 0]] - Epca[ij[:, 1]], y, Epca, scaleE, rr)
            lbs.append(lb)
            ranks.append(rank_correlation(lb.tolist(), vt.tolist()))
        rank_j = float(np.mean(ranks))

        rec = dict(readout_r2=round(float(read_r2), 3),
                   readout_r2_pca=round(float(read_r2_pca), 3),
                   corr_true=[round(v, 3) for v in r_true],
                   corr_readout=[round(v, 3) for v in r_hat],
                   corr_readout_pca=[round(v, 3) for v in r_hat_pca],
                   judge_rank_on_s=round(rank_j, 3),
                   judge_rank_seeds=[round(v, 3) for v in ranks], topq={})
        lines += [f"\n## {name}  (readout R^2 = {read_r2:.2f} full / "
                  f"{read_r2_pca:.2f} pca-{args.pca}; judge ranks this standard at "
                  f"{rank_j:+.2f} across {NSEEDS} refits)\n",
                  f"- corr(TRUE {name}, correct)        = {r_true[0]:+.3f} "
                  f"[{r_true[1]:+.3f}, {r_true[2]:+.3f}]",
                  f"- corr(READOUT {name}, correct)     = {r_hat[0]:+.3f} "
                  f"[{r_hat[1]:+.3f}, {r_hat[2]:+.3f}]",
                  f"- corr(READOUT-pca {name}, correct) = {r_hat_pca[0]:+.3f} "
                  f"[{r_hat_pca[1]:+.3f}, {r_hat_pca[2]:+.3f}]\n",
                  "| top-q | by TRUE s | by READOUT s_hat | by JUDGE bound "
                  "(seed-0 CI; per-seed points) |",
                  "|---|---|---|---|"]
        for q in QS:
            tt = topq(s_true, iscorr, q, qid, rng, jitter=True)
            th = topq(s_hat, iscorr, q, qid, rng, jitter=True)
            tj0 = topq(lbs[0], iscorr, q, qid, rng)
            pts = [topq(lb, iscorr, q, qid, rng, nb=0)[0] if s_i else tj0[0]
                   for s_i, lb in enumerate(lbs)]
            rec["topq"][str(q)] = dict(true=[round(v, 3) for v in tt],
                                       readout=[round(v, 3) for v in th],
                                       judge_ci=[round(v, 3) for v in tj0],
                                       judge_seeds=[round(v, 3) for v in pts])
            def fmt(v):
                mark = "**" if (v[2] < base or v[1] > base) else ""
                return f"{mark}{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]{mark}"
            lines.append(f"| {int(q*100)}% | {fmt(tt)} | {fmt(th)} | {fmt(tj0)} "
                         f"({', '.join(f'{p:.2f}' for p in pts)}) |")

        # regime verdict at q=10%: clustered CI (seed 0) AND unanimity across refits
        v10 = rec["topq"]["0.1"]["judge_ci"]
        pts10 = rec["topq"]["0.1"]["judge_seeds"]
        if v10[2] < base and all(p < base for p in pts10):
            regime = "NEGATIVE (Goodhart)"
        elif v10[1] > base and all(p > base for p in pts10):
            regime = "POSITIVE (drag-along)"
        else:
            regime = "UNCORRELATED / not separable from flat"
        rec["regime"] = regime
        lines.append(f"\n**Regime for {name}: {regime}** (judge top-10% correctness "
                     f"{v10[0]:.2f} vs base {base:.2f}; clustered CI "
                     f"[{v10[1]:.2f},{v10[2]:.2f}]; refit points "
                     f"{', '.join(f'{p:.2f}' for p in pts10)})")
        dec = np.array_split(np.argsort(s_hat), 10)
        rec["decile_curve"] = [round(float(iscorr[d].mean()), 3) for d in dec]
        out["attrs"][name] = rec
        print(json.dumps({k: v for k, v in rec.items() if k != "decile_curve"})[:500],
              flush=True)

    lines.append("\n## Reading the verdict\n"
                 "- The dangerous condition (per the taxonomy) is low coverage of "
                 "correctness PLUS a NEGATIVE sign on a readable attribute the "
                 "standard rewards: optimizing the readable criterion then actively "
                 "destroys correctness, and no uncertainty flag fires.\n"
                 "- POSITIVE means proxy optimization drags correctness along (why "
                 "rich judges look robust); UNCORRELATED means a pure missed "
                 "opportunity growing with the weight on correctness.\n"
                 "- Compare TRUE vs READOUT vs READOUT-pca rows: a sign flip at "
                 "READOUT (full E) means the judge's *reading* of the attribute "
                 "carries the anti-correctness direction; a flip only at "
                 "READOUT-pca/JUDGE means the PCA-128 truncation contributes.\n")

    json.dump(out, open(os.path.join(args.out, "corrtax.json"), "w"), indent=1)
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "results.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
