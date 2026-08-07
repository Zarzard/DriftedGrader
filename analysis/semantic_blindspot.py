"""Semantic blind spot: is a frozen LLM judge blind to FACTUAL CORRECTNESS, and does the
selection fail (reward-hacking) when the evaluation standard drifts onto correctness?

TruthfulQA gives human-written correct & incorrect answers per question -> ground-truth
is_correct in {0,1}. We embed (question + answer) with the frozen judge, measure whether its
representation linearly/nonlinearly encodes correctness (R^2), then drift the standard from a
covered surface criterion (answer length) onto correctness. Headline metric: the fraction of the
judge's TOP-k picks that are actually correct, vs an oracle with the true attribute.

Usage: HF_HOME=... python -m analysis.semantic_blindspot --out <dir> [--model ...]
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
from drifted_grader.metrics import rank_correlation
from analysis.llm_judge_demo import embed, _std, _pca, _fit_lb
from analysis.llm_judge_refine import ridge_r2
from analysis.nonlinear_probe import mlp_r2

ATTRS = ["is_correct", "length", "avg_word_len", "digit_frac"]
CORRECT, LENGTH = 0, 1


def _surface(ans):
    w = ans.split(); n = max(len(w), 1)
    return [min(len(w) / 40.0, 3.0), sum(len(x) for x in w) / n / 6.0, sum(c.isdigit() for c in ans) / max(len(ans), 1) * 50]


def load_items(cap=6000):
    import datasets
    d = datasets.load_dataset("truthful_qa", "generation", split="validation")
    pos, neg = [], []
    for r in d:
        q = r["question"]
        for a in r["correct_answers"]:
            if a and a.strip():
                pos.append((q + " " + a, a, 1))
        for a in r["incorrect_answers"]:
            if a and a.strip():
                neg.append((q + " " + a, a, 0))
    rng = np.random.default_rng(0)
    m = min(len(pos), len(neg), cap // 2)
    items = [pos[i] for i in rng.choice(len(pos), m, replace=False)] + [neg[i] for i in rng.choice(len(neg), m, replace=False)]
    rng.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--stages", type=int, default=9); ap.add_argument("--npairs", type=int, default=4000)
    ap.add_argument("--pca", type=int, default=128); ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(0)
    items = load_items()
    texts = [t for t, a, c in items]
    A_raw = np.array([[c] + _surface(a) for t, a, c in items], float)
    base_rate = float(A_raw[:, CORRECT].mean())
    A = _std(A_raw)
    print(f"items {len(items)}  base correct-rate {base_rate:.2f}", flush=True)
    E = _std(embed(texts, args.model)); Epca = _std(_pca(E, args.pca))
    print(f"embeddings {E.shape}", flush=True)

    # (1) coverage of each criterion
    cov = []
    for i, name in enumerate(ATTRS):
        cov.append(dict(attr=name, ridge_r2=round(ridge_r2(E, A_raw[:, i]), 3), mlp_r2=round(mlp_r2(E, A_raw[:, i]), 3)))
        print(cov[-1], flush=True)

    # (2) drift standard: length (covered) -> is_correct (semantic). Report top-k correctness rate.
    scaleA = 1.0 / (np.sqrt((np.diff(A[rng.integers(0, len(A), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    scaleE = 1.0 / (np.sqrt((np.diff(Epca[rng.integers(0, len(Epca), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    w0 = np.zeros(4); w0[LENGTH] = 1.0
    wE = np.zeros(4); wE[CORRECT] = 1.0
    iscorr = A_raw[:, CORRECT]
    per = []
    for s in range(args.stages):
        frac = s / (args.stages - 1); wt = (1 - frac) * w0 + frac * wE; wt /= np.linalg.norm(wt)
        vt = A @ wt
        ij = rng.integers(0, len(A), (args.npairs, 2)); ij = ij[ij[:, 0] != ij[:, 1]]
        y = (rng.random(len(ij)) < 1.0 / (1.0 + np.exp(-10.0 * ((A[ij[:, 0]] - A[ij[:, 1]]) @ wt)))).astype(float)
        rr = np.random.default_rng(s + 1)
        lb_o = _fit_lb(A[ij[:, 0]] - A[ij[:, 1]], y, A, scaleA, rr)
        lb_q = _fit_lb(Epca[ij[:, 0]] - Epca[ij[:, 1]], y, Epca, scaleE, rr)
        tk = args.topk
        per.append(dict(stage=s, frac=round(frac, 2),
                        oracle_topk_correct=round(float(iscorr[np.argsort(-lb_o)[:tk]].mean()), 3),
                        judge_topk_correct=round(float(iscorr[np.argsort(-lb_q)[:tk]].mean()), 3),
                        judge_rank=round(rank_correlation(lb_q.tolist(), vt.tolist()), 3)))
        print(per[-1], flush=True)

    json.dump({"coverage": cov, "per_stage": per, "base_rate": base_rate}, open(os.path.join(args.out, "sem.json"), "w"), indent=2)
    L = [f"# Semantic blind spot: does a frozen LLM judge lose FACTUAL CORRECTNESS? ({args.model})\n",
         f"TruthfulQA, {len(items)} (Q,A) items, base correct-rate {base_rate:.2f}.\n",
         "## (1) Representation coverage of each criterion (held-out R^2)\n",
         "| criterion | ridge R^2 | MLP R^2 |", "|---|---|---|"]
    for c in cov:
        L.append(f"| {c['attr']} | {c['ridge_r2']:.3f} | {c['mlp_r2']:.3f} |")
    L.append("\n## (2) Standard drifts length(covered) -> correctness. Top-%d picks that are actually correct:\n" % args.topk)
    L.append("| stage | frac | oracle top-k correct | JUDGE top-k correct | judge rank |")
    L.append("|---|---|---|---|---|")
    for p in per:
        L.append(f"| {p['stage']} | {p['frac']} | {p['oracle_topk_correct']:.2f} | **{p['judge_topk_correct']:.2f}** | {p['judge_rank']:+.2f} |")
    L.append(f"\nbase correct-rate = {base_rate:.2f}. If correctness is a blind spot, at frac=1 the JUDGE's "
             "top-k correct-rate collapses toward the base rate (chance) while the oracle's -> ~1.0 "
             "= the judge selects confident-but-wrong answers (reward hacking under drift onto a blind criterion).\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
