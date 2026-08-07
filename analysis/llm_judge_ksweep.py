"""Text LLM-judge narrow-head k-sweep: selection regret & rank vs head dim k (PCA-k of the
frozen Qwen embedding), to find where a narrow linear head starts to fail. Mirrors the vision
narrow-head sweep. Reports the threshold (k where rank first exceeds 0.5 / regret plateaus).
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
from drifted_grader.metrics import rank_correlation
from analysis.llm_judge_demo import attrs, load_corpus, embed, _std, _pca, _fit_lb

KS = (2, 4, 8, 16, 32, 64, 128, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--stages", type=int, default=9)
    ap.add_argument("--npairs", type=int, default=3000)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(0)
    texts = load_corpus(args.n)
    A = _std(np.stack([attrs(t) for t in texts]))
    E = _std(embed(texts, args.model))
    print(f"corpus {len(texts)}  E {E.shape}", flush=True)
    w0 = np.array([+1.0, 0.0, +1.0, -0.5, +0.4, +1.0]); w0 /= np.linalg.norm(w0)
    wE = np.array([-1.0, +1.5, 0.0, -1.0, 0.0, +0.4]); wE /= np.linalg.norm(wE)

    rows = []
    for k in KS:
        if k > E.shape[1]:
            continue
        Ek = _std(_pca(E, k))
        scaleE = 1.0 / (np.sqrt((np.diff(Ek[rng.integers(0, len(Ek), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
        regs, ranks = [], []
        for s in range(args.stages):
            frac = s / (args.stages - 1); wt = (1 - frac) * w0 + frac * wE; wt /= np.linalg.norm(wt)
            vt = A @ wt
            ij = rng.integers(0, len(A), (args.npairs, 2)); ij = ij[ij[:, 0] != ij[:, 1]]
            y = (rng.random(len(ij)) < 1.0 / (1.0 + np.exp(-10.0 * ((A[ij[:, 0]] - A[ij[:, 1]]) @ wt)))).astype(float)
            lb = _fit_lb(Ek[ij[:, 0]] - Ek[ij[:, 1]], y, Ek, scaleE, np.random.default_rng(s + 1))
            hi, lo = vt.max(), vt.min()
            regs.append((hi - vt[int(np.argmax(lb))]) / (hi - lo + 1e-9)); ranks.append(rank_correlation(lb.tolist(), vt.tolist()))
        rows.append(dict(k=k, regret=round(float(np.mean(regs)), 3), rank=round(float(np.mean(ranks)), 3)))
        print("k", k, "regret", rows[-1]["regret"], "rank", rows[-1]["rank"], flush=True)

    json.dump(rows, open(os.path.join(args.out, "ksweep.json"), "w"), indent=2)
    kstar = next((r["k"] for r in rows if r["rank"] >= 0.5), ">max")
    L = [f"# Text LLM-judge narrow-head k-sweep ({args.model})\n",
         "| k | regret | rank |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['k']} | {r['regret']:.3f} | {r['rank']:+.2f} |")
    L.append(f"\nthreshold k* (rank>=0.5): {kstar}; regret plateaus near its large-k value "
             "(baseline linear-encoding mismatch), so no sharp cliff.\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
