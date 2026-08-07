"""The practically-meaningful demonstration: when an evaluation standard DRIFTS ONTO a criterion
the frozen LLM judge cannot represent (a pretraining blind spot), VaR-based selection silently
fails -- and this IS drift-driven, unlike the mixed-criteria demo.

We use measured Qwen blind spots: 'hedging' is ~unrepresented (ridge R^2 0.01, MLP 0.12) while
'specificity'/'formality' are well represented (R^2 ~0.75). Two scenarios:
  toward_blind   : standard drifts specificity(covered) -> hedging(BLIND)   [expect regret RISES]
  toward_covered : standard drifts specificity(covered) -> formality(covered) [control, flat]
Report selection regret per drift stage for the frozen Qwen judge vs an oracle (true attrs).
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
from drifted_grader.metrics import rank_correlation
from analysis.llm_judge_demo import attrs, load_corpus, embed, _std, _pca, _fit_lb, ATTR_NAMES

# indices into ATTR_NAMES = [length, sentiment, specificity, hedging, formality, structure]
SPEC, HEDGE, FORM = 2, 3, 4


def _onehot(i, d=6):
    v = np.zeros(d); v[i] = 1.0; return v


def _regret(lb, vt):
    hi, lo = vt.max(), vt.min()
    return float((hi - vt[int(np.argmax(lb))]) / (hi - lo + 1e-9))


def run_scenario(A, Epca, scaleE, scaleA, w0, wE, stages, npairs, rng):
    out = {"oracle": [], "qwen_frozen": [], "rank_qwen": []}
    for s in range(stages):
        frac = s / (stages - 1); wt = (1 - frac) * w0 + frac * wE; wt /= (np.linalg.norm(wt) + 1e-9)
        vt = A @ wt
        ij = rng.integers(0, len(A), (npairs, 2)); ij = ij[ij[:, 0] != ij[:, 1]]
        y = (rng.random(len(ij)) < 1.0 / (1.0 + np.exp(-10.0 * ((A[ij[:, 0]] - A[ij[:, 1]]) @ wt)))).astype(float)
        rr = np.random.default_rng(s + 1)
        lb_o = _fit_lb(A[ij[:, 0]] - A[ij[:, 1]], y, A, scaleA, rr)
        lb_q = _fit_lb(Epca[ij[:, 0]] - Epca[ij[:, 1]], y, Epca, scaleE, rr)
        out["oracle"].append(round(_regret(lb_o, vt), 3))
        out["qwen_frozen"].append(round(_regret(lb_q, vt), 3))
        out["rank_qwen"].append(round(rank_correlation(lb_q.tolist(), vt.tolist()), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--stages", type=int, default=9)
    ap.add_argument("--npairs", type=int, default=3000); ap.add_argument("--pca", type=int, default=128)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(0)
    texts = load_corpus(args.n)
    A = _std(np.stack([attrs(t) for t in texts]))
    E = _std(embed(texts, args.model)); Epca = _std(_pca(E, args.pca))
    scaleA = 1.0 / (np.sqrt((np.diff(A[rng.integers(0, len(A), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    scaleE = 1.0 / (np.sqrt((np.diff(Epca[rng.integers(0, len(Epca), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    scen = {
        "toward_blind (specificity->hedging)": (_onehot(SPEC), _onehot(HEDGE)),
        "toward_covered (specificity->formality)": (_onehot(FORM) * 0 + _onehot(SPEC), _onehot(FORM)),
    }
    res = {name: run_scenario(A, Epca, scaleE, scaleA, w0, wE, args.stages, args.npairs, np.random.default_rng(7))
           for name, (w0, wE) in scen.items()}
    json.dump(res, open(os.path.join(args.out, "blindspot.json"), "w"), indent=2)
    L = [f"# Drift ONTO a frozen-judge blind spot ({args.model})\n",
         "Qwen blind spot: hedging (R^2~0.01-0.12); covered: specificity/formality (R^2~0.75). "
         "Selection regret per drift stage (frac 0->1). If drifting onto the blind criterion, "
         "the frozen judge's regret RISES while the oracle stays ~0.\n"]
    for name, r in res.items():
        L.append(f"\n## {name}\n| metric | " + " | ".join(f"s{i}" for i in range(args.stages)) + " |")
        L.append("|" + "---|" * (1 + args.stages))
        L.append("| oracle regret | " + " | ".join(f"{x:.2f}" for x in r["oracle"]) + " |")
        L.append("| qwen_frozen regret | " + " | ".join(f"{x:.2f}" for x in r["qwen_frozen"]) + " |")
        L.append("| qwen rank | " + " | ".join(f"{x:+.2f}" for x in r["rank_qwen"]) + " |")
        slope = np.polyfit(np.arange(args.stages), r["qwen_frozen"], 1)[0]
        L.append(f"\nqwen_frozen regret slope = {slope:+.4f}/stage "
                 f"({'RISES onto blind spot' if slope > 0.01 else 'flat'}).\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
