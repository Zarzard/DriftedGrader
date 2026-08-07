"""Aggregate the §6 scaling sweep into a coverage-vs-scale table + figure.

Reads llm_judge_drift/scaling/<TAG>/{semantic/sem.json, coverage_law/law.json} for
each Qwen size and reports, vs. parameter count:
  - factual-correctness coverage (TruthfulQA is_correct: ridge + MLP R^2)  [headline]
  - the coverage-law Spearman (does coverage->ranking hold at every scale?)
  - per-attribute coverage of the 6 interpretable attributes
The headline question: does correctness coverage rise with scale or plateau?

Run: python -m analysis.scaling_aggregate --root <.../llm_judge_drift/scaling> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0, "32B": 32.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/uufs/chpc.utah.edu/common/home/dbrown-group1/"
                    "zifan/llm_judge_drift/scaling")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.root
    os.makedirs(out, exist_ok=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        tag = os.path.basename(d)
        if tag not in PARAMS:
            continue
        sem_p = os.path.join(d, "semantic", "sem.json")
        law_p = os.path.join(d, "coverage_law", "law.json")
        row = {"tag": tag, "params": PARAMS[tag]}
        if os.path.exists(sem_p):
            sem = json.load(open(sem_p))
            cov = {c["attr"]: c for c in sem["coverage"]}
            row["correct_ridge"] = cov.get("is_correct", {}).get("ridge_r2")
            row["correct_mlp"] = cov.get("is_correct", {}).get("mlp_r2")
            # correctness ranking degradation across the drift stages (first->last)
            ps = sem.get("per_stage", [])
            if ps:
                row["correct_rank_start"] = ps[0]["judge_rank"]
                row["correct_rank_end"] = ps[-1]["judge_rank"]
        if os.path.exists(law_p):
            law = json.load(open(law_p))
            row["law_spearman"] = law.get("spearman_covR2_vs_judgerank")
            row["law_null_r2"] = law.get("mean_null_r2")
            # the 6 pure attributes are the first 6 rows of law.json
            row["attr_cov6"] = [r["covR2"] for r in law["rows"][:6]]
        rows.append(row)
    rows.sort(key=lambda r: r["params"])
    if not rows:
        print("no per-model results found under", args.root); return

    json.dump(rows, open(os.path.join(out, "scaling.json"), "w"), indent=1)

    L = ["# §6 Coverage vs. scale (Qwen2.5 family, frozen judge)\n",
         "Headline: does factual-correctness coverage improve with model size, or "
         "plateau? (Flat = a capability the field cannot buy with scale.)\n",
         "| params (B) | correctness cov (ridge) | correctness cov (MLP) | "
         "correctness rank (drift start->end) | coverage-law Spearman | null R^2 |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        def g(k, f="{:.2f}"):
            v = r.get(k)
            return f.format(v) if isinstance(v, (int, float)) else "n/a"
        rs = (f"{r.get('correct_rank_start'):+.2f}->{r.get('correct_rank_end'):+.2f}"
              if r.get("correct_rank_start") is not None else "n/a")
        L.append(f"| {r['params']:g} | {g('correct_ridge')} | {g('correct_mlp')} | "
                 f"{rs} | {g('law_spearman','{:+.2f}')} | {g('law_null_r2','{:+.2f}')} |")

    L += ["\n## Per-attribute coverage vs. scale (6 interpretable attributes)\n",
          "| params (B) | length | sentiment | specificity | hedging | formality | structure |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        a = r.get("attr_cov6")
        L.append(f"| {r['params']:g} | " + " | ".join(f"{x:.2f}" for x in a) + " |"
                 if a else f"| {r['params']:g} | n/a |")

    L.append("\n## Reading\n"
             "- correctness cov FLAT across scale -> a representational blind spot the "
             "field cannot buy its way out of (the stronger paper).\n"
             "- correctness cov RISING -> coverage is a capability that scales; the audit "
             "then tells you the size threshold for a given criterion.\n"
             "- coverage-law Spearman staying high at every scale -> the law is not an "
             "artifact of one model.\n")
    md = "\n".join(L) + "\n"
    open(os.path.join(out, "results.md"), "w").write(md)
    print(md)

    # figure: correctness coverage + law spearman vs params (log-x)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = [r["params"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, lab, mk in (("correct_ridge", "factual correctness (ridge R²)", "-o"),
                             ("correct_mlp", "factual correctness (MLP R²)", "--s"),
                             ("law_spearman", "coverage-law Spearman", "-^")):
            y = [r.get(key) for r in rows]
            xy = [(p, v) for p, v in zip(P, y) if isinstance(v, (int, float))]
            if xy:
                ax.plot([a for a, _ in xy], [b for _, b in xy], mk, label=lab)
        ax.set_xscale("log"); ax.set_xticks(P); ax.set_xticklabels([f"{p:g}" for p in P])
        ax.set_xlabel("model size (B params, log)"); ax.set_ylabel("coverage / correlation")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.set_title("Does a frozen judge's coverage of factual correctness scale?")
        ax.legend(fontsize=8)
        fig.tight_layout(); fp = os.path.join(out, "fig_scaling.png")
        fig.savefig(fp, dpi=150); print("wrote", fp)
    except Exception as e:
        print("skip figure:", e)


if __name__ == "__main__":
    main()
