"""Layer-depth coverage sweep: does a MIDDLE layer of the frozen judge read factual
correctness better than the last layer (ridge R^2 ~0.20)?

The scaling sweep used mean-pooled LAST-layer hidden states. The review's point: if a
mid-layer linear readout covers correctness much better, then the "correctness blind
spot" is partly a POOLING/LAYER choice, not a capacity limit -- directly actionable
(and it softens the "wall" claim). If correctness coverage is low at EVERY layer, the
wall is robust.

For one model, extract mean-pooled hidden states at a sampled set of layers and, per
layer, measure coverage (held-out ridge + MLP R^2) of:
  - is_correct (TruthfulQA, the headline)
  - the 6 interpretable attributes (length/sentiment/specificity/hedging/formality/
    structure), computed on the same texts, for context.

Usage: HF_HOME=... python -m analysis.layer_sweep --model Qwen/Qwen2.5-3B-Instruct --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.llm_judge_demo import attrs, ATTR_NAMES, _std
from analysis.llm_judge_refine import ridge_r2
from analysis.nonlinear_probe import mlp_r2
from analysis.semantic_blindspot import load_items


def embed_layers(texts, model_name, n_sample=9, batch=16):
    """Mean-pooled hidden state at a sampled set of layers. Returns
    ({layer_index: [N, H]}, total_layers). Layer 0 = embedding output; L = final."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16,
                                      device_map="cuda", output_hidden_states=True)
    model.eval()
    L = model.config.num_hidden_layers
    layer_idxs = sorted(set(np.linspace(0, L, n_sample).astype(int).tolist()))
    acc = {li: [] for li in layer_idxs}
    post = []                                                     # post-final-norm (last_hidden_state)
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            out = model(**enc)
            hs = out.hidden_states                                # tuple len L+1 (pre-final-norm at [-1])
            m = enc["attention_mask"].unsqueeze(-1).float()
            denom = m.sum(1).clamp(min=1)
            for li in layer_idxs:
                acc[li].append(((hs[li] * m).sum(1) / denom).float().cpu().numpy())
            post.append(((out.last_hidden_state * m).sum(1) / denom).float().cpu().numpy())
    return ({li: np.concatenate(v, 0) for li, v in acc.items()},
            L, np.concatenate(post, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_sample", type=int, default=9)
    ap.add_argument("--cap", type=int, default=6000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    items = load_items(cap=args.cap)                              # (text, answer, is_correct)
    texts = [t for t, a, c in items]
    iscorr = np.array([c for t, a, c in items], float)
    Attr = _std(np.stack([attrs(t) for t in texts]))             # [N,6] interpretable attrs
    print(f"items {len(texts)}  base correct-rate {iscorr.mean():.3f}", flush=True)

    layers, L, postnorm = embed_layers(texts, args.model, n_sample=args.n_sample)
    print(f"model {args.model}  total layers {L}  sampled {sorted(layers)}", flush=True)

    rows = []
    for li in sorted(layers):
        E = _std(layers[li])              # standardize to match §3/§6/§7 (ridge R² is scale-sensitive)
        rec = {"layer": li, "depth_frac": round(li / L, 3),
               "correct_ridge": round(float(ridge_r2(E, iscorr)), 3),
               "correct_mlp": round(float(mlp_r2(E, iscorr)), 3)}
        for j, name in enumerate(ATTR_NAMES):
            rec[f"cov_{name}"] = round(float(ridge_r2(E, Attr[:, j])), 3)
        rows.append(rec)
        print(rec, flush=True)
    # post-final-norm last_hidden_state = the DEPLOYED final representation (matches §6);
    # distinguishes the pre-norm hidden_states[-1] from the post-norm deployed readout.
    post_correct = round(float(ridge_r2(_std(postnorm), iscorr)), 3)
    print(f"POST-NORM final (last_hidden_state, deployed) correctness ridge R² = {post_correct} "
          f"(vs pre-norm hidden_states[-1] = {rows[-1]['correct_ridge']})", flush=True)

    best = max(rows, key=lambda r: r["correct_ridge"])
    last = [r for r in rows if r["layer"] == L][0]
    out = dict(model=args.model, total_layers=L, base_rate=float(iscorr.mean()),
               rows=rows, best_correct_layer=best["layer"],
               best_correct_ridge=best["correct_ridge"],
               last_correct_ridge_prenorm=last["correct_ridge"],
               last_correct_ridge_postnorm=post_correct)
    json.dump(out, open(os.path.join(args.out, "layers.json"), "w"), indent=1)

    L_ = ["# Layer-depth coverage sweep ({}, {} layers)\n".format(args.model, L),
          f"Does a mid-layer read factual correctness better than the last layer? "
          f"base correct-rate {iscorr.mean():.2f}.\n",
          "| layer | depth | correctness (ridge) | correctness (MLP) | length | "
          "sentiment | specificity | hedging | formality | structure |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L_.append(f"| {r['layer']} | {r['depth_frac']:.2f} | **{r['correct_ridge']:.2f}** "
                  f"| {r['correct_mlp']:.2f} | " +
                  " | ".join(f"{r[f'cov_{n}']:.2f}" for n in ATTR_NAMES) + " |")
    verdict = ("MID-LAYER READS CORRECTNESS BETTER -> the blind spot is partly a "
               "layer/pooling choice, not pure capacity"
               if best["layer"] != L and best["correct_ridge"] - last["correct_ridge"] > 0.05
               else "correctness coverage is ~flat across depth -> the blind spot is "
               "robust to layer choice (a real capacity/representation limit)")
    L_.append(f"\n**Best correctness coverage: layer {best['layer']} "
              f"(depth {best['depth_frac'] if False else round(best['layer']/L,2)}), "
              f"ridge R²={best['correct_ridge']:.2f}, vs last-layer {last['correct_ridge']:.2f}.**")
    L_.append(f"\n**Verdict:** {verdict}.\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [r["layer"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(x, [r["correct_ridge"] for r in rows], "-o", label="factual correctness (ridge)")
        ax.plot(x, [r["correct_mlp"] for r in rows], "--s", label="factual correctness (MLP)")
        for n, c in (("specificity", "#0072B2"), ("hedging", "#D55E00")):
            ax.plot(x, [r[f"cov_{n}"] for r in rows], ":", alpha=0.7, label=f"{n} (ridge)")
        ax.axvline(L, color="k", lw=0.5, ls=":")
        ax.set_xlabel(f"layer (0=embedding, {L}=final)"); ax.set_ylabel("coverage R²")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"Coverage by layer depth ({args.model.split('/')[-1]})")
        fig.tight_layout(); fp = os.path.join(args.out, "fig_layers.png")
        fig.savefig(fp, dpi=150); print("wrote", fp)
    except Exception as e:
        print("skip figure:", e)


if __name__ == "__main__":
    main()
