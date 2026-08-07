"""Is the baseline mismatch a LINEARITY limit or genuine absence? Compare a linear (ridge) probe
vs a nonlinear (MLP) probe recovering the drift direction a.u and each attribute from the frozen
LLM embedding. If MLP R^2 >> ridge R^2 -> a nonlinear head would sharpen the k-threshold; if
MLP ~ ridge -> the info is genuinely (not just linearly) missing.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
from analysis.llm_judge_demo import attrs, load_corpus, embed, _std, ATTR_NAMES
from analysis.llm_judge_refine import ridge_r2


def mlp_r2(E, t, hidden=256, epochs=300, seed=0):
    import torch
    rng = np.random.default_rng(seed); n = len(E); idx = rng.permutation(n)
    tr, te = idx[:int(0.7 * n)], idx[int(0.7 * n):]
    Xtr = torch.tensor(E[tr], dtype=torch.float32); ttr = torch.tensor(t[tr], dtype=torch.float32)
    Xte = torch.tensor(E[te], dtype=torch.float32); tte = t[te]
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(E.shape[1], hidden), torch.nn.ReLU(),
                            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); lf = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); l = lf(m(Xtr).squeeze(-1), ttr); l.backward(); opt.step()
    with torch.no_grad():
        pred = m(Xte).squeeze(-1).numpy()
    return float(1 - ((tte - pred) ** 2).sum() / (((tte - tte.mean()) ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    texts = load_corpus(args.n)
    A = _std(np.stack([attrs(t) for t in texts]))
    E = _std(embed(texts, args.model))
    w0 = np.array([+1.0, 0.0, +1.0, -0.5, +0.4, +1.0]); w0 /= np.linalg.norm(w0)
    wE = np.array([-1.0, +1.5, 0.0, -1.0, 0.0, +0.4]); wE /= np.linalg.norm(wE)
    u = wE - w0; u /= np.linalg.norm(u)
    targets = [("drift_u", A @ u)] + [(a, A[:, i]) for i, a in enumerate(ATTR_NAMES)]
    rows = []
    for name, t in targets:
        rows.append(dict(target=name, ridge_r2=round(ridge_r2(E, t), 3), mlp_r2=round(mlp_r2(E, t), 3)))
        print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(args.out, "probe.json"), "w"), indent=2)
    L = [f"# Linear (ridge) vs nonlinear (MLP) probe of the frozen LLM embedding ({args.model})\n",
         "MLP >> ridge => attributes are nonlinearly encoded (a nonlinear head would help / sharpen "
         "the k-threshold); MLP ~ ridge => info genuinely (not just linearly) missing.\n",
         "| target | ridge R^2 | MLP R^2 | gain |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['target']} | {r['ridge_r2']:.3f} | {r['mlp_r2']:.3f} | {r['mlp_r2']-r['ridge_r2']:+.3f} |")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
