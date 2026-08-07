"""LLM-as-judge under a DRIFTING evaluation standard (real frozen LLM representation).

The judge's features are a frozen LLM's hidden state e(x) of a response x. Real assistant
responses have interpretable attributes a(x) in R^d (length, sentiment, specificity, hedging,
formality, structure). The TRUE evaluation standard is g_t(x)=w_t . a(x), and w_t DRIFTS
(scheduled w0 -> w_end, e.g. "thoroughness" -> "conciseness/positivity"). We generate
Bradley-Terry preferences by the drifting grader, fit a reward posterior for alpha=0.05
VaR-based SELECTION over the response pool, and ask at each drift stage:
  - does the frozen LLM representation COVER the drifting standard?  cov_r2(a.u | e(x))
  - is VaR selection ROBUST, or does it fail like a low-capacity / drift-blind head?

Conditions (same drifting labels, different reward-model features):
  oracle_attr : fit on the true attributes a(x)         (well-specified; grader is linear in a)
  llm_frozen  : Bayesian last layer on the frozen LLM embedding e(x) (PCA-64)  -- the real judge
  llm_kproj   : narrow k=8 linear head on e(x)          (low-capacity frozen head, B-REX-like)
  attr_adv    : attrs but blind to the drift direction u (imposed worst case, reference)

Usage: HF_HOME=/scratch/.../huggingface python -m analysis.llm_judge_demo --out <dir> [--model ...] [--n 1500]
"""
from __future__ import annotations
import argparse, os, re, json
import numpy as np

from drifted_grader.posterior import RewardPosterior
from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import rank_correlation

BETAT, ALPHA, PRIOR, NS = 10.0, 0.05, 2.0, 4000
POS = set("good great excellent helpful clear useful nice love best wonderful positive happy easy effective strong".split())
NEG = set("bad poor unclear useless confusing hard difficult wrong negative sad hate worst terrible weak".split())
HEDGE = set("maybe perhaps might could possibly probably seems likely appears suggests generally often sometimes may".split())
ATTR_NAMES = ["length", "sentiment", "specificity", "hedging", "formality", "structure"]


def attrs(text):
    w = text.split(); n = max(len(w), 1); sents = max(len(re.split(r"[.!?]+", text)), 1)
    return np.array([
        min(len(w) / 100.0, 3.0),                                                  # length
        (sum(x.lower().strip(".,!?") in POS for x in w) - sum(x.lower().strip(".,!?") in NEG for x in w)) / n * 10,  # sentiment
        sum(c.isdigit() for c in text) / max(len(text), 1) * 50,                   # specificity (digits)
        sum(x.lower().strip(".,!?") in HEDGE for x in w) / n * 10,                 # hedging
        sum(1 for x in w if x[:1].isupper()) / n,                                  # formality/proper-noun proxy
        (text.count("\n") + text.count("- ") + text.count("* ") + text.count("?")) / sents,  # structure
    ], float)


def load_corpus(n):
    import datasets
    d = datasets.load_dataset("tatsu-lab/alpaca", split="train")
    texts, seen = [], set()
    for r in d:
        t = (r.get("output") or "").strip()
        wc = len(t.split())
        if 25 <= wc <= 180 and t[:40] not in seen:
            seen.add(t[:40]); texts.append(t)
        if len(texts) >= n:
            break
    return texts


def embed(texts, model_name, batch=16):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16,
                                      device_map="cuda", output_hidden_states=False)
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            bt = texts[i:i + batch]
            enc = tok(bt, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
            out = model(**enc)
            h = out.last_hidden_state                                  # [B,T,D]
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)                 # mean-pool
            embs.append(e.float().cpu().numpy())
    return np.concatenate(embs, 0)


def _pca(E, k):
    Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    return Ec @ Vt[:k].T


def _std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def _fit_lb(dPhi, y, F, scale, rng, P=None):
    X = dPhi if P is None else dPhi @ P.T
    Fe = F if P is None else F @ P.T
    post = RewardPosterior(X.shape[1])
    post.fit(X * scale, y, BETAT, prior_scale=PRIOR, n_samples=NS, rng=rng)
    return np.array([post.lower_bound(f * scale, ALPHA) for f in Fe])


def _r2(z, tgt):
    Z = np.hstack([z, np.ones((len(z), 1))]); c, *_ = np.linalg.lstsq(Z, tgt, rcond=None)
    p = Z @ c; return float(1 - ((tgt - p) ** 2).sum() / (((tgt - tgt.mean()) ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--stages", type=int, default=9)
    ap.add_argument("--npairs", type=int, default=3000)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--kproj", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    texts = load_corpus(args.n)
    A = _std(np.stack([attrs(t) for t in texts]))                      # [N,d] standardized attributes
    print(f"corpus {len(texts)}  attrs {A.shape}", flush=True)
    E = embed(texts, args.model)
    print(f"embeddings {E.shape}  (model {args.model})", flush=True)
    Epca = _std(_pca(E, args.pca))
    np.savez(os.path.join(args.out, "features.npz"), A=A, Epca=Epca)

    d = A.shape[1]
    w0 = np.array([+1.0, 0.0, +1.0, -0.5, +0.4, +1.0]); w0 /= np.linalg.norm(w0)   # thoroughness/detail
    wE = np.array([-1.0, +1.5, 0.0, -1.0, 0.0, +0.4]); wE /= np.linalg.norm(wE)    # concise/positive
    u = wE - w0
    scaleA = 1.0 / (np.sqrt((np.diff(A[rng.integers(0, len(A), (5000, 2))], axis=1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    scaleE = 1.0 / (np.sqrt((np.diff(Epca[rng.integers(0, len(Epca), (5000, 2))], axis=1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    P_adv_rows = None
    # adv projection in attribute space blind to u
    uu = u / (np.linalg.norm(u) + 1e-12); B = rng.standard_normal((d, d)); B[:, 0] = uu
    Q, _ = np.linalg.qr(B); P_adv = Q[:, 1:1 + (d - 1)].T
    P_kproj = make_projection(Epca.shape[1], args.kproj, rng)

    conds = ["oracle_attr", "attr_adv", "llm_frozen", "llm_kproj"]
    rows = []
    cov = []
    for s in range(args.stages):
        frac = s / (args.stages - 1)
        wt = (1 - frac) * w0 + frac * wE; wt /= np.linalg.norm(wt)
        vt = A @ wt                                                    # true value of each response now
        # BT preferences from the drifting grader
        ij = rng.integers(0, len(A), (args.npairs, 2))
        ij = ij[ij[:, 0] != ij[:, 1]]
        dA = A[ij[:, 0]] - A[ij[:, 1]]
        pref = 1.0 / (1.0 + np.exp(-BETAT * (dA @ wt)))               # P(i > j)
        y = (rng.random(len(pref)) < pref).astype(float)             # y=1 => item i preferred
        dE = Epca[ij[:, 0]] - Epca[ij[:, 1]]
        rr = np.random.default_rng(s + 1)
        lbs = {
            "oracle_attr": _fit_lb(dA, y, A, scaleA, rr),
            "attr_adv":    _fit_lb(dA, y, A, scaleA, rr, P=P_adv),
            "llm_frozen":  _fit_lb(dE, y, Epca, scaleE, rr),
            "llm_kproj":   _fit_lb(dE, y, Epca, scaleE, rr, P=P_kproj),
        }
        rng_hi, rng_lo = vt.max(), vt.min()
        for c in conds:
            lb = lbs[c]
            reg = float((rng_hi - vt[int(np.argmax(lb))]) / (rng_hi - rng_lo + 1e-9))
            rk = rank_correlation(lb.tolist(), vt.tolist())
            rows.append(dict(stage=s, frac=round(frac, 2), condition=c, sel_regret=round(reg, 3), rank=round(rk, 3)))
        cov.append(dict(stage=s, cov_r2_llm=round(_r2(Epca, A @ u), 3),
                        cov_r2_attr=round(_r2(A, A @ u), 3)))

    json.dump(rows, open(os.path.join(args.out, "per_stage.json"), "w"), indent=2)
    # aggregate over stages (early third / late third)
    def agg(cond, key, lo, hi):
        v = [r[key] for r in rows if r["condition"] == cond and lo <= r["stage"] < hi]
        return float(np.mean(v)) if v else float("nan")
    T = args.stages
    L = ["# LLM-as-judge under a drifting evaluation standard\n",
         f"model={args.model}  corpus={len(texts)}  attrs={ATTR_NAMES}\n",
         "Drift: standard w_t moves from 'thoroughness/detail' (w0) to 'concise/positive' (w_end). "
         "Selection regret (argmax-VaR pick vs true best under current standard; lower=better) and "
         "rank corr, early vs late drift.\n",
         "| condition | regret early | regret late | rank early | rank late |",
         "|---|---|---|---|---|"]
    for c in conds:
        L.append(f"| {c} | {agg(c,'sel_regret',0,T//3):.3f} | {agg(c,'sel_regret',2*T//3,T):.3f} "
                 f"| {agg(c,'rank',0,T//3):+.2f} | {agg(c,'rank',2*T//3,T):+.2f} |")
    L.append("\n## Does the frozen LLM representation cover the drift axis?  R^2(a.u | features)\n")
    L.append(f"llm_frozen (PCA-{args.pca}): {np.mean([c['cov_r2_llm'] for c in cov]):.3f}   "
             f"oracle attrs: {np.mean([c['cov_r2_attr'] for c in cov]):.3f}  "
             f"(high => representation linearly encodes the drifting standard)\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("wrote", os.path.join(args.out, "results.md"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
