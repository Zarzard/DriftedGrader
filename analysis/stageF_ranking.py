"""Stage F ranking-validity-at-scale: does a reward-posterior VaR bound RANK
policies correctly under a drifting grader, at neural (Metaworld/PEBBLE) scale?

Reuses the validated tabular machinery (RewardPosterior, make_projection,
rank_correlation) but feeds it REAL data exported from a drifting PEBBLE run
(`ranking_export.pkl`): each feedback round t provides
    w      : grader weights w_t (d,)            -- the CURRENT grader
    m      : deployed policy's achieved feature profile m_t (d,)  [eval-policy]
    sphi1  : per-segment summed features Sum phi(seg1)   [Nt, d]
    sphi2  : per-segment summed features Sum phi(seg2)   [Nt, d]
    lbl    : B-Pref label (1=>seg2 preferred, 0=>seg1, -1=>tie)

Eval policies = the spread of achieved profiles {m_t}; f_pi -> m_t.
Conditions:
  full  : linear posterior on full d-dim features (well-specified reference)
  kproj : linear posterior on a RANDOM k-dim projection (generic misspec)
  adv   : linear posterior on a k-dim projection BLIND to the drift direction u
          (adversarial/worst-case misspec)
  small : small-capacity NONLINEAR MLP ensemble r=f(phi), conservative bound
          mean_k - std_k (needs the per-segment features -> f(s1)-f(s2))

Run: python -m analysis.stageF_ranking --exp_glob '/home/zifan/NoisyPbRL/exp/stageFrank/*'
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np

from drifted_grader.posterior import RewardPosterior
from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import rank_correlation

BETAT = 10.0
ALPHA = 0.05
PRIOR = 2.0
NS = 4000


def _load(path):
    with open(os.path.join(path, "ranking_export.pkl"), "rb") as fh:
        rounds = pickle.load(fh)
    return [r for r in rounds if len(r["lbl"]) > 0]


def _pool(rounds, upto):
    """Pool comparisons in rounds[:upto+1] -> (s1[N,d], s2[N,d], y[N]) in the
    RewardPosterior convention (y=1 => FIRST segment preferred), ties dropped."""
    s1s, s2s, ys = [], [], []
    for r in rounds[: upto + 1]:
        lbl = r["lbl"]; keep = lbl >= 0
        if keep.sum() == 0:
            continue
        s1s.append(r["sphi1"][keep]); s2s.append(r["sphi2"][keep])
        ys.append((lbl[keep] == 0).astype(float))     # label==0 => seg1 preferred => y=1
    if not s1s:
        return None, None, None
    return np.concatenate(s1s, 0), np.concatenate(s2s, 0), np.concatenate(ys, 0)


def _adv_projection(d, k, u, rng):
    """k x d projection whose ROW space lies in u^perp, so P u ~ 0: the reward
    model is BLIND to the drift direction u (adversarial/worst-case misspec)."""
    u = u / (np.linalg.norm(u) + 1e-12)
    B = rng.standard_normal((d, d)); B[:, 0] = u
    Q, _ = np.linalg.qr(B)                              # Q[:,0] ~ +-u ; Q[:,1:] spans u^perp
    return Q[:, 1:1 + k].T


def _fit_linear(dphi, y, feats, kind, k, scale, rng, Pmat=None):
    """Linear reward-posterior VaR bound (kind in full/kproj/adv)."""
    if kind == "kproj":
        P = make_projection(dphi.shape[1], k, rng)
    elif kind == "adv":
        P = Pmat
    else:
        P = None
    X = (dphi @ P.T) if P is not None else dphi
    F = (feats @ P.T) if P is not None else feats
    post = RewardPosterior(X.shape[1])
    post.fit(X * scale, y, BETAT, prior_scale=PRIOR, n_samples=NS, rng=rng)
    lb = np.array([post.lower_bound(f * scale, ALPHA) for f in F])
    return lb


def _fit_small(s1, s2, y, feats, scale, rng, K=5, hidden=8, steps=400):
    """Small-capacity NONLINEAR reward: ensemble of tiny MLPs r=f(phi) trained on
    the per-segment BT comparison set  P(seg1>seg2)=sigmoid(betaT*(f(s1)-f(s2))).
    Conservative bound = mean_k - std_k over the ensemble (affine-invariant rank)."""
    X1, X2, Fs = s1 * scale, s2 * scale, feats * scale
    d = X1.shape[1]; n = len(y); preds = []
    for _ in range(K):
        r = np.random.default_rng(rng.integers(1 << 30))
        W1 = r.standard_normal((hidden, d)) * 0.5; b1 = np.zeros(hidden)
        w2 = r.standard_normal(hidden) * 0.5; lr = 0.1
        for _ in range(steps):
            h1 = np.tanh(X1 @ W1.T + b1); h2 = np.tanh(X2 @ W1.T + b1)
            z = BETAT * (h1 @ w2 - h2 @ w2)
            g = (1.0 / (1.0 + np.exp(-z))) - y            # y=1 => seg1 preferred
            gr1 = g * BETAT; gr2 = -g * BETAT
            gw2 = (gr1[:, None] * h1 + gr2[:, None] * h2).mean(0)
            gh1 = gr1[:, None] * w2 * (1 - h1 ** 2)
            gh2 = gr2[:, None] * w2 * (1 - h2 ** 2)
            gW1 = (gh1.T @ X1 + gh2.T @ X2) / n
            gb1 = (gh1 + gh2).mean(0)
            w2 -= lr * gw2; W1 -= lr * gW1; b1 -= lr * gb1
        preds.append(np.tanh(Fs @ W1.T + b1) @ w2)
    preds = np.stack(preds, 0)
    return preds.mean(0) - preds.std(0)


def analyze_run(path, k=3, stages=12):
    rounds = _load(path)
    if len(rounds) < 4:
        return None
    feats = np.stack([r["m"] for r in rounds])
    alls1 = np.concatenate([r["sphi1"] for r in rounds])
    alls2 = np.concatenate([r["sphi2"] for r in rounds])
    scale = 1.0 / (np.sqrt(((alls1 - alls2) ** 2).sum(1).mean()) + 1e-8)
    u = rounds[-1]["w"] - rounds[0]["w"]
    rng = np.random.default_rng(0)
    P_adv = (_adv_projection(feats.shape[1], k, u, rng) if np.linalg.norm(u) > 1e-6
             else make_projection(feats.shape[1], k, rng))
    idx = np.unique(np.linspace(len(rounds) // 3, len(rounds) - 1, stages).astype(int))
    out = {c: [] for c in ("full", "kproj", "adv", "small")}
    for t in idx:
        s1, s2, y = _pool(rounds, t)
        if s1 is None or len(y) < 20:
            continue
        dphi = s1 - s2
        w_t = rounds[t]["w"]; evalset = feats[: t + 1]
        v_true = (evalset @ w_t).tolist()
        rng2 = np.random.default_rng(int(t) + 1)
        for kind in ("full", "kproj", "adv"):
            lb = _fit_linear(dphi, y, evalset, kind, k, scale, rng2, Pmat=P_adv)
            out[kind].append(rank_correlation(lb.tolist(), v_true))
        lb_s = _fit_small(s1, s2, y, evalset, scale, rng2)
        out["small"].append(rank_correlation(lb_s.tolist(), v_true))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_glob", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    agg = {}
    for p in sorted(glob.glob(args.exp_glob)):
        name = os.path.basename(p.rstrip("/"))
        drift = "SAT" if "SAT" in name else ("NONE" if "NONE" in name else "?")
        r = analyze_run(p, k=args.k)
        if r is None:
            print(f"  skip {name}"); continue
        agg.setdefault(drift, {c: [] for c in r})
        for c in r:
            agg[drift][c] += r[c]
        print(f"  {name}: " + "  ".join(f"{c}={np.mean(r[c]):+.2f}" for c in r))
    print(f"\n{'drift':>5} {'cond':>6} | {'rank_corr':>10}")
    for drift in ("NONE", "SAT"):
        if drift not in agg:
            continue
        for c in ("full", "kproj", "adv", "small"):
            if agg[drift][c]:
                print(f"{drift:>5} {c:>6} | {np.mean(agg[drift][c]):>+10.3f}")
    print("\nfull=well-spec linear; kproj=random k<d linear; adv=k<d linear blind to"
          "\ndrift dir; small=tiny nonlinear MLP ensemble. Inversion reproduces iff a"
          "\nmisspec condition goes NEGATIVE under SAT while full stays robust.")


if __name__ == "__main__":
    main()
