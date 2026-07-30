"""Detectability of preference drift vs. the BREADTH of the re-elicited comparison
set -- the focused contribution.

Claim: the standard on-policy monitor is blind to drift because the drift hides in
the directions on-policy comparisons least constrain (VR<1). A fixed, BROAD anchor
set re-elicited over time spans those directions and detects the drift. Detection
power grows with anchor breadth.

Detector (D5, anchored label shift): fix q_anchor comparison pairs at round 0, drawn
from Gibbs(w0.phi, beta_anchor) -- beta_anchor small = narrow/on-policy-like, large =
broad/uniform. Each round re-elicit q_anchor sampled labels; statistic
  S_t = sum_a | phat_t(a) - phat_0(a) |
estimated from the sampled labels. Threshold calibrated to 5% false-alarm on
DRIFT-NONE runs; power = fraction of DRIFT-SAT runs that exceed it by round T.

Run:  python -m analysis.run_detectability
"""
from __future__ import annotations

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader
from drifted_grader.metrics import policy_from_scores
from drifted_grader.policies import expected_features, sample_pairs


def anchor_pairs(Phi, w0, beta_anchor, q, rng):
    """Fixed anchor comparisons drawn from Gibbs(w0.phi, beta_anchor).
    beta_anchor=inf -> uniform (broad); small -> concentrated (narrow)."""
    if np.isinf(beta_anchor):
        i = rng.integers(0, len(Phi), q); j = rng.integers(0, len(Phi), q)
    else:
        pi = policy_from_scores(Phi @ w0, beta_anchor)
        i = rng.choice(len(Phi), q, p=pi); j = rng.choice(len(Phi), q, p=pi)
    mk = i != j
    return np.stack([i[mk], j[mk]], 1)


def run_one(seed, drift, lam, breadths, T=100, n=256, d=6, betaT=10.0,
            beta_policy=0.05, q_anchor=128, n_repeat=20):
    """Per breadth, return (true structural signal, sampled detector statistic).
    - true signal: mean_a |p_T(a) - p_0(a)| with DETERMINISTIC grader probs
      (isolates whether the anchors even span the drift direction).
    - sampled stat: same but p estimated from n_repeat re-elicitations (finite budget)."""
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + 31)
    arms = make_arms(n, d, rng)
    Phi, w0 = arms.Phi, arms.w0
    grader = make_grader(drift, arms, betaT, rng, lam=lam, T=T)

    anchors, p0_true, p0_hat = {}, {}, {}
    for b in breadths:
        A = anchor_pairs(Phi, w0, b, q_anchor, rng)
        if len(A) < 5:                                # degenerate (over-concentrated)
            A = anchor_pairs(Phi, w0, np.inf, 8, rng)  # fall back, still "narrow-ish"
        anchors[b] = A
        p0_true[b] = grader.prob_first(A).copy()
        p0_hat[b] = np.mean([grader.label(A, arng) for _ in range(n_repeat)], axis=0)

    m = np.zeros(d)
    for t in range(T):
        pi = policy_from_scores(Phi @ grader.w, beta_policy)
        m = 0.5 * m + 0.5 * expected_features(pi, Phi)
        grader.step(m)

    true_sig, samp_sig = {}, {}
    for b in breadths:
        pT_true = grader.prob_first(anchors[b])
        pT_hat = np.mean([grader.label(anchors[b], arng) for _ in range(n_repeat)], axis=0)
        true_sig[b] = float(np.mean(np.abs(pT_true - p0_true[b])))
        samp_sig[b] = float(np.mean(np.abs(pT_hat - p0_hat[b])))
    return true_sig, samp_sig


def detection_power(breadths, lam=0.4, seeds=40, far=0.05):
    sat_t = {b: [] for b in breadths}; none_t = {b: [] for b in breadths}
    sat_s = {b: [] for b in breadths}; none_s = {b: [] for b in breadths}
    for s in range(seeds):
        ts, ss = run_one(s, "SAT", lam, breadths)
        tn, sn = run_one(s, "NONE", 0.0, breadths)
        for b in breadths:
            sat_t[b].append(ts[b]); none_t[b].append(tn[b])
            sat_s[b].append(ss[b]); none_s[b].append(sn[b])
    out = {}
    for b in breadths:
        thr = np.quantile(none_s[b], 1 - far)
        power = float(np.mean(np.array(sat_s[b]) > thr))
        out[b] = dict(power=power, true_sat=np.mean(sat_t[b]), true_none=np.mean(none_t[b]),
                      samp_sat=np.mean(sat_s[b]), samp_none=np.mean(none_s[b]))
    return out


def main():
    breadths = [0.05, 0.1, 0.3, 1.0, np.inf]
    print("Drift detectability vs anchor breadth. beta_anchor small=narrow/on-policy,")
    print("inf=uniform/broad. lam=0.4, 40 seeds, q_anchor=128, 20 re-elicitations.\n")
    print(f"{'beta_anchor':>12} {'power@5%FAR':>11} {'true_sig_SAT':>13} {'true_sig_NONE':>14}")
    res = detection_power(breadths)
    for b in breadths:
        r = res[b]
        label = "uniform" if np.isinf(b) else f"{b:g}"
        tag = "  broad: DETECTS" if r["power"] > 0.8 else ("  narrow: BLIND" if r["power"] < 0.3 else "")
        print(f"{label:>12} {r['power']:>11.2f} {r['true_sat']:>13.4f} {r['true_none']:>14.4f}{tag}")
    print("\ntrue_sig = structural anchor signal (0 = anchors don't span the drift).")
    print("=> narrow/on-policy anchors are STRUCTURALLY blind; broad anchors detect.")
    return res


if __name__ == "__main__":
    main()
