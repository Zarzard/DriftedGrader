"""One definition of the pairwise agreement metric, shared by every judge harness.

Previously each harness inlined `hit += (S[i] > S[j]) == (G[i] > G[j])`. That scores a tie
(S[i] == S[j]) as correct whenever gold happens to rank j above i and incorrect otherwise, i.e. it
breaks ties by the arbitrary order of responses in the pool rather than treating them as ties. Over
many pairs it averages near 0.5 so the published numbers were approximately right, but it added
variance that was not in the stated interval, and the effect grows with the judge's tie rate. The
0-100 integer judges saturate (deepseek-v4-pro answers 100 to most responses), so ties are common
there and must be scored explicitly.
"""
import numpy as np


def pair_score(S, G):
    """Fraction of gold-separated pairs the judge orders correctly; ties score 0.5.

    S, G: (P, k) judge scores and gold scores. Rows with any nan in S are skipped.
    Returns (score, n_pairs, tie_fraction).
    """
    S, G = np.asarray(S, float), np.asarray(G, float)
    hit = tot = ties = 0.0
    for p in np.where(~np.isnan(S).any(1))[0]:
        for i in range(S.shape[1]):
            for j in range(i + 1, S.shape[1]):
                if G[p, i] == G[p, j]:
                    continue
                tot += 1
                if S[p, i] == S[p, j]:
                    hit += 0.5
                    ties += 1
                else:
                    hit += (S[p, i] > S[p, j]) == (G[p, i] > G[p, j])
    return hit / max(tot, 1), int(tot), ties / max(tot, 1)


def report(S, G, label=""):
    sc, n, tf = pair_score(S, G)
    print(f"  orders gold-separated pairs: {sc:.3f}  (chance 0.500, n={n}, "
          f"judge ties {tf:.1%}){(' ' + label) if label else ''}")
    return sc, n, tf


def top1(S, G):
    """Fraction of pools where the judge's top-scored response is a gold-top response.

    np.argmax must not be used for this: it returns the FIRST maximal index, so a judge that ties
    every response scores 1.0 whenever the gold's top sits at index 0. On the Arena pool (k=2, gold
    always [1, 0]) that inflated Sonnet 4.5 from 0.566 to 0.737 -- exactly the 34.3% tie rate halved.
    A tie for the judge's maximum scores the fraction of tied slots that are gold-top.
    """
    S, G = np.asarray(S, float), np.asarray(G, float)
    ok = ~np.isnan(S).any(1)
    hit = 0.0
    for p in np.where(ok)[0]:
        best = S[p] == S[p].max()
        hit += float((G[p][best] == G[p].max()).mean())
    return hit / max(int(ok.sum()), 1), int(ok.sum())
