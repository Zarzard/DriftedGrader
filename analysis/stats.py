"""Statistical protocol (plan Section 8): bootstrap CIs, paired-difference tests,
permutation tests. All comparisons are PAIRED BY SEED (conditions sharing a seed
share Phi and w0). Report effect sizes in the metric's own units; never bare
p-values. Detection delays are heavily skewed / censored -> use medians, never means.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n_boot=10000, alpha=0.05, agg=np.median, rng=None):
    """Bootstrap CI for an aggregate (default median) of a 1-D sample."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = agg(x[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(agg(x)), float(lo), float(hi)


def paired_diff_ci(a, b, n_boot=10000, alpha=0.05, agg=np.median, rng=None):
    """Paired difference a-b (by seed) with a bootstrap CI on the aggregate.
    Returns (point, lo, hi, excludes_zero)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    point, lo, hi = bootstrap_ci(d, n_boot, alpha, agg, rng)
    return point, lo, hi, bool(lo > 0 or hi < 0)


def permutation_test(a, b, n_perm=10000, agg=np.median, rng=None):
    """Two-sided paired permutation test: randomly swap a<->b within each seed,
    recompute agg(diff), p = fraction of |perm| >= |observed|."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = rng or np.random.default_rng(0)
    obs = agg(a - b)
    signs = rng.integers(0, 2, size=(n_perm, len(a))) * 2 - 1
    perms = agg(signs * (a - b), axis=1)
    p = float(np.mean(np.abs(perms) >= abs(obs) - 1e-12))
    return float(obs), p


def condition_permutation_test(values_by_condition, n_perm=10000, rng=None):
    """C3: shuffle condition labels across runs, recompute the effect (difference
    of medians between the two conditions), confirm the observed effect sits
    outside the permutation distribution. Two conditions."""
    keys = list(values_by_condition)
    assert len(keys) == 2, "condition permutation expects two conditions"
    a = np.asarray(values_by_condition[keys[0]], float)
    b = np.asarray(values_by_condition[keys[1]], float)
    pooled = np.concatenate([a, b]); na = len(a)
    rng = rng or np.random.default_rng(0)
    obs = np.median(a) - np.median(b)
    perms = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pooled)
        perms[i] = np.median(p[:na]) - np.median(p[na:])
    pval = float(np.mean(np.abs(perms) >= abs(obs) - 1e-12))
    return float(obs), pval


def censored_median(delays, cap=np.inf):
    """Median of detection delays with +inf censoring; report censoring rate too."""
    d = np.asarray(delays, float)
    n_cens = int(np.sum(~np.isfinite(d)))
    med = float(np.median(d)) if np.isfinite(np.median(d)) else float("inf")
    return med, n_cens / max(1, len(d))


def holm_correction(pvals):
    """Holm step-down correction within a pre-registered family. Returns adjusted
    p-values in the original order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj
