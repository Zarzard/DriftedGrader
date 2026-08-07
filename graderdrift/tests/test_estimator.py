"""Test 6 (spec v0.2 Section 6): estimator sanity.

(a) Linear-Gaussian reduction: on y = H v + eps the filter is an exact Kalman
    filter, so per-coordinate 90% CI coverage is nominal within MC error.
(b) Logistic S0: 90% on-support CI coverage lands in [85, 95] over many short runs.
(c) S2 TV regression: full-vector TV picks up pure norm drift (>> the S0 floor)
    while direction-only TV is blind to it.
"""
import numpy as np
import pytest

from graderdrift import estimator as est
from graderdrift.experiments.e3_run import run_a0_filter


def test_6a_linear_gaussian_coverage():
    rng = np.random.default_rng(0)
    d, T, trials = 4, 15, 400
    H = np.eye(d)
    R = 0.5 ** 2 * np.eye(d)
    Q = 0.1 ** 2 * np.eye(d)
    hits = 0
    total = 0
    for _ in range(trials):
        v = rng.standard_normal(d)
        state = est.GaussState(np.zeros(d), 1.0 * np.eye(d))
        for _t in range(T):
            v = v + rng.multivariate_normal(np.zeros(d), Q)
            y = H @ v + rng.multivariate_normal(np.zeros(d), R)
            state = est.predict(state, Q)
            state = est.update_linear(state, H, y, R)
            sd = np.sqrt(np.diag(state.cov))
            hits += np.sum(np.abs(state.mean - v) <= 1.645 * sd)
            total += d
    cov = hits / total
    assert 0.86 <= cov <= 0.94, f"linear-Gaussian coverage {cov:.3f} not nominal"


def test_6b_logistic_s0_coverage():
    covs = []
    for seed in range(60):
        r = run_a0_filter("S0", seed, mode="known_calendar", K=12)
        covs.append(r["cov_hit_on"])
    cov = float(np.mean(np.concatenate(covs)))
    assert 0.85 <= cov <= 0.95, f"logistic S0 on-support coverage {cov:.3f} outside [85,95]"


def test_6c_net_norm_regression_s2_vs_s0():
    # The drift statistic must detect S2's pure norm decay via its NORM channel, and
    # the DIRECTION channel must stay blind to it. Net (endpoint) statistics, not
    # path-length TV, which is noise-dominated for slow drift (AMENDMENTS wave 3 C4).
    s0 = [run_a0_filter("S0", s, K=60) for s in range(6)]
    s2 = [run_a0_filter("S2", s, K=60) for s in range(6)]
    ln0 = np.array([est.net_lognorm_change(r["means"]) for r in s0])
    ln2 = np.array([est.net_lognorm_change(r["means"]) for r in s2])
    an0 = np.array([est.net_angle_change(r["means"]) for r in s0])
    an2 = np.array([est.net_angle_change(r["means"]) for r in s2])
    # norm channel: S2 clearly above the S0 floor (separated by > 2 pooled sd)
    assert ln2.mean() - ln0.mean() > 2 * np.hypot(ln0.std(), ln2.std()), \
        f"net log-norm blind to S2: S2={ln2.mean():.3f} S0={ln0.mean():.3f}"
    # direction channel: S2 (pure norm) not distinguishable from S0
    assert an2.mean() < an0.mean() + an0.std() + 0.5, \
        f"net angle should be ~blind to S2: S2={an2.mean():.2f} S0={an0.mean():.2f}"
