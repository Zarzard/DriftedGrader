"""Plan 02 posterior correctness gates (Section 8): PG1, PG4, PG5.
PG2 (calibration) is a heavier statistical gate -- see analysis/run_pg2.py and the
EXPERIMENT_LOG; it is checked at scale, not in the fast unit suite.

Run:  python -m pytest tests/test_posterior.py -q
"""
import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import DriftNone
from drifted_grader.policies import expected_features
from drifted_grader.metrics import policy_from_scores
from drifted_grader.posterior import RewardPosterior, weighted_quantile


def _uniform_comps(arms, grader, N, rng):
    i = rng.integers(0, arms.n, N); j = rng.integers(0, arms.n, N)
    mk = i != j
    pairs = np.stack([i[mk], j[mk]], 1)
    y = grader.label(pairs, rng).astype(float)
    return arms.Phi[pairs[:, 0]] - arms.Phi[pairs[:, 1]], y


def _eval_pols(arms):
    Phi, w0 = arms.Phi, arms.w0
    pols = [policy_from_scores(Phi @ w0, b) for b in (2.0, 0.2, 0.05)]
    pols.append(np.full(len(Phi), 1.0 / len(Phi)))
    o = np.zeros(len(Phi)); o[(Phi @ w0).argmax()] = 1.0; pols.append(o)
    return pols


def test_PG1_posterior_recovery():
    """PG1: well-specified, many comparisons -> posterior mean recovers w_true
    (cos > 0.99) and posterior covariance trace collapses."""
    for s in range(3):
        rng = np.random.default_rng(s)
        arms = make_arms(256, 6, rng)
        dPhi, y = _uniform_comps(arms, DriftNone(arms, 10.0), 20000, rng)
        post = RewardPosterior(6).fit(dPhi, y, betaT=10.0, prior_scale=2.0,
                                      n_samples=2000, rng=rng)
        em = post.posterior_mean()
        cos = np.dot(em, arms.w0) / (np.linalg.norm(em) * np.linalg.norm(arms.w0))
        assert cos > 0.99, f"seed {s}: cos={cos}"
        assert post.diagnostics()["cov_trace"] < 0.02


def test_PG4_bound_monotone_in_alpha():
    """PG4: LB_alpha non-decreasing in alpha (catches quantile-direction bugs)."""
    rng = np.random.default_rng(0)
    arms = make_arms(256, 6, rng)
    dPhi, y = _uniform_comps(arms, DriftNone(arms, 10.0), 3000, rng)
    post = RewardPosterior(6).fit(dPhi, y, betaT=10.0, prior_scale=2.0,
                                  n_samples=6000, rng=rng)
    for pi in _eval_pols(arms):
        f = expected_features(pi, arms.Phi)
        lb = [post.lower_bound(f, a) for a in (0.05, 0.10, 0.25)]
        assert lb[0] <= lb[1] + 1e-9 and lb[1] <= lb[2] + 1e-9, f"non-monotone: {lb}"


def test_PG5_IS_vs_MH_agreement():
    """PG5: importance-sampling and Metropolis-Hastings lower bounds agree to
    within Monte-Carlo error (< 0.02 * achievable range)."""
    rng = np.random.default_rng(3)
    arms = make_arms(256, 6, rng)
    dPhi, y = _uniform_comps(arms, DriftNone(arms, 10.0), 2000, rng)
    post = RewardPosterior(6).fit(dPhi, y, betaT=10.0, prior_scale=2.0,
                                  n_samples=8000, rng=rng)
    mh = post.mh_sample(dPhi, y, 10.0, 2.0, n=15000, burn=2000, rng=rng)
    rng_range = float((arms.Phi @ arms.w0).ptp())
    for pi in _eval_pols(arms):
        f = expected_features(pi, arms.Phi)
        lb_is = post.lower_bound(f, 0.05)
        lb_mh = weighted_quantile(mh @ f, np.ones(len(mh)), 0.05)
        assert abs(lb_is - lb_mh) < 0.02 * rng_range, \
            f"IS/MH disagree: {abs(lb_is - lb_mh)} vs {0.02 * rng_range}"
