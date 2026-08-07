"""Test 2 (spec Section 6): Proposition 2 -- the beta/w factorization is a gauge.

(beta, w) and (beta/c, c*w) have identical v = beta*w, hence identical label
probabilities for all pairs. The factorization is not identifiable from choices.
"""
import numpy as np

from graderdrift.teacher import Teacher


def test_prop2_beta_w_rescaling_is_unobservable():
    d = 24
    r = np.random.default_rng(99)
    w = r.normal(size=d)
    beta = 2.3
    dphis = r.normal(size=(300, d))

    base = Teacher(w=w, s=0.0, beta=beta)
    for c in [0.1, 0.5, 2.0, 10.0, 137.0]:
        alt = Teacher(w=c * w, s=0.0, beta=beta / c)
        # Same identifiable state...
        assert np.allclose(alt.v(), base.v())
        # ...hence identical label probabilities on every pair (exact in R; to
        # float64 precision here, since c*w vs w route through different rounding).
        pb = np.array([base.prob(g) for g in dphis])
        pa = np.array([alt.prob(g) for g in dphis])
        assert np.allclose(pa, pb, atol=1e-12, rtol=1e-9), f"c={c} changed a label prob"
