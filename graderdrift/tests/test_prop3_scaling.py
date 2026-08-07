"""Test 3 (spec Section 6): Proposition 3 -- rater-scale / gap-magnitude aliasing.

The label law under (c*v, gaps G) equals the law under (v, c*G) on a fixed pair
set, exactly. This is the confound that makes the operational stream unable to
attribute change to the rater (note Section 4) -- and the reason anchors (fixed
gaps) are necessary.
"""
import numpy as np

from graderdrift.teacher import Teacher


def test_prop3_scale_gap_aliasing_exact():
    d = 16
    r = np.random.default_rng(2024)
    w = r.normal(size=d)
    base = Teacher(w=w, s=0.0, beta=1.0)
    v = base.v()
    G = r.normal(size=(500, d))  # fixed pair set

    for c in [0.25, 0.5, 2.0, 4.0, 50.0]:
        scaled_rater = Teacher(w=c * w, s=0.0, beta=1.0)  # rater scale c*v
        p_scaled_rater = np.array([scaled_rater.prob(g) for g in G])
        # gaps scaled by c instead, rater unchanged
        p_scaled_gaps = np.array([base.prob(c * g) for g in G])
        # Exact in R; equal to float64 precision (the two scalings round differently).
        assert np.allclose(p_scaled_rater, p_scaled_gaps, atol=1e-12, rtol=1e-9), (
            f"c={c}: (c*v, G) and (v, c*G) must induce identical label laws"
        )
        # sanity: both equal sigmoid((c*v).g)
        assert np.allclose(p_scaled_rater, 1.0 / (1.0 + np.exp(-(c * v) @ G.T)))
