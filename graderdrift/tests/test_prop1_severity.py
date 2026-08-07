"""Test 1 (spec Section 6): Proposition 1 + Corollary 1.

Prop 1: the label law of a within-rater, same-sitting comparison does not depend
on s_t -- arbitrary severity drift (including huge jumps) leaves every comparison
probability unchanged.

Cor 1: a pair synthesized from scores collected under different severity states
DOES depend on the severity difference s_x - s_y.
"""
import numpy as np
import pytest

from graderdrift.teacher import Teacher


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def test_prop1_same_sitting_prob_invariant_to_severity(rng):
    d = 24
    w = rng.normal(size=d)
    beta = 1.7
    dphis = rng.normal(size=(200, d))

    base = Teacher(w=w, s=0.0, beta=beta)
    base_probs = np.array([base.prob(g) for g in dphis])

    # Arbitrary severity trajectory, including enormous jumps.
    for s in [1e-3, 1.0, -5.0, 1e3, -1e6, 1e9]:
        t = Teacher(w=w, s=s, beta=beta)
        probs = np.array([t.prob(g) for g in dphis])
        assert np.allclose(probs, base_probs, atol=0.0, rtol=0.0), f"severity s={s} changed a comparison prob"


def test_prop1_label_draws_match_with_common_rng(rng):
    # Even the sampled labels are identical given a common RNG, since prob is identical.
    d = 8
    w = rng.normal(size=d)
    dphis = rng.normal(size=(50, d))
    t_a = Teacher(w=w, s=0.0, beta=1.0)
    t_b = Teacher(w=w, s=1e6, beta=1.0)
    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(7)
    labels_a = [t_a.label(g, r1) for g in dphis]
    labels_b = [t_b.label(g, r2) for g in dphis]
    assert labels_a == labels_b


def test_cor1_score_derived_pair_depends_on_severity_difference():
    d = 6
    r = np.random.default_rng(0)
    w = r.normal(size=d)
    phi_x = r.normal(size=d)
    phi_y = r.normal(size=d)
    t = Teacher(w=w, s=0.0, beta=1.0)

    # Vary only the severity difference; the score-derived prob must move.
    p_equal = t.prob_from_scores(phi_x, phi_y, s_x=0.0, s_y=0.0)
    p_shift = t.prob_from_scores(phi_x, phi_y, s_x=2.0, s_y=0.0)
    assert not np.isclose(p_equal, p_shift), "Cor 1: severity difference must survive in score-derived pairs"

    # And it depends on the DIFFERENCE only: a common shift cancels.
    p_common = t.prob_from_scores(phi_x, phi_y, s_x=3.5, s_y=3.5)
    assert np.isclose(p_common, p_equal)

    # Explicit functional form: prob = sigmoid(beta*(w.dphi + (s_x - s_y))).
    from graderdrift._math import sigmoid

    expected = sigmoid(t.beta * (w @ (phi_x - phi_y) + (2.0 - 0.0)))
    assert np.isclose(p_shift, expected)
