"""Test 9 (spec v0.2 Section 6): norm-jump exactness.

A norm jump ``v -> factor*v`` moves each anchor's logit by exactly
``Delta z_j = (factor - 1) * z_j(0)`` -- a logit difference, NOT ``log(factor)``.
This is the amendment A2 target: E2(iii) predicts the panel with the per-anchor
``Delta z_j``, whose norm-panel-center value at |z0| = 1.5434, factor 0.6 is
-0.617 (not log(0.6) = -0.511).
"""
import numpy as np
import pytest

from graderdrift import drift
from graderdrift._math import Z_STAR


def test_norm_jump_logit_shift_exact():
    rng = np.random.default_rng(3)
    d = 24
    v0 = rng.standard_normal(d)
    anchors = rng.standard_normal((30, d))  # frozen anchor gaps
    z0 = anchors @ v0

    for factor in (0.6, 0.8, 1.25, 2.0):
        v1 = drift.norm_scale(v0, factor)
        z1 = anchors @ v1
        dz = z1 - z0
        assert np.allclose(dz, (factor - 1.0) * z0, atol=1e-12, rtol=0.0)


def test_norm_jump_is_not_log_factor():
    # The wrong v0.1 prediction was log(factor); confirm the panel-center value is
    # (factor-1)*z_star, and that it differs materially from log(factor).
    factor = 0.6
    predicted = (factor - 1.0) * Z_STAR
    wrong = np.log(factor)
    assert predicted == pytest.approx(-0.6174, abs=1e-3)
    assert wrong == pytest.approx(-0.5108, abs=1e-3)
    assert abs(predicted - wrong) > 0.1  # not interchangeable


def test_direction_jump_preserves_norm():
    # Amendment cross-check: rotations preserve ||v|| (so S1p == S1).
    ctx = drift.DriftContext.a0()
    rng = np.random.default_rng(5)
    v0 = np.zeros(ctx.d)
    v0[ctx.hist_idx] = rng.standard_normal(len(ctx.hist_idx))
    for angle in (10, 25, 45):
        v1 = drift.direction_jump(v0, ctx, rng, angle_deg=angle, rho_target=1.0)
        assert np.linalg.norm(v1) == pytest.approx(np.linalg.norm(v0), rel=1e-12)
