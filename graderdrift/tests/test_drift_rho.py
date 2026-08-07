"""Test 5 (spec v0.2 Section 6): rho mechanics.

- delta in the rowspace -> rho = 1.
- a pure-novel-axis ROTATION by theta -> rho = (1 - cos theta)/2 (the floor; the
  increment retains a component along vhat, which lies in the battery span).
- novel_translation -> rho = 0 exactly.
- the identity ||P_A delta|| = sqrt(rho) ||delta||.

In A0 the achievable rowspace equals S_hist (battery rank 16 = dim S_hist), so
P_A = P_{S_hist} here.
"""
import numpy as np
import pytest

from graderdrift import drift
from graderdrift.geometry import on_support_norm, rho, subspace_projector


@pytest.fixture
def setup():
    ctx = drift.DriftContext.a0(d=24, hist_dim=16)
    P_hist = subspace_projector(ctx.hist_idx, ctx.d)
    rng = np.random.default_rng(11)
    v0 = np.zeros(ctx.d)
    v0[ctx.hist_idx] = rng.standard_normal(len(ctx.hist_idx))
    v0 /= np.linalg.norm(v0)
    return ctx, P_hist, v0, rng


def test_on_support_jump_rho_one(setup):
    ctx, P_hist, v0, rng = setup
    v1 = drift.direction_jump(v0, ctx, rng, angle_deg=25, rho_target=1.0)
    assert rho(v1 - v0, P_hist) == pytest.approx(1.0, abs=1e-9)


def test_novel_translation_rho_zero(setup):
    ctx, P_hist, v0, rng = setup
    for mag in (0.2, 0.5, 1.3):
        v1 = drift.novel_translation(v0, ctx, rng, magnitude=mag)
        assert rho(v1 - v0, P_hist) == pytest.approx(0.0, abs=1e-9)


def test_pure_novel_rotation_hits_floor(setup):
    ctx, P_hist, v0, rng = setup
    for angle in (10.0, 25.0, 45.0):
        v1 = drift.direction_jump(v0, ctx, rng, angle_deg=angle, rho_target=0.0)
        expected = (1.0 - np.cos(np.radians(angle))) / 2.0
        assert rho(v1 - v0, P_hist) == pytest.approx(expected, abs=1e-6)


def test_recovered_norm_identity(setup):
    ctx, P_hist, v0, rng = setup
    for rt in (0.0, 0.25, 0.5, 0.75, 1.0):
        v1 = drift.direction_jump(v0, ctx, rng, angle_deg=30, rho_target=rt)
        delta = v1 - v0
        lhs = on_support_norm(delta, P_hist)
        rhs = np.sqrt(rho(delta, P_hist)) * np.linalg.norm(delta)
        assert lhs == pytest.approx(rhs, abs=1e-12)


def test_midlow_target_matches_analytic(setup):
    # Realized rho of a target-rho* rotation:
    #   rho = [(1-cos t)^2 + rho* sin^2 t] / [2 (1-cos t)]
    ctx, P_hist, v0, rng = setup
    theta = np.radians(25.0)
    for rt in (0.0, 0.5, 1.0):
        v1 = drift.direction_jump(v0, ctx, rng, angle_deg=25, rho_target=rt)
        realized = rho(v1 - v0, P_hist)
        analytic = ((1 - np.cos(theta)) ** 2 + rt * np.sin(theta) ** 2) / (2 * (1 - np.cos(theta)))
        assert realized == pytest.approx(analytic, abs=1e-6)
