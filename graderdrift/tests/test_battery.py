"""Test 7 (spec v0.2 Section 6): battery construction.

- direction panel true p in [0.45, 0.55]; norm panel in the two bands;
- rank(Phi_A) = 16 in A0; rowspace ⊂ S_hist;
- battery bytes identical across the run (frozen);
- min pairwise angle between selected gaps > 5deg.
"""
import numpy as np
import pytest

from graderdrift import drift
from graderdrift._math import sigmoid
from graderdrift.anchors import build_battery_a0


@pytest.fixture
def battery():
    ctx = drift.DriftContext.a0(d=24, hist_dim=16)
    rng = np.random.default_rng(0)
    v0 = np.zeros(ctx.d)
    v0[ctx.hist_idx] = rng.standard_normal(len(ctx.hist_idx))
    v0 /= np.linalg.norm(v0)
    return ctx, v0, build_battery_a0(v0, ctx, np.random.default_rng(1))


def test_panel_bands(battery):
    ctx, v0, bat = battery
    p = sigmoid(bat.z0)
    p_dir = p[bat.dir_idx]
    p_norm = p[bat.norm_idx]
    assert np.all((p_dir >= 0.45) & (p_dir <= 0.55)), p_dir
    in_band = ((p_norm >= 0.78) & (p_norm <= 0.86)) | ((p_norm >= 0.14) & (p_norm <= 0.22))
    assert np.all(in_band), p_norm
    # norm panel sign-balanced 4/4
    assert np.sum(bat.z0[bat.norm_idx] > 0) == 4
    assert np.sum(bat.z0[bat.norm_idx] < 0) == 4


def test_rank_and_support(battery):
    ctx, v0, bat = battery
    assert bat.rank == 16
    # rowspace ⊂ S_hist: novel coordinates of every gap are ~0
    assert np.allclose(bat.gaps[:, ctx.nov_idx], 0.0, atol=1e-12)
    # P_A leaves S_hist vectors alone and annihilates S_nov
    e_nov = np.zeros(ctx.d); e_nov[ctx.nov_idx[0]] = 1.0
    assert np.allclose(bat.P_A @ e_nov, 0.0, atol=1e-9)


def test_min_pairwise_angle(battery):
    _ctx, _v0, bat = battery
    assert bat.min_pairwise_angle_deg > 5.0


def test_frozen_bytes(battery):
    _ctx, _v0, bat = battery
    before = bat.gaps.tobytes()
    # gaps are read-only; attempting to write must fail, and bytes are stable
    with pytest.raises(ValueError):
        bat.gaps[0, 0] = 999.0
    assert bat.gaps.tobytes() == before


def test_reproducible_given_seed():
    ctx = drift.DriftContext.a0()
    v0 = np.zeros(ctx.d); v0[ctx.hist_idx] = np.random.default_rng(0).standard_normal(16)
    v0 /= np.linalg.norm(v0)
    b1 = build_battery_a0(v0, ctx, np.random.default_rng(7))
    b2 = build_battery_a0(v0, ctx, np.random.default_rng(7))
    assert np.array_equal(b1.gaps, b2.gaps)
