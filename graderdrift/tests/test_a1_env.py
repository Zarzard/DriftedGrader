"""Tests 10 + 11 (spec v0.2 Section 6): A1 rank geometry and gold evaluation.

10: achievable-gap matrix rank = 6; null space = span{constant feature, uniform
    cell-type shift}; the SN schedule's null-space translation leaves every label
    probability bit-identical and causes zero harm.
11: exact policy evaluation against an analytic 2-state MDP.
"""
import numpy as np
import pytest

from graderdrift import drift
from graderdrift.envs.grid import GAMMA, TRUE_W, get_env, uniform_policy
from graderdrift.experiments.a1_run import _reference_geometry
from graderdrift.policy import evaluate_policy, soft_value_iteration
from graderdrift.teacher import Teacher


@pytest.fixture(scope="module")
def geometry():
    env = get_env()
    rng = np.random.default_rng(7)
    return env, _reference_geometry(env, rng)


def test_gap_rank_six(geometry):
    _env, (gaps, P_ach, eigvecs, null_basis, rank) = geometry
    assert rank == 6
    assert null_basis.shape == (8, 2)


def test_null_space_span(geometry):
    _env, (gaps, P_ach, _eig, null_basis, _r) = geometry
    # claimed null vectors: constant feature e7, and uniform cell-type shift
    e_const = np.zeros(8); e_const[7] = 1.0
    u_cell = np.zeros(8); u_cell[[0, 2, 3, 4]] = 1.0
    for u in (e_const, u_cell):
        assert np.allclose(gaps @ u, 0.0, atol=1e-9), "claimed null vector hits a gap"
        assert np.linalg.norm(P_ach @ u) < 1e-6
    # and the null basis reproduces exactly that 2-dim space
    M = np.column_stack([e_const, u_cell / np.linalg.norm(u_cell)])
    proj = null_basis @ (null_basis.T @ M)
    assert np.allclose(proj, M, atol=1e-6)


def test_sn_null_translation_invisible_and_harmless(geometry):
    _env, (gaps, P_ach, _eig, null_basis, _r) = geometry
    ctx = drift.DriftContext.a1(P_ach, _eig, null_basis)
    rng = np.random.default_rng(3)
    v0 = 3.0 * TRUE_W
    v1 = drift.null_translation(v0, ctx, rng, magnitude=1.92)
    assert np.linalg.norm(v1 - v0) == pytest.approx(1.92, rel=1e-9)
    # invisible: identical label probability on every achievable gap
    assert np.allclose(gaps @ v1, gaps @ v0, atol=1e-8)
    # harmless: per-step reward shift is constant => same soft-VI policy
    env = _env
    r0 = env.F @ (v0 / 3.0)
    r1 = env.F @ (v1 / 3.0)
    shift = r1 - r0
    assert np.allclose(shift, shift.flat[0], atol=1e-9), "reward shift must be constant"
    pi0 = soft_value_iteration(env.P, r0, GAMMA)
    pi1 = soft_value_iteration(env.P, r1, GAMMA)
    assert np.allclose(pi0, pi1, atol=1e-8)


def test_gold_eval_two_state_analytic():
    # 2-state MDP, 1 action: stay in state with prob 1. r = [1, 0].
    # V(s0) = 1/(1-gamma), V(s1) = 0, J = mean = 0.5/(1-gamma).
    gamma = 0.9
    P = np.zeros((1, 2, 2)); P[0] = np.eye(2)
    r = np.array([[1.0], [0.0]])
    pi = np.ones((2, 1))
    J = evaluate_policy(P, r, gamma, pi)
    assert J == pytest.approx(0.5 / (1 - gamma), rel=1e-12)

    # 2 states, 2 actions: action 0 stays, action 1 swaps; reward 1 only in s0.
    # Deterministic swap-policy from s1, stay-policy from s0:
    # V(s0) = 1 + g V(s0) -> 1/(1-g); V(s1) = 0 + g V(s0) = g/(1-g).
    P = np.zeros((2, 2, 2)); P[0] = np.eye(2); P[1] = np.array([[0, 1], [1, 0]])
    r = np.array([[1.0, 1.0], [0.0, 0.0]])
    pi = np.array([[1.0, 0.0], [0.0, 1.0]])
    J = evaluate_policy(P, r, gamma, pi)
    expected = 0.5 * (1 / (1 - gamma) + gamma / (1 - gamma))
    assert J == pytest.approx(expected, rel=1e-12)


def test_a1_teacher_scale():
    t = Teacher(w=TRUE_W, beta=3.0)
    assert np.linalg.norm(t.v()) == pytest.approx(3.0 * np.linalg.norm(TRUE_W), rel=1e-12)
    assert np.linalg.norm(t.v()) == pytest.approx(4.44, abs=0.02)
