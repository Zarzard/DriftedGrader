"""C8 policy machinery: soft value iteration + exact policy evaluation.

Generic over (P, r): P is (A, S, S) transition tensor, r is (S, A) per-step reward.
Used with the gridworld's exact matrices and, in test 11, a hand-built 2-state MDP.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

SWEEPS = 80
TAU_POL = 0.05


def soft_value_iteration(P: np.ndarray, r: np.ndarray, gamma: float,
                         tau: float = TAU_POL, sweeps: int = SWEEPS) -> np.ndarray:
    """Soft VI: V = tau logsumexp(Q/tau); returns the softmax policy (S, A)."""
    A_n, S_n, _ = P.shape
    V = np.zeros(S_n)
    Q = np.zeros((S_n, A_n))
    for _ in range(sweeps):
        for a in range(A_n):
            Q[:, a] = r[:, a] + gamma * (P[a] @ V)
        V = tau * logsumexp(Q / tau, axis=1)
    Z = Q / tau
    Z -= Z.max(axis=1, keepdims=True)
    pi = np.exp(Z)
    pi /= pi.sum(axis=1, keepdims=True)
    return pi


def evaluate_policy(P: np.ndarray, r: np.ndarray, gamma: float, pi: np.ndarray) -> float:
    """Exact J(pi) = mean_s V^pi(s) (uniform start): V = (I - gamma P^pi)^-1 r^pi."""
    A_n, S_n, _ = P.shape
    P_pi = np.einsum("sa,ast->st", pi, P)   # (S, S)
    r_pi = np.einsum("sa,sa->s", pi, r)
    V = np.linalg.solve(np.eye(S_n) - gamma * P_pi, r_pi)
    return float(V.mean())


def reward_from_w(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-step expected reward r(s, a) = w . F[s, a]."""
    return F @ np.asarray(w, dtype=float)


def feature_expectation(P: np.ndarray, F: np.ndarray, gamma: float, pi: np.ndarray) -> np.ndarray:
    """Discounted state-action feature expectation of ``pi`` from a uniform start:
    mu = mean_s (I - gamma P^pi)^-1 f^pi, where f^pi(s) = sum_a pi(s,a) F(s,a).
    Returns a d-vector (the policy's 'behavior direction' in feature space, for E6)."""
    A_n, S_n, _ = P.shape
    P_pi = np.einsum("sa,ast->st", pi, P)
    f_pi = np.einsum("sa,sad->sd", pi, F)          # (S, d)
    Psi = np.linalg.solve(np.eye(S_n) - gamma * P_pi, f_pi)
    return Psi.mean(axis=0)


def optimal_return(P: np.ndarray, F: np.ndarray, gamma: float, w: np.ndarray,
                   tau: float = TAU_POL) -> tuple[float, np.ndarray]:
    """(J_w(pi*_w), pi*_w) with pi*_w from soft VI on w."""
    r = reward_from_w(F, w)
    pi = soft_value_iteration(P, r, gamma, tau=tau)
    return evaluate_policy(P, r, gamma, pi), pi
