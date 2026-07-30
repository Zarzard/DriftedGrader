"""Policies over the arm set.  No deep RL in Stages A-D; two standard
optimisation-pressure knobs plus a differentiable softmax for diagnostics.

  GibbsPolicy   pi(a) ~ pi_ref(a) * exp(r_hat(a) / beta),  pi_ref uniform.
                Pressure increases as beta -> 0.  Closed form.
  BestOfN       sample n_bo arms from pi_ref, take the r_hat-argmax.
                Induced arm distribution computed in closed form
                (with-replacement order statistics).  Pressure grows with n_bo.
  SoftmaxPolicy pi(a) ~ exp(theta_a),  theta in R^n.  Differentiable; used as
                ground truth for the performative-gradient diagnostic.
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def gibbs_policy(scores: np.ndarray, beta: float) -> np.ndarray:
    """pi(a) ~ exp(scores(a) / beta) with uniform reference (const drops out)."""
    if beta <= 0:
        raise ValueError("beta must be > 0")
    return softmax(scores / beta)


def best_of_n_policy(scores: np.ndarray, n_bo: int) -> np.ndarray:
    """Exact induced distribution of best-of-n over arms, uniform ref, sampling
    WITH replacement.  P(a chosen) = P(a is the argmax of n_bo iid uniform draws).

    For distinct scores with c = #{arms with score <= score(a)}:
        P(a) = (c/n)^n_bo - ((c-1)/n)^n_bo.
    Ties within a score group share the group's mass equally.
    """
    n = len(scores)
    n_bo = int(n_bo)
    if n_bo <= 1:
        # degenerates to uniform (single uniform draw)
        return np.full(n, 1.0 / n)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    prob = np.zeros(n)
    i = 0
    below = 0  # count of arms strictly below current group
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        g = j - i + 1                       # group size (ties)
        c_hi = below + g                    # arms with score <= group
        mass = (c_hi / n) ** n_bo - (below / n) ** n_bo
        for k in range(i, j + 1):
            prob[order[k]] = mass / g
        below = c_hi
        i = j + 1
    prob /= prob.sum()
    return prob


def make_policy(kind: str, scores: np.ndarray, beta: float = None,
                n_bo: int = None) -> np.ndarray:
    """Return the policy's probability vector over arms."""
    kind = kind.lower()
    if kind == "gibbs":
        return gibbs_policy(scores, beta)
    if kind in ("bestofn", "best_of_n", "bon"):
        return best_of_n_policy(scores, n_bo)
    if kind == "softmax":
        return softmax(scores)  # `scores` are theta here
    raise ValueError(f"unknown policy kind: {kind}")


def expected_features(pi: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """E_{a~pi}[phi(a)] = pi . Phi   (shape (d,))."""
    return pi @ Phi


def sample_pairs(pi: np.ndarray, q: int, rng: np.random.Generator) -> np.ndarray:
    """Draw q comparison pairs, each two independent arms ~ pi, i != j."""
    n = len(pi)
    first = rng.choice(n, size=q, p=pi)
    second = rng.choice(n, size=q, p=pi)
    # resolve self-comparisons
    collide = np.where(first == second)[0]
    guard = 0
    while len(collide) > 0 and guard < 50:
        second[collide] = rng.choice(n, size=len(collide), p=pi)
        collide = collide[first[collide] == second[collide]]
        guard += 1
    return np.stack([first, second], axis=1)
