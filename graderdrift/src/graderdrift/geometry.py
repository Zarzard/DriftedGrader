"""Subspace projectors and the on-support fraction rho.

Spec v0.2 Section 2:
    rho(delta) = ||P_A delta||^2 / ||delta||^2,   P_A = projector onto rowspace(Phi_A)
    recovered-norm identity: ||P_A delta|| = sqrt(rho) * ||delta||

``P_A`` is the achievable-gap rowspace projector. In A0 it equals the projector
onto S_hist (the battery is built from historical items, rank 16 = dim S_hist);
in A1 it is the rank-6 achievable-gap projector (C8). The rho function is agnostic
to which -- it takes an explicit projector, so tests can supply P_{S_hist} and the
estimator can supply the empirical rowspace projector.
"""
from __future__ import annotations

import numpy as np

_ZERO = 1e-12


def subspace_projector(indices: np.ndarray | list[int], d: int) -> np.ndarray:
    """Orthogonal projector onto the coordinate subspace ``span(e_i : i in indices)``."""
    P = np.zeros((d, d))
    for i in indices:
        P[i, i] = 1.0
    return P


def projector_from_rowspace(Phi: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Projector onto rowspace(Phi) via SVD; singular values below ``tol`` dropped.

    ``Phi`` is (m x d); rowspace lives in R^d. Returns a (d x d) projector.
    """
    Phi = np.atleast_2d(np.asarray(Phi, dtype=float))
    _, s, vt = np.linalg.svd(Phi, full_matrices=False)
    keep = s > tol
    V = vt[keep]  # (r x d), orthonormal rows
    if V.shape[0] == 0:
        return np.zeros((Phi.shape[1], Phi.shape[1]))
    return V.T @ V


def rho(delta: np.ndarray, P_A: np.ndarray) -> float:
    """On-support fraction ||P_A delta||^2 / ||delta||^2.

    Returns 0.0 for a (near-)zero increment -- no-drift rounds have delta = 0 and
    a well-defined rho of 0 rather than 0/0.
    """
    delta = np.asarray(delta, dtype=float)
    nd2 = float(delta @ delta)
    if nd2 < _ZERO:
        return 0.0
    pd = P_A @ delta
    return float((pd @ pd) / nd2)


def on_support_norm(delta: np.ndarray, P_A: np.ndarray) -> float:
    """||P_A delta|| = sqrt(rho) * ||delta||."""
    delta = np.asarray(delta, dtype=float)
    pd = P_A @ delta
    return float(np.sqrt(pd @ pd))
