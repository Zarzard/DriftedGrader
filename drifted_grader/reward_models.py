"""Learned reward models fit by Bradley-Terry MLE (plan Section 3.5).

**The RM is deliberately misspecified**: it sees a strictly poorer basis than the
grader scores with. A correctly-specified linear RM on a finite arm set cannot
over-optimise (measured: <=0.017 Gibbs, 0.000 best-of-n -- this is null N6). So:

    zeta(a) = P phi(a),   P in R^{k x d} row-orthonormal, k < d   (fixed per seed)
    r_hat(a) = psi . zeta(a) = (P^T psi) . phi(a)

The fit is a projection of the grader's true weighting onto row(P); optimisation
exploits the projection error. Sweep k in {1,2,3,4,6}; the effect deepens as k
shrinks and vanishes at k=d (N6). With k=d and P=I this reduces to the full basis.

BT model: P(i > j) = expit( psi . (zeta_i - zeta_j) ). Temperature absorbed into
psi scale (P1 checks direction recovery of P w via cosine). Uses scipy.special.expit
and np.logaddexp throughout (numerical-stability requirement, Section 7.4).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------
def make_projection(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Row-orthonormal k x d projection: first k rows of a random orthonormal
    basis of R^d. k == d returns an orthonormal (rotation) matrix; pass identity
    semantics via k == d then callers may treat row(P) == R^d."""
    if k >= d:
        return np.eye(d)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].T.copy()          # (k, d), orthonormal rows


# --------------------------------------------------------------------------
# BT likelihood
# --------------------------------------------------------------------------
def _bt_nll_and_grad(psi, dz, y, reg):
    """Negative BT log-likelihood + gradient (L2-regularised), on projected
    difference features dz = zeta_i - zeta_j."""
    z = dz @ psi
    nll = np.sum(y * np.logaddexp(0.0, -z) + (1.0 - y) * np.logaddexp(0.0, z))
    nll += 0.5 * reg * np.dot(psi, psi)
    p = expit(z)
    grad = dz.T @ (p - y) + reg * psi
    return nll, grad


class LinearBT:
    """Linear BT reward model on the projected basis zeta = P phi."""

    def __init__(self, d: int, P: Optional[np.ndarray] = None, reg: float = 1e-3):
        self.d = int(d)
        self.P = np.eye(self.d) if P is None else np.asarray(P, dtype=float)
        self.k = self.P.shape[0]
        self.reg = float(reg)
        self.psi = np.zeros(self.k)
        self.fit_loss = float("nan")

    def project(self, Phi: np.ndarray) -> np.ndarray:
        return Phi @ self.P.T                      # (n, k)

    def fit(self, diffs_full: np.ndarray, y: np.ndarray):
        """diffs_full is (M, d) = phi_i - phi_j in the FULL basis; projected
        internally to (M, k). y in {0,1}, y=1 means first preferred."""
        if diffs_full is None or len(diffs_full) == 0:
            self.psi = np.zeros(self.k)
            self.fit_loss = float("nan")
            return self
        dz = np.asarray(diffs_full, dtype=float) @ self.P.T
        y = np.asarray(y, dtype=float)
        res = minimize(lambda p: _bt_nll_and_grad(p, dz, y, self.reg),
                       self.psi, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.psi = res.x
        self.fit_loss = float(res.fun) / max(1, len(y))
        return self

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        return (Phi @ self.P.T) @ self.psi

    def effective_weight(self) -> np.ndarray:
        """P^T psi: the RM's implied reward direction in the FULL d-space."""
        return self.P.T @ self.psi


def _bootstrap_index(m: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, m, size=m)


class EnsembleBT:
    """K bootstrap LinearBT heads sharing the same projection P. Mean prediction
    for the policy; cross-head std (D2) for the disagreement detector."""

    def __init__(self, d: int, P: Optional[np.ndarray] = None, K: int = 5, reg: float = 1e-3):
        self.d = int(d)
        self.P = np.eye(self.d) if P is None else np.asarray(P, dtype=float)
        self.k = self.P.shape[0]
        self.K = int(K)
        self.reg = float(reg)
        self.heads = [LinearBT(d, self.P, reg) for _ in range(self.K)]
        self.fit_loss = float("nan")

    def fit(self, diffs_full: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        if diffs_full is None or len(diffs_full) == 0:
            for h in self.heads:
                h.psi = np.zeros(self.k)
            self.fit_loss = float("nan")
            return self
        diffs_full = np.asarray(diffs_full, dtype=float)
        y = np.asarray(y, dtype=float)
        m = len(y)
        losses = []
        for h in self.heads:
            idx = _bootstrap_index(m, rng)
            h.fit(diffs_full[idx], y[idx])
            losses.append(h.fit_loss)
        self.fit_loss = float(np.nanmean(losses))
        return self

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        return np.mean([h.predict(Phi) for h in self.heads], axis=0)

    def predict_all(self, Phi: np.ndarray) -> np.ndarray:
        return np.stack([h.predict(Phi) for h in self.heads], axis=0)

    def disagreement(self, Phi: np.ndarray, standardize: bool = True) -> np.ndarray:
        preds = self.predict_all(Phi)                 # (K, n)
        if standardize:
            sd = preds.std(axis=1, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            preds = (preds - preds.mean(axis=1, keepdims=True)) / sd
        return preds.std(axis=0)                      # (n,)


def make_reward_model(kind: str, d: int, P: Optional[np.ndarray] = None, K: int = 5,
                      reg: float = 1e-3):
    kind = kind.lower()
    if kind == "linear":
        return LinearBT(d, P=P, reg=reg)
    if kind == "ensemble":
        return EnsembleBT(d, P=P, K=K, reg=reg)
    raise ValueError(f"unknown reward model kind: {kind}")
