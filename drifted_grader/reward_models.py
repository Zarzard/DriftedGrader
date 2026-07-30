"""Learned reward models fit by Bradley-Terry MLE.

Primary setting is now the NON-LINEAR MLP RM (plan reframe after the Tandem-RL
connection, Ostrovski et al. NeurIPS 2021): over-optimisation in a finite-arm
bandit requires non-linear function approximation. Measured:
  - LinearBT on the full basis (k=d): regret <= ~0.01 across the pressure sweep
    (null N6 -- "the RM class is not at fault" is provable). This is the CONTROL.
  - MLPBT: real over-optimisation (regret ~0.2), attenuating with more data. This
    is the (F) factor and the primary over-opt source.

The RM sees phi (full d) by default. An optional projection P (k<d) reduces it to
zeta = P phi -- kept as a STRUCTURAL-MISSPECIFICATION ablation, no longer primary.

Unified fit interface: fit_buffer(Phi, pairs, y) where `pairs` are (M,2) arm
indices and y in {0,1} (1 = first preferred). MLP needs the pair features (r_i-r_j
!= MLP(phi_i-phi_j)); LinearBT reduces to fitting on differences. Uses
scipy.special.expit / np.logaddexp throughout (numerical-stability requirement).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


# --------------------------------------------------------------------------
# projection (structural-misspecification ablation)
# --------------------------------------------------------------------------
def make_projection(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Row-orthonormal k x d projection; k>=d returns identity (full basis)."""
    if k >= d:
        return np.eye(d)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].T.copy()


def _apply_P(Phi, P):
    return Phi if P is None else Phi @ P.T


# --------------------------------------------------------------------------
# Linear BT (clean control, N6)
# --------------------------------------------------------------------------
def _bt_nll_and_grad(psi, dz, y, reg):
    z = dz @ psi
    nll = np.sum(y * np.logaddexp(0.0, -z) + (1.0 - y) * np.logaddexp(0.0, z))
    nll += 0.5 * reg * np.dot(psi, psi)
    grad = dz.T @ (expit(z) - y) + reg * psi
    return nll, grad


class LinearBT:
    def __init__(self, d: int, P: Optional[np.ndarray] = None, reg: float = 1e-3):
        self.d = int(d)
        self.P = P
        self.k = self.d if P is None else P.shape[0]
        self.reg = float(reg)
        self.psi = np.zeros(self.k)
        self.fit_loss = float("nan")

    def fit(self, diffs_full: np.ndarray, y: np.ndarray):
        """Fit on precomputed FULL-basis difference features (used by tests)."""
        if diffs_full is None or len(diffs_full) == 0:
            self.psi = np.zeros(self.k); self.fit_loss = float("nan"); return self
        dz = _apply_P(np.asarray(diffs_full, float), self.P)
        y = np.asarray(y, float)
        res = minimize(lambda p: _bt_nll_and_grad(p, dz, y, self.reg), self.psi,
                       jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.psi = res.x
        self.fit_loss = float(res.fun) / max(1, len(y))
        return self

    def fit_buffer(self, Phi, pairs, y, rng=None):
        diffs = Phi[pairs[:, 0]] - Phi[pairs[:, 1]]
        return self.fit(diffs, y)

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        return _apply_P(Phi, self.P) @ self.psi

    def effective_weight(self) -> np.ndarray:
        return self.psi if self.P is None else self.P.T @ self.psi


# --------------------------------------------------------------------------
# MLP BT (non-linear FA -- the (F) over-optimisation source, PRIMARY)
# --------------------------------------------------------------------------
class MLPBT:
    """2-layer tanh MLP reward, r(a) = w2 . tanh(W1 x(a) + b1) + b2, x = phi or Pphi.
    Fit by BT MLE with Adam. Warm-starts across fit_buffer calls (online refit)."""

    def __init__(self, d: int, P: Optional[np.ndarray] = None, hidden: int = 64,
                 reg: float = 1e-4, steps: int = 80, lr: float = 5e-2,
                 seed: int = 0):
        self.d = int(d); self.P = P
        self.in_dim = d if P is None else P.shape[0]
        self.h = int(hidden); self.reg = float(reg)
        self.steps = int(steps); self.lr = float(lr)
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((self.h, self.in_dim)) * np.sqrt(1.0 / self.in_dim)
        self.b1 = np.zeros(self.h)
        self.w2 = rng.standard_normal(self.h) * np.sqrt(1.0 / self.h)
        self.b2 = 0.0
        self._adam = None
        self.fit_loss = float("nan")

    def _forward(self, X):
        Z = X @ self.W1.T + self.b1
        H = np.tanh(Z)
        return H @ self.w2 + self.b2, H

    def predict(self, Phi):
        return self._forward(_apply_P(Phi, self.P))[0]

    def fit_buffer(self, Phi, pairs, y, rng=None):
        if pairs is None or len(pairs) == 0:
            return self
        X = _apply_P(Phi, self.P)
        i, j = pairs[:, 0], pairs[:, 1]
        y = np.asarray(y, float)
        params = [self.W1, self.b1, self.w2, np.array(float(self.b2))]
        if self._adam is None:
            self._adam = ([np.zeros_like(p) for p in params],
                          [np.zeros_like(p) for p in params], 0)
        m, v, tstart = self._adam
        b1_, b2_ = 0.9, 0.999
        for s in range(1, self.steps + 1):
            r, H = self._forward(X)
            g = expit(r[i] - r[j]) - y
            dr = np.zeros(len(X)); np.add.at(dr, i, g); np.add.at(dr, j, -g)
            dw2 = H.T @ dr + self.reg * self.w2
            db2 = dr.sum()
            dZ = (dr[:, None] * self.w2[None, :]) * (1 - H ** 2)
            dW1 = dZ.T @ X + self.reg * self.W1
            db1 = dZ.sum(0)
            grads = [dW1, db1, dw2, np.array(db2)]
            t = tstart + s
            for k, (pp, gg) in enumerate(zip(params, grads)):
                m[k] = b1_ * m[k] + (1 - b1_) * gg
                v[k] = b2_ * v[k] + (1 - b2_) * gg ** 2
                pp -= self.lr * (m[k] / (1 - b1_ ** t)) / (np.sqrt(v[k] / (1 - b2_ ** t)) + 1e-8)
        self.W1, self.b1, self.w2, self.b2 = params[0], params[1], params[2], float(params[3])
        self._adam = (m, v, tstart + self.steps)
        r, _ = self._forward(X)
        z = r[i] - r[j]
        self.fit_loss = float(np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)))
        return self


# --------------------------------------------------------------------------
# Ensemble (D2 disagreement); base = linear (bootstrap) or mlp (init diversity)
# --------------------------------------------------------------------------
class EnsembleBT:
    def __init__(self, d: int, P: Optional[np.ndarray] = None, K: int = 5,
                 reg: float = 1e-3, base: str = "linear", hidden: int = 64,
                 steps: int = 80):
        self.d = int(d); self.P = P; self.K = int(K); self.base = base
        if base == "linear":
            self.heads = [LinearBT(d, P, reg) for _ in range(self.K)]
        else:
            self.heads = [MLPBT(d, P, hidden=hidden, reg=reg, steps=steps, seed=h)
                          for h in range(self.K)]
        self.fit_loss = float("nan")

    def fit_buffer(self, Phi, pairs, y, rng):
        if pairs is None or len(pairs) == 0:
            return self
        y = np.asarray(y, float); m = len(y); losses = []
        for h in self.heads:
            idx = rng.integers(0, m, size=m) if self.base == "linear" else np.arange(m)
            h.fit_buffer(Phi, pairs[idx], y[idx], rng)
            losses.append(h.fit_loss)
        self.fit_loss = float(np.nanmean(losses))
        return self

    def predict(self, Phi):
        return np.mean([h.predict(Phi) for h in self.heads], axis=0)

    def predict_all(self, Phi):
        return np.stack([h.predict(Phi) for h in self.heads], axis=0)

    def disagreement(self, Phi, standardize=True):
        preds = self.predict_all(Phi)
        if standardize:
            sd = preds.std(axis=1, keepdims=True)
            preds = (preds - preds.mean(axis=1, keepdims=True)) / np.where(sd < 1e-12, 1.0, sd)
        return preds.std(axis=0)


def make_reward_model(kind: str, d: int, P: Optional[np.ndarray] = None, K: int = 5,
                      reg: float = 1e-3, hidden: int = 64, steps: int = 80):
    kind = kind.lower()
    if kind == "linear":
        return LinearBT(d, P=P, reg=reg)
    if kind == "mlp":
        return MLPBT(d, P=P, hidden=hidden, reg=reg, steps=steps)
    if kind in ("ensemble", "ensemble_linear"):
        return EnsembleBT(d, P=P, K=K, reg=reg, base="linear")
    if kind == "ensemble_mlp":
        return EnsembleBT(d, P=P, K=K, reg=reg, base="mlp", hidden=hidden, steps=steps)
    raise ValueError(f"unknown reward model kind: {kind}")
