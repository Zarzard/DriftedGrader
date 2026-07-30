"""Outcome space: a finite set of arms with non-negative feature vectors.

Features are the *evaluative dimensions* the grader may care about
(task success, speed, tidiness, politeness, legibility, safety-margin, ...).

Per plan Section 3.1:
  - sample Phi ~ Uniform[0,1]^{n x d}
  - standardise each column to zero-mean/unit-variance
  - shift to be non-negative (per-column min -> 0)
  - global rescale so that  max_a w0.phi(a) - min_a w0.phi(a) == 1
    (fixes the achievable range at 1 so Delta is directly interpretable).

w0 (the grader's initial weights, on the d-simplex) is generated here too,
because the range-fix couples Phi and w0 and every drift condition that shares
a seed must share *both* Phi and w0 (Section 8, pairing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Arms:
    """A per-seed problem instance: features + initial grader weights."""
    Phi: np.ndarray          # (n, d) non-negative feature matrix
    w0: np.ndarray           # (d,) initial grader weights on the simplex
    per_dim_max: np.ndarray  # (d,) max_a Phi[:, i]  (for m-tilde normalisation)
    cond_Phi: float          # condition number of Phi (analysis covariate, C6)
    range0: float            # achievable range under w0 (== 1 after rescale)
    col_corr_mean_abs: float # mean |correlation| between feature columns

    @property
    def n(self) -> int:
        return self.Phi.shape[0]

    @property
    def d(self) -> int:
        return self.Phi.shape[1]


def make_w0(d: int, rng: np.random.Generator) -> np.ndarray:
    """Initial grader weights, uniform on the d-simplex."""
    return rng.dirichlet(np.ones(d))


def _induce_correlation(Phi: np.ndarray, corr: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Optionally mix columns so dimensions are correlated (kill-recovery knob,
    Section 5 Stage-A kill remedy 'stronger dimension correlation in Phi').

    corr == 0 leaves Phi untouched (the plan's default: iid uniform columns).
    corr in (0, 1] blends each column toward a shared random latent factor.
    """
    if corr <= 0.0:
        return Phi
    n, d = Phi.shape
    shared = rng.uniform(0.0, 1.0, size=(n, 1))
    return (1.0 - corr) * Phi + corr * shared


def make_arms(n: int, d: int, rng: np.random.Generator,
              corr: float = 0.0) -> Arms:
    """Generate a per-seed problem instance.  RNG draw order is fixed:
    w0 first, then Phi (so conditions sharing a seed share both)."""
    w0 = make_w0(d, rng)

    Phi = rng.uniform(0.0, 1.0, size=(n, d))
    Phi = _induce_correlation(Phi, corr, rng)

    # standardise columns to zero-mean/unit-variance
    mu = Phi.mean(axis=0, keepdims=True)
    sd = Phi.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Phi = (Phi - mu) / sd

    # shift each column to be non-negative (per-column min -> 0)
    Phi = Phi - Phi.min(axis=0, keepdims=True)

    # global rescale so the achievable range under w0 is exactly 1
    scores0 = Phi @ w0
    rng0 = float(scores0.max() - scores0.min())
    if rng0 < 1e-12:
        raise ValueError("degenerate arms: zero achievable range under w0")
    Phi = Phi / rng0

    per_dim_max = Phi.max(axis=0)
    cond_Phi = float(np.linalg.cond(Phi))

    # mean absolute off-diagonal column correlation (geometry covariate)
    if d > 1:
        C = np.corrcoef(Phi, rowvar=False)
        off = C[~np.eye(d, dtype=bool)]
        col_corr = float(np.mean(np.abs(off)))
    else:
        col_corr = 0.0

    scores0 = Phi @ w0
    range0 = float(scores0.max() - scores0.min())
    return Arms(Phi=Phi, w0=w0, per_dim_max=per_dim_max, cond_Phi=cond_Phi,
                range0=range0, col_corr_mean_abs=col_corr)
