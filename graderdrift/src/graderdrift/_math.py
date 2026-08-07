"""Elementary numerics shared across components.

The Bradley-Terry link and its derivative appear everywhere; centralize them so
the constants (z* = 1.5434, p* = 0.8237) have a single source of truth.
"""
from __future__ import annotations

import numpy as np

# Norm-sensitivity optimum: |z * sigma'(z)| is maximized at |z*|, giving p*.
# z* is the positive root of z * (2*sigma(z) - 1) = 1.
Z_STAR = 1.5434
P_STAR = 0.8237


def sigmoid(z):
    """Logistic link sigma(z) = 1/(1+exp(-z)), overflow-safe and vectorized."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out[()] if out.ndim == 0 else out


def sigmoid_prime(z):
    """sigma'(z) = sigma(z) (1 - sigma(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


def logit(p):
    """Inverse link log(p/(1-p))."""
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)
