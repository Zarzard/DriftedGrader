"""C6. Operational diagnostics -- the "naive detector" family (spec v0.2 C6).

Computed ONLY from the operational stream (never from anchors). These are the
kappa-hat analogues; E1 exists to show they move under a stationary rater (S0),
which no operational statistic can distinguish from genuine drift (note Prop 3).

- op_agree / op_flip: agreement / flip rate on the double-labeled pairs.
- op_temp_inv_std (PRIMARY, amendment A1): fit a scalar temperature on within-round
  STANDARDIZED margins (m_i = (what.dphi_i)/std_round). Under S0 this rises ~ 1/c_k
  as gaps shrink -- the aliasing statistic dashboards actually watch.
- op_temp_inv_raw (SECONDARY): the same fit on unstandardized margins. With observed
  gaps and a fitted direction the temperature is identified (~||v||) and stays flat
  under S0. The divergence of the two under a constant rater is the finding.

``what`` (the reward-model direction) is a running L2-logistic fit on accumulated
operational pairs (A0); the A1 loop supplies the RM instead.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ._math import sigmoid

L2_LAMBDA = 1e-3


def op_agree(dup_labels_a: np.ndarray, dup_labels_b: np.ndarray) -> float:
    """Agreement rate on the double-labeled pairs."""
    if len(dup_labels_a) == 0:
        return float("nan")
    return float(np.mean(dup_labels_a == dup_labels_b))


def fit_logistic_direction(gaps: np.ndarray, labels: np.ndarray, lam: float = L2_LAMBDA,
                           w_init: np.ndarray | None = None) -> np.ndarray:
    """L2-regularized logistic MLE of the reward direction on (gaps, labels).

    Model P(y=1) = sigma(w . dphi); returns the fitted w (NOT normalized -- its norm
    is the estimated ||v||, which op_temp_inv_raw relies on).
    """
    X = np.asarray(gaps, dtype=float)
    y = np.asarray(labels, dtype=float)
    d = X.shape[1]
    w0 = np.zeros(d) if w_init is None else np.asarray(w_init, dtype=float)

    def nll_and_grad(w):
        z = X @ w
        p = sigmoid(z)
        # stable nll
        nll = float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * (w @ w))
        grad = X.T @ (p - y) + lam * w
        return nll, grad

    res = minimize(nll_and_grad, w0, jac=True, method="L-BFGS-B")
    return res.x


def _temp_inv(margins: np.ndarray, labels: np.ndarray,
              tau_bounds: tuple[float, float] = (1e-2, 1e2)) -> float:
    """1/tau_hat where tau_hat maximizes sum log sigma(tau * margin * (2y-1)).

    Bounds keep the boundary blow-up finite when the round's labels carry little
    signal (tiny gaps): 1/tau is capped at 1/tau_bounds[0]. Late-round estimates
    are still high-variance -- callers should aggregate robustly and report the
    identifiability of each round (see ``round_info``).
    """
    s = margins * (2.0 * np.asarray(labels, dtype=float) - 1.0)

    def nll(tau):
        return -np.sum(np.log(sigmoid(tau * s) + 1e-300))

    res = minimize_scalar(nll, bounds=tau_bounds, method="bounded")
    return 1.0 / float(res.x)


def round_info(gaps: np.ndarray, what: np.ndarray) -> float:
    """Fisher information the round carries about the temperature at tau=1:
    sum_i margin_i^2 * sigma'(margin_i). Small => temperature ~unidentified."""
    m = np.asarray(gaps, dtype=float) @ np.asarray(what, dtype=float)
    return float(np.sum(m * m * (sigmoid(m) * (1.0 - sigmoid(m)))))


def op_temp_inv(gaps: np.ndarray, labels: np.ndarray, what: np.ndarray) -> tuple[float, float]:
    """Return (op_temp_inv_std, op_temp_inv_raw) for one round given RM direction ``what``.

    std: margins standardized by their within-round std (rises ~1/c_k under S0).
    raw: unstandardized margins (identified; flat under S0).
    """
    raw_margins = np.asarray(gaps, dtype=float) @ np.asarray(what, dtype=float)
    sd = raw_margins.std()
    if sd < 1e-12:
        return float("nan"), float("nan")
    std_margins = raw_margins / sd
    return _temp_inv(std_margins, labels), _temp_inv(raw_margins, labels)
