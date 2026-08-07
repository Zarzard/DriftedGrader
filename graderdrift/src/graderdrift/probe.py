"""C5. Probing protocol (spec v0.2 Section 3, C5).

Every round, after drift and before anything else uses the teacher, draw
``n_probe = 50`` labels per anchor from the current teacher state. Anchors never
enter RM training data. Readouts:

    D_dir(t)  = mean over direction anchors of |p_hat_j(t) - p_hat_j(0)|
    D_norm(t) = mean over norm anchors of sign(z_j(0)) * (logit p_hat_j(t) - logit p_hat_j(0))

with p_hat clipped to [0.02, 0.98] before the logit. Negative D_norm = anchors
sliding toward 0.5 (norm shrinking). The baseline p_hat_j(0) is itself a 50-draw
probe estimate (per lead: n_probe(0) stays 50).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ._math import logit
from .anchors import AnchorBattery

N_PROBE_DEFAULT = 50
_CLIP = (0.02, 0.98)


def probe_battery(battery,  # AnchorBattery | A1Battery -- anything with .gaps (m, d)
                  rng: np.random.Generator,
                  prob_fn: Callable[[np.ndarray], float] | None = None,
                  sample_label: Callable[[np.ndarray, np.random.Generator], int] | None = None,
                  n_probe: int = N_PROBE_DEFAULT) -> np.ndarray:
    """Return p_hat (m,) = mean of ``n_probe`` labels per anchor from the current teacher.

    Two modes:
    - ``prob_fn(dphi) -> p`` (fast, single-teacher): the teacher's exact label prob;
      draws are vectorized (rng.random(n_probe) < p).
    - ``sample_label(dphi, rng) -> {0,1}`` (population): resamples the rater per label.
    Exactly one must be provided.
    """
    if (prob_fn is None) == (sample_label is None):
        raise ValueError("provide exactly one of prob_fn (single teacher) or sample_label (population)")
    m = battery.gaps.shape[0]
    p_hat = np.empty(m)
    if prob_fn is not None:
        for j in range(m):
            p = prob_fn(battery.gaps[j])
            p_hat[j] = np.mean(rng.random(n_probe) < p)
    else:
        for j in range(m):
            g = battery.gaps[j]
            p_hat[j] = np.mean(np.fromiter((sample_label(g, rng) for _ in range(n_probe)),
                                           dtype=float, count=n_probe))
    return p_hat


def d_dir(p_hat_t: np.ndarray, p_hat_0: np.ndarray, dir_idx: np.ndarray) -> float:
    """Direction-panel readout: mean absolute probability change."""
    return float(np.mean(np.abs(p_hat_t[dir_idx] - p_hat_0[dir_idx])))


def d_norm(p_hat_t: np.ndarray, p_hat_0: np.ndarray, z0: np.ndarray,
           norm_idx: np.ndarray, clip: tuple[float, float] = _CLIP) -> float:
    """Norm-panel readout: mean sign-aligned logit change (negative = slide to 0.5)."""
    lo, hi = clip
    lt = logit(np.clip(p_hat_t[norm_idx], lo, hi))
    l0 = logit(np.clip(p_hat_0[norm_idx], lo, hi))
    return float(np.mean(np.sign(z0[norm_idx]) * (lt - l0)))
