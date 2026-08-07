"""C3. A0 operational item streams (spec v0.2 Section 3, C3).

Operational pairs at round k have feature gaps
    dphi ~ N(0, c_k^2 Sigma_k),  c_k = c_0 * gamma^k  (c_0 = 2.0, gamma = 0.95),
the "JND concentration" knob: gap magnitudes shrink as the policy improves. Sigma_k
is identity on S_hist and zero on S_nov until round t_nov = 20, after which the
novel-dimension variance ramps linearly to 0.5 * c_k^2 by round 40 (the policy
starts exploring novel behavior). n_pairs = 100 per round; a fraction q_dup = 0.15
of pairs is labeled twice (independent draws) to feed the operational diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .drift import DriftContext

C0_DEFAULT = 2.0
GAMMA_DEFAULT = 0.95
T_NOV = 20
T_FULL = 40
N_PAIRS_DEFAULT = 100
Q_DUP_DEFAULT = 0.15


def gap_scale(k: int, c0: float = C0_DEFAULT, gamma: float = GAMMA_DEFAULT) -> float:
    """c_k = c0 * gamma^k."""
    return c0 * gamma ** k


def novel_ramp(k: int, t_nov: int = T_NOV, t_full: int = T_FULL) -> float:
    """Linear ramp 0 (k <= t_nov) -> 1 (k >= t_full) for the novel-dim variance."""
    if k <= t_nov:
        return 0.0
    if k >= t_full:
        return 1.0
    return (k - t_nov) / (t_full - t_nov)


def sigma_diag(k: int, ctx: DriftContext) -> np.ndarray:
    """Diagonal of Sigma_k: 1 on S_hist, 0.5 * ramp(k) on S_nov."""
    diag = np.zeros(ctx.d)
    diag[ctx.hist_idx] = 1.0
    diag[ctx.nov_idx] = 0.5 * novel_ramp(k)
    return diag


def sample_gaps(k: int, ctx: DriftContext, rng: np.random.Generator,
                n_pairs: int = N_PAIRS_DEFAULT, c0: float = C0_DEFAULT,
                gamma: float = GAMMA_DEFAULT) -> np.ndarray:
    """Draw ``n_pairs`` operational feature gaps for round k: (n_pairs, d)."""
    c_k = gap_scale(k, c0, gamma)
    std = c_k * np.sqrt(sigma_diag(k, ctx))  # per-coordinate std
    return rng.standard_normal((n_pairs, ctx.d)) * std[None, :]


@dataclass
class OperationalRound:
    """One round's operational pairs and (possibly duplicated) labels."""

    gaps: np.ndarray              # (n_pairs, d)
    labels: np.ndarray            # (n_pairs,) in {0,1}: teacher preferred x over y
    dup_index: np.ndarray         # indices of double-labeled pairs
    dup_labels: np.ndarray        # second independent label for those pairs


def sample_operational_round(k: int, ctx: DriftContext, teacher_label_fn,
                             rng_stream: np.random.Generator, rng_labels: np.random.Generator,
                             n_pairs: int = N_PAIRS_DEFAULT, q_dup: float = Q_DUP_DEFAULT,
                             c0: float = C0_DEFAULT, gamma: float = GAMMA_DEFAULT) -> OperationalRound:
    """Sample gaps and label them with ``teacher_label_fn(dphi, rng) -> {0,1}``.

    A ``q_dup`` fraction is labeled a second time with independent draws (in
    population mode, ``teacher_label_fn`` resamples the rater internally).
    """
    gaps = sample_gaps(k, ctx, rng_stream, n_pairs, c0, gamma)
    labels = np.array([teacher_label_fn(g, rng_labels) for g in gaps], dtype=int)
    n_dup = int(round(q_dup * n_pairs))
    dup_index = rng_stream.choice(n_pairs, size=n_dup, replace=False)
    dup_labels = np.array([teacher_label_fn(gaps[i], rng_labels) for i in dup_index], dtype=int)
    return OperationalRound(gaps=gaps, labels=labels, dup_index=dup_index, dup_labels=dup_labels)
