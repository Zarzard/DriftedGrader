"""C1. Teacher and population.

Model (spec Section 2, note Sections 1-3):
    utility      u_t(x) = w_t . phi(x) + s_t
    label law    P_t(x > y) = sigma(beta_t (u_t(x) - u_t(y))) = sigma(v_t . dphi)
    identifiable state  v_t := beta_t * w_t         (Prop 2)

``s_t`` (severity) is carried explicitly but MUST NOT enter any within-sitting
comparison label -- it cancels in the utility difference (Prop 1). It exists so
the unit tests can verify that cancellation, and so ``label_from_scores`` can
demonstrate the contamination of cross-context / score-derived pairs (Cor 1).

M0 scope: the ``Teacher`` state container, ``v()``, within-sitting ``label`` /
``prob``, and the score-derived ``prob_from_scores`` / ``label_from_scores`` used
by tests 1-3. Drift (C2), populations (churn), and defaults (d=24, S_hist/S_nov)
arrive in M1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._math import sigmoid


@dataclass
class Teacher:
    """A single rater state ``(w, s, beta)`` exposing the identifiable ``v = beta*w``."""

    w: np.ndarray
    s: float = 0.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=float)
        self.s = float(self.s)
        self.beta = float(self.beta)

    # --- identifiable state -------------------------------------------------
    def v(self) -> np.ndarray:
        """Identifiable rater state v = beta * w (Prop 2)."""
        return self.beta * self.w

    # --- within-sitting comparison (severity-immune, Prop 1) ----------------
    def prob(self, dphi: np.ndarray) -> float:
        """P(x > y) for a same-sitting comparison with feature gap ``dphi``.

        Depends on ``(w, beta)`` only through ``v``; independent of ``s`` (Prop 1).
        """
        dphi = np.asarray(dphi, dtype=float)
        return float(sigmoid(self.v() @ dphi))

    def label(self, dphi: np.ndarray, rng: np.random.Generator) -> int:
        """Bernoulli(sigma(v . dphi)) label of a same-sitting comparison."""
        return int(rng.random() < self.prob(dphi))

    # --- score-derived / cross-context pair (severity-contaminated, Cor 1) --
    def prob_from_scores(
        self,
        phi_x: np.ndarray,
        phi_y: np.ndarray,
        s_x: float,
        s_y: float,
    ) -> float:
        """P(x > y) when x and y were scored under DIFFERENT severity states.

        Synthesizing a pair from absolute scores collected at different times
        gives utility difference ``w.phi_x + s_x - (w.phi_y + s_y)``; the severity
        difference ``s_x - s_y`` survives the link (Cor 1). This is the
        severity-contaminated construction the note warns about; it uses this
        teacher's ``(w, beta)`` with the two supplied severity offsets.
        """
        phi_x = np.asarray(phi_x, dtype=float)
        phi_y = np.asarray(phi_y, dtype=float)
        u_diff = (self.w @ phi_x + s_x) - (self.w @ phi_y + s_y)
        return float(sigmoid(self.beta * u_diff))

    def label_from_scores(
        self,
        phi_x: np.ndarray,
        phi_y: np.ndarray,
        s_x: float,
        s_y: float,
        rng: np.random.Generator,
    ) -> int:
        """Bernoulli label of a severity-contaminated, score-derived pair (Cor 1)."""
        return int(rng.random() < self.prob_from_scores(phi_x, phi_y, s_x, s_y))


@dataclass
class Population:
    """A pool of raters with mixture weights (M0 stub; churn lands in M1).

    A comparison is labeled by a single sampled rater, so per-rater severity
    cancellation (Prop 1) holds by construction. Default ``M = 1`` reduces to a
    single ``Teacher``.
    """

    teachers: list[Teacher]
    weights: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        m = len(self.teachers)
        if m == 0:
            raise ValueError("Population needs at least one teacher")
        if self.weights is None:
            self.weights = np.full(m, 1.0 / m)
        else:
            self.weights = np.asarray(self.weights, dtype=float)
            self.weights = self.weights / self.weights.sum()

    def sample_rater(self, rng: np.random.Generator) -> Teacher:
        idx = int(rng.choice(len(self.teachers), p=self.weights))
        return self.teachers[idx]

    def v_mean(self) -> np.ndarray:
        """Mixture-mean identifiable state E_r[v_r] (the S3 estimand, per audit fix #9)."""
        return np.average([t.v() for t in self.teachers], axis=0, weights=self.weights)

    def label(self, dphi: np.ndarray, rng: np.random.Generator) -> int:
        """Label a comparison by one sampled rater (severity cancels per rater)."""
        return self.sample_rater(rng).label(dphi, rng)
