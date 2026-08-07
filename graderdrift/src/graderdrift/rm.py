"""C8 reward model + E5 data-curation arms (spec v0.2 C8, E5).

P(x > y) = sigma(w_hat . dphi), L2 lambda = 1e-3, warm-started across rounds. The
fitted w_hat's rank deficiency along the achievable-gap null space is pinned by the
L2 penalty (amendment A6) and is policy-irrelevant.

E5 arms (matched label budget B/round):
- "accumulate" (default): fit on every pair ever collected.
- "window": keep only the last W rounds of pairs.
- "refresh": rolling buffer capped at W rounds; each round spend f*B fresh labels +
  (1-f)*B RE-LABELING the oldest retained gaps with the CURRENT teacher. Relabeling
  is done by the caller (it holds the teacher); the RM just stores per-round buffers
  and fits on the retained window.
"""
from __future__ import annotations

import numpy as np

from .diagnostics import fit_logistic_direction


class RewardModel:
    """Per-round pair buffers; fits on the arm-appropriate subset."""

    def __init__(self, d: int, arm: str = "accumulate", window: int = 10) -> None:
        self.d = d
        self.arm = arm
        self.window = window
        self._rounds: list[tuple[np.ndarray, np.ndarray]] = []  # (gaps, labels) per round
        self.w_hat = np.zeros(d)

    def add_round(self, gaps: np.ndarray, labels: np.ndarray) -> None:
        self._rounds.append((np.asarray(gaps, dtype=float), np.asarray(labels, dtype=int)))

    def _training_set(self):
        if self.arm == "accumulate":
            rounds = self._rounds
        else:  # window and refresh both keep only the last `window` rounds
            rounds = self._rounds[-self.window:]
        X = np.vstack([g for g, _ in rounds])
        y = np.concatenate([lab for _, lab in rounds]).astype(float)
        return X, y

    def oldest_retained_gaps(self, n: int) -> np.ndarray | None:
        """The n oldest gaps within the current retention window (for refresh
        re-labeling). Returns None if fewer than one retained round exists."""
        rounds = self._rounds[-self.window:]
        if not rounds:
            return None
        allg = np.vstack([g for g, _ in rounds])
        return allg[:n]

    def fit(self) -> np.ndarray:
        X, y = self._training_set()
        self.w_hat = fit_logistic_direction(X, y, w_init=self.w_hat)
        return self.w_hat

    @property
    def n_pairs(self) -> int:
        return sum(len(lab) for _, lab in self._rounds)
