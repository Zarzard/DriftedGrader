"""Grader models: a preference oracle whose weights (or reported scale) drift
as a function of the agent's achieved behaviour.

Bradley-Terry label (Section 3.2):
    P(a > b | w_t) = sigmoid( betaT * ( w_t.phi(a) - w_t.phi(b) ) )

Convention warning (plan Section 3.2): betaT here is a *positive* inverse
temperature (our default 10 == B-Pref teacher_beta=10).  Do NOT use B-Pref's
teacher_beta=-1 "perfectly rational" sentinel.

Drift models (Section 3.3):
  DriftNone   lambda = 0, w constant. The null.
  DriftSat    satiation / dimension reweighting -> ORDER-CHANGING (primary).
  DriftScale  reported-scale recentring -> ORDER-PRESERVING control (H2/N2).
  DriftExo    fixed exogenous w schedule, independent of agent.
  DriftReplay open-loop replay of a recorded DriftSat w-trajectory (H3/N5).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .arms import Arms


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class Grader(ABC):
    """Preference oracle over arms.  Subclasses implement `step`."""

    name = "base"

    def __init__(self, arms: Arms, betaT: float):
        self.Phi = arms.Phi
        self.w0 = arms.w0.copy()
        self.per_dim_max = arms.per_dim_max
        self.betaT = float(betaT)
        self.w = arms.w0.copy()      # current *reference* weights w_t
        self.t = 0

    # ---- grader internal comparison scale --------------------------------
    def utilities(self) -> np.ndarray:
        """Raw per-arm utility w_t . phi(a)  (shape (n,))."""
        return self.Phi @ self.w

    def comp_scores(self) -> np.ndarray:
        """Per-arm score fed into the BT comparison (identity by default;
        DriftScale applies a monotone recentring transform)."""
        return self.utilities()

    # ---- labelling -------------------------------------------------------
    def prob_first(self, pairs: np.ndarray) -> np.ndarray:
        """P(first > second) for each row of `pairs` (M, 2) of arm indices."""
        s = self.comp_scores()
        logit = self.betaT * (s[pairs[:, 0]] - s[pairs[:, 1]])
        return sigmoid(logit)

    def label(self, pairs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sampled labels: 1 if first preferred, else 0.  (M,) int array."""
        p = self.prob_first(pairs)
        return (rng.random(len(pairs)) < p).astype(np.int64)

    # ---- drift -----------------------------------------------------------
    @abstractmethod
    def step(self, m: np.ndarray) -> None:
        """Advance w_t -> w_{t+1} using achieved profile m (EMA, d-vector)."""
        raise NotImplementedError


class DriftNone(Grader):
    name = "NONE"

    def step(self, m: np.ndarray) -> None:
        self.t += 1  # w unchanged


class DriftSat(Grader):
    """Satiation: sensitivity migrates away from mastered dimensions.

        w~_{t+1,i} = w_{t,i} * exp(-lambda * m-tilde_{t,i})
        w_{t+1}    = w~_{t+1} / sum(w~_{t+1})          (project to simplex)

    where m-tilde = m / per_dim_max in [0,1].  ORDER-CHANGING.
    """
    name = "SAT"

    def __init__(self, arms: Arms, betaT: float, lam: float, **kw):
        super().__init__(arms, betaT)
        self.lam = float(lam)

    def step(self, m: np.ndarray) -> None:
        m_tilde = np.clip(m / np.maximum(self.per_dim_max, 1e-12), 0.0, 1.0)
        w_new = self.w * np.exp(-self.lam * m_tilde)
        s = w_new.sum()
        if s < 1e-300:
            w_new = np.ones_like(w_new)
            s = w_new.sum()
        self.w = w_new / s
        self.t += 1


class DriftScale(Grader):
    """Ordering-preserving control: w == w0 fixed, but the reported comparison
    scale recentres around achieved quality.

        sigma_theta(x) = sigmoid(kappa * (x - theta))
        theta_{t+1}    = theta_t + lambda * (mean achieved quality - theta_t)

    'mean achieved quality' = w0 . m (achieved profile projected on w0).
    Monotone in x -> argmax invariant -> predicted no true-reward decline (H2).
    """
    name = "SCALE"

    def __init__(self, arms: Arms, betaT: float, lam: float, kappa: float = 4.0,
                 **kw):
        super().__init__(arms, betaT)
        self.lam = float(lam)
        self.kappa = float(kappa)
        # neutral init: mean quality under the uniform reference policy
        self.theta = float((arms.Phi @ arms.w0).mean())

    def comp_scores(self) -> np.ndarray:
        u = self.Phi @ self.w0  # w == w0 always
        return sigmoid(self.kappa * (u - self.theta))

    def step(self, m: np.ndarray) -> None:
        quality = float(self.w0 @ m)
        self.theta = self.theta + self.lam * (quality - self.theta)
        self.t += 1


class DriftExo(Grader):
    """Exogenous control: w follows a fixed schedule w0 -> w_end over T rounds,
    independent of the agent.  Isolates non-stationarity from endogeneity."""
    name = "EXO"

    def __init__(self, arms: Arms, betaT: float, T: int, rng: np.random.Generator,
                 lam: float = 1.0, **kw):
        super().__init__(arms, betaT)
        self.T = int(T)
        w_end = rng.dirichlet(np.ones(arms.d))
        # linear interpolation, w_0 = w0 ... w_{T-1} = w_end (convex -> simplex)
        fracs = np.linspace(0.0, 1.0, self.T)
        self.sched = [(1 - f) * self.w0 + f * w_end for f in fracs]
        self.w = self.sched[0].copy()

    def step(self, m: np.ndarray) -> None:
        self.t += 1
        self.w = self.sched[min(self.t, self.T - 1)].copy()


class DriftReplay(Grader):
    """Open-loop replay of a recorded {w_t} trajectory from a DriftSat run at
    the same seed (control for H3 / N5).  Ignores the fresh agent's m."""
    name = "REPLAY"

    def __init__(self, arms: Arms, betaT: float, w_traj, **kw):
        super().__init__(arms, betaT)
        self.w_traj = [np.asarray(w, dtype=float) for w in w_traj]
        self.T = len(self.w_traj)
        self.w = self.w_traj[0].copy()

    def step(self, m: np.ndarray) -> None:
        self.t += 1
        self.w = self.w_traj[min(self.t, self.T - 1)].copy()


_DRIFT_REGISTRY = {
    "NONE": DriftNone,
    "SAT": DriftSat,
    "SCALE": DriftScale,
    "EXO": DriftExo,
    "REPLAY": DriftReplay,
}


def make_grader(drift: str, arms: Arms, betaT: float, rng: np.random.Generator,
                lam: float = 0.0, T: int = 100, kappa: float = 4.0,
                w_traj=None) -> Grader:
    drift = drift.upper()
    if drift == "NONE":
        return DriftNone(arms, betaT)
    if drift == "SAT":
        return DriftSat(arms, betaT, lam=lam)
    if drift == "SCALE":
        return DriftScale(arms, betaT, lam=lam, kappa=kappa)
    if drift == "EXO":
        return DriftExo(arms, betaT, T=T, rng=rng, lam=lam)
    if drift == "REPLAY":
        if w_traj is None:
            raise ValueError("DRIFT-REPLAY requires a recorded w_traj")
        return DriftReplay(arms, betaT, w_traj=w_traj)
    raise ValueError(f"unknown drift model: {drift}")
