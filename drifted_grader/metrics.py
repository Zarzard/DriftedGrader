"""Primary metrics (plan Section 3.6). Build and test this module FIRST; N7
(oracle-tracking regret floor) gates everything downstream.

R_t  = regret against the grader CURRENTLY being served, at matched pressure:
         pi*_t = Gibbs(rangenorm(Phi w_t), beta)      # best achievable at THIS beta
         R_t   = [ pi*_t.(Phi w_t) - pi_t.(Phi w_t) ] / range_t
       measured *within* one grader, so no "which preferences are real" choice
       enters. R_t >= 0 always; R_t == 0 when the agent is handed w_t exactly (N7).

range-normalisation: beta is a pressure knob only if agent and oracle scores share
a scale (see conversation / EXPERIMENT_LOG). Both are divided by their own arm-range
before Gibbs. rangenorm is monotone so it never changes an argmax (P2 safe) and it
makes pi*_t == pi_t identically when the agent uses w_t (N7 exact).

A_t  = |u_t . v_t|, drift direction vs least-informed comparison direction
       (Section 3.6 mechanism metric). A_null uses the EXACT E|u.v| for random
       unit vectors, not the asymptotic sqrt(2/(pi d)) (measured 4% low at d=6).

proj_angle = angle between u_t and P's row space (RM's structural blind spot).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .policies import gibbs_policy


# --------------------------------------------------------------------------
# range normalisation (shared by agent + oracle so beta is a pure pressure knob)
# --------------------------------------------------------------------------
def score_range(scores: np.ndarray) -> float:
    return float(scores.max() - scores.min())


def rangenorm(scores: np.ndarray) -> np.ndarray:
    r = scores.max() - scores.min()
    if r < 1e-12:
        return scores - scores.mean()
    return scores / r


def policy_from_scores(scores: np.ndarray, beta: float) -> np.ndarray:
    """The canonical agent/oracle policy: Gibbs on range-normalised scores."""
    return gibbs_policy(rangenorm(scores), beta)


# --------------------------------------------------------------------------
# R_t: regret vs the same-beta oracle that knows w_t
# --------------------------------------------------------------------------
def oracle_policy(Phi: np.ndarray, w: np.ndarray, beta: float) -> np.ndarray:
    return policy_from_scores(Phi @ w, beta)


def regret_Rt(pi: np.ndarray, Phi: np.ndarray, w: np.ndarray,
              beta: float) -> tuple:
    """Return (R_t, range_t). R_t is normalised regret against the same-beta
    oracle for grader weights w."""
    true = Phi @ w
    range_t = float(true.max() - true.min())
    if range_t < 1e-12:
        return 0.0, range_t
    pi_star = policy_from_scores(true, beta)
    R = float((pi_star @ true - pi @ true) / range_t)
    # numerical floor: tiny negatives from float error are clamped to 0
    if -1e-9 < R < 0:
        R = 0.0
    return R, range_t


# --------------------------------------------------------------------------
# A_t: drift/blind-spot alignment + its analytic null
# --------------------------------------------------------------------------
def A_null(d: int) -> float:
    """Exact E|u.v| for u,v iid uniform on the unit sphere S^{d-1}:
        E|X| = (2/sqrt(pi)) * Gamma(d/2) / ((d-1) * Gamma((d-1)/2)).
    (The plan's sqrt(2/(pi d)) is only the large-d asymptote; ~4% low at d=6.)"""
    if d < 2:
        return float("nan")
    return float((2.0 / np.sqrt(np.pi))
                 * np.exp(gammaln(d / 2.0) - np.log(d - 1) - gammaln((d - 1) / 2.0)))


def drift_direction(w_t: np.ndarray, w_next: np.ndarray):
    """Unit drift direction u_t = (w_{t+1}-w_t)/||.||, or None if no drift."""
    du = np.asarray(w_next) - np.asarray(w_t)
    nrm = float(np.linalg.norm(du))
    if nrm < 1e-12:
        return None
    return du / nrm


def least_informed_direction(diffs: np.ndarray):
    """Smallest-eigenvalue eigenvector of the comparison DESIGN matrix
    D = (1/M) sum_i (phi_a-phi_b)(phi_a-phi_b)^T  over the round's comparisons.

    We use the uncentered second moment (the BT information geometry), not a
    mean-subtracted covariance: it is the object whose small eigenvalue is the
    direction the data cannot constrain. (Pairs are two iid draws from pi, so the
    mean difference is ~0 anyway and centred/uncentred coincide -- see log.)
    Returns (v_min, eigvals_ascending)."""
    M = diffs.shape[0]
    D = (diffs.T @ diffs) / max(1, M)
    vals, vecs = np.linalg.eigh(D)          # ascending
    return vecs[:, 0], vals


def A_t(u_t: np.ndarray, v_t: np.ndarray) -> float:
    if u_t is None or v_t is None:
        return float("nan")
    return float(abs(np.dot(u_t, v_t)))


def variance_ratio(u_t: np.ndarray, diffs: np.ndarray) -> float:
    """VR = (u^T C u) / (tr(C)/d), C = D^T D / M  (comparison design matrix).

    Variance of the comparison differences ALONG the drift direction u_t, relative
    to a random direction (tr C / d). VR < 1 => drift points into LOW-information
    (blind) directions; null = 1.0. This is the primary mechanism metric: unlike
    the single-eigenvector A_t, it captures the whole low-variance SUBSPACE, so it
    does not wash into the null when the blind spot is multi-dimensional.
    """
    if u_t is None or diffs is None or len(diffs) == 0:
        return float("nan")
    M, d = diffs.shape
    C = (diffs.T @ diffs) / M
    denom = np.trace(C) / d
    if denom < 1e-12:
        return float("nan")
    return float((u_t @ C @ u_t) / denom)


# --------------------------------------------------------------------------
# proj_angle: drift vs the RM's structural blind spot (P's row space)
# --------------------------------------------------------------------------
def proj_blindspot_fraction(u_t: np.ndarray, P: np.ndarray) -> float:
    """Fraction of the unit drift direction lying OUTSIDE P's row space
    (the RM's structural blind spot). P is (k,d) row-orthonormal so the
    projector onto row(P) is P^T P."""
    if u_t is None or P is None:
        return float("nan")
    proj = P.T @ (P @ u_t)                   # component inside row(P)
    blind = u_t - proj
    frac = float(np.linalg.norm(blind))      # ||u||=1 so this is in [0,1]
    return min(1.0, max(0.0, frac))


# --------------------------------------------------------------------------
# Plan 02: reward-posterior coverage metrics (Section 4)
# --------------------------------------------------------------------------
def coverage_hit(v_true: float, lower_bound: float) -> int:
    """1 if the true value clears the high-confidence lower bound (Section 4.1)."""
    return int(v_true >= lower_bound)


def bound_width(post_mean_value: float, lower_bound: float, range_t: float) -> float:
    """W_t = (posterior mean value - LB) / range_t: how much the bound gives up to
    be safe. Falling W_t while coverage falls = the dangerous quadrant (Section 4.2)."""
    if range_t < 1e-12:
        return float("nan")
    return float((post_mean_value - lower_bound) / range_t)


def rank_correlation(lower_bounds, true_values) -> float:
    """Spearman rho between {LB_alpha(pi^j)} and {V_true(pi^j)} over a fixed set of
    evaluation policies (Section 4.1). A bound that loses ranking validity is broken
    for its actual use case (ranking policies by risk) even if marginal coverage holds."""
    lb = np.asarray(lower_bounds, float)
    tv = np.asarray(true_values, float)
    if len(lb) < 3 or np.std(lb) < 1e-12 or np.std(tv) < 1e-12:
        return float("nan")
    # Spearman = Pearson on ranks
    rl = np.argsort(np.argsort(lb)).astype(float)
    rt = np.argsort(np.argsort(tv)).astype(float)
    return float(np.corrcoef(rl, rt)[0, 1])


def proj_angle(u_t: np.ndarray, P: np.ndarray) -> float:
    """Angle (radians in [0, pi/2]) between u_t and P's row space.
    0 => drift fully visible to the RM; pi/2 => drift entirely in the blind spot."""
    frac = proj_blindspot_fraction(u_t, P)
    if not np.isfinite(frac):
        return float("nan")
    return float(np.arcsin(frac))
