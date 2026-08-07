"""C7. State-space drift estimator (spec v0.2 Section 3, C7).

An iterated EKF / Laplace filter tracking the latent grader state v in R^d from the
frozen anchors' binomial label counts. This is the "drift tracker": each round it
predicts v drifted a little (plus a jump if the calendar says so), then corrects
against the fresh anchor probes, returning a posterior mean m_k and covariance
Sigma_k with honest uncertainty.

Design (spec):
- Dynamics v_{k+1} = v_k + q_k, q_k ~ N(0, Q_k). Q scales relative to ||m_0||:
  known_calendar Q_k = (sig_diff^2 + 1[event] sig_jump^2) ||m_0||^2 I;
  blind Q_k = sig_blind^2 ||m_0||^2 I.  (sig_diff=.01, sig_jump=.5, sig_blind=.05)
- Measurement: y_j ~ Binomial(n_probe, sigma(v . g_j)). Update = 3 Newton iterations
  on the (concave) log-posterior; observed information gives Sigma. Innovation capped
  at 6 sigma / coordinate.
- Init m_0 = MAP to the round-0 probe under prior N(0, 10 I); Sigma_0 = Laplace Hessian.
- Derived: 90% CIs for P_A v and full v; ||v|| and angle intervals by 2000 draws.
- Primary drift statistic (E4): TV = sum_k ||m_k - m_{k-1}|| (full vector), minus the
  S0 noise floor. A direction-only TV is blind to pure norm drift (test 6c).

Also supports a linear-Gaussian measurement (y = H v + eps) so the Gaussian machinery
can be checked against an exact Kalman filter (test 6a).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from ._math import sigmoid

SIG_DIFF = 0.01
SIG_JUMP = 0.5
SIG_BLIND = 0.05


@dataclass
class GaussState:
    mean: np.ndarray
    cov: np.ndarray


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
def predict(state: GaussState, Q: np.ndarray) -> GaussState:
    return GaussState(state.mean.copy(), state.cov + Q)


def process_noise(mode: str, event: bool, m0_norm: float, d: int,
                  sig_diff: float = SIG_DIFF, sig_jump: float = SIG_JUMP,
                  sig_blind: float = SIG_BLIND) -> np.ndarray:
    """Q_k for the given mode (scales relative to ||m_0||)."""
    s2 = m0_norm ** 2
    if mode == "known_calendar":
        var = sig_diff ** 2 * s2 + (sig_jump ** 2 * s2 if event else 0.0)
    elif mode == "blind":
        var = sig_blind ** 2 * s2
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return var * np.eye(d)


def informed_process(sched, k: int, m_norm: float, m0_norm: float, d: int,
                     P_A: np.ndarray | None = None, sig_floor: float = 3e-4,
                     sig_jump: float = SIG_JUMP) -> tuple[float, np.ndarray]:
    """Generator-matched predict law for the ``informed`` calendar mode.

    Motivation (E3 finding, AMENDMENTS wave 3 C5): a FIXED sigma_diff cannot be
    calibrated per-schedule -- S0/S1/S5 over-cover (variance pumped into a static
    state) while S2/S4 under-cover (true per-round drift exceeds the floor) with the
    same constant. "Knowing the calendar" is therefore taken to mean knowing the
    drift process LAW (types, times, magnitudes -- as a platform knows its own
    rubric-edit/judge-swap/churn processes), not just event times; the realization
    stays unknown. Returns ``(A, Q)`` for the predict step
    ``m <- A m,  S <- A^2 S + Q``:

    - deterministic norm decay -> known mean map A = factor (a linear dynamics);
    - known direction diffusion -> Q matched to its per-round covariance, supported
      on the battery span (the generator's diffusion lives in S_hist = span(P_A));
    - jump-type events -> sigma_jump^2 at their round (as in the fixed mode);
    - mixture redraws -> variance from the redraw law: replacing a fraction f of M
      raters with vMF(kappa)/LogNormal draws moves the mixture mean by
      ~ f * sqrt(2(1/kappa) + 2 sig_ln^2) / sqrt(fM) * ||v|| per event;
    - a tiny floor sigma_floor keeps the filter from freezing.
    """
    A = 1.0
    Q = (sig_floor * m0_norm) ** 2 * np.eye(d)
    pr = sched.per_round
    decay = pr.get("norm_decay")
    if decay and decay.get("factor", 1.0) != 1.0:
        lo = decay.get("from") or 0
        hi = decay.get("to") or 10 ** 9
        if lo <= k <= hi:
            A = float(decay["factor"])
    sd = pr.get("dir_diffusion_sigma", 0.0)
    if sd:
        B = P_A if P_A is not None else np.eye(d)
        Q = Q + (sd * m_norm) ** 2 * B
    sn = pr.get("norm_diffusion_sigma", 0.0)
    if sn:
        Q = Q + (sn * m_norm) ** 2 * np.eye(d)
    for ev in sched.events:
        if ev.t != k:
            continue
        if ev.type == "mixture_redraw":
            f = float(ev.params["fraction"])
            M = 10  # population size (schedule preset)
            kappa, sig_ln = 20.0, 0.1
            per_rater = np.sqrt(2.0 / kappa + 2.0 * sig_ln ** 2)
            sig_rd = f * per_rater / np.sqrt(max(f * M, 1.0)) * m0_norm
            Q = Q + sig_rd ** 2 * np.eye(d)
        elif ev.type == "novel_translation":
            # off-support by construction: an informed filter inflates uncertainty
            # in the null(battery) directions only, NOT on-support (isotropic jump
            # would falsely kick the visible estimate -- E4b NT-cell finding).
            Pnull = np.eye(d) - (P_A if P_A is not None else np.eye(d))
            n_null = max(float(np.trace(Pnull)), 1.0)
            Q = Q + (ev.params["magnitude"] ** 2 / n_null) * Pnull
        else:
            Q = Q + (sig_jump * m0_norm) ** 2 * np.eye(d)
    return A, Q


def predict_informed(state: GaussState, A: float, Q: np.ndarray) -> GaussState:
    """Predict with a known scalar mean map: m <- A m, S <- A^2 S + Q."""
    return GaussState(A * state.mean, A * A * state.cov + Q)


def informed_predict_a1(state: GaussState, sched, k: int, m0_norm: float, ctx,
                        v_star: np.ndarray, sig_floor: float = 3e-4,
                        sig_jump: float = SIG_JUMP) -> GaussState:
    """Generator-matched predict for A1 schedules (C5 extended to the A1 laws).

    Deterministic laws apply to the posterior as exact linear maps (mean AND
    covariance): norm decay (m <- f m), S7 rotation toward/away w* (the law's
    target is known; realization-free), named-criterion edits (diagonal /
    plane-rotation maps). Stochastic events (direction_jump, family_swap,
    mixture_redraw) contribute jump variance supported on S_ach; SN's null
    translation contributes variance on the null basis (invisible to anchors --
    honestly inflates uncertainty there).
    """
    from . import drift as _drift  # one-way dep: drift never imports estimator

    m = state.mean.copy()
    S = state.cov.copy()
    d = len(m)
    Q = (sig_floor * m0_norm) ** 2 * np.eye(d)
    P_ach = ctx.P_ach if ctx.P_ach is not None else np.eye(d)

    pr = sched.per_round
    decay = pr.get("norm_decay")
    if decay and decay.get("factor", 1.0) != 1.0:
        lo = decay.get("from") or 0
        hi = decay.get("to") or 10 ** 9
        if lo <= k <= hi:
            f = float(decay["factor"])
            m = f * m
            S = f * f * S
    s7 = pr.get("s7")
    if s7 and s7.get("eta_deg", 0.0) > 0.0:
        m = _drift.rotate_toward(m, v_star, s7["eta_deg"], s7["sign"])
        Q = Q + (0.01 * m0_norm) ** 2 * P_ach   # slack: the map acts on m, not v

    for ev in sched.events:
        if ev.t != k:
            continue
        p = ev.params
        if ev.type == "criterion_scale":
            A = np.eye(d); A[p["feature"], p["feature"]] = p["factor"]
            m = A @ m; S = A @ S @ A.T
        elif ev.type == "rubric_edit":
            A = np.eye(d)
            for feat, fac in p["edits"]:
                A[feat, feat] = fac
            m = A @ m; S = A @ S @ A.T
        elif ev.type == "tradeoff_rotation":
            R = _drift.tradeoff_rotation_matrix(ctx, p["feature_a"], p["feature_b"],
                                                p["angle_deg"])
            m = R @ m; S = R @ S @ R.T
        elif ev.type == "null_translation":
            Pn = ctx.null_basis @ ctx.null_basis.T
            Q = Q + (p["magnitude"] ** 2 / ctx.null_basis.shape[1]) * Pn
        elif ev.type == "mixture_redraw":
            f = float(p["fraction"])
            per_rater = np.sqrt(2.0 / 20.0 + 2.0 * 0.1 ** 2)
            sig = f * per_rater / np.sqrt(max(f * 10, 1.0)) * m0_norm
            Q = Q + sig ** 2 * P_ach
        else:  # direction_jump / family_swap: stochastic jumps within S_ach
            Q = Q + (sig_jump * m0_norm) ** 2 * P_ach
    return GaussState(m, 0.5 * (S + S.T) + Q)


# --------------------------------------------------------------------------- #
# Update -- binomial-logit (iterated EKF / Laplace)
# --------------------------------------------------------------------------- #
def update_binomial(state: GaussState, G: np.ndarray, y: np.ndarray, n: int,
                    newton_iters: int = 3, innov_cap_sigma: float = 6.0) -> GaussState:
    """Bayesian logistic update: posterior of v given counts y ~ Binomial(n, sigma(G v)),
    prior N(state.mean, state.cov). Concave -> Newton to the MAP; observed information
    for the covariance. Per-coordinate innovation capped at ``innov_cap_sigma`` prior sd.
    """
    m_pred = state.mean
    P_pred = state.cov
    P_inv = np.linalg.inv(P_pred)
    v = m_pred.copy()
    for _ in range(newton_iters):
        z = G @ v
        p = sigmoid(z)
        grad = P_inv @ (v - m_pred) + G.T @ (n * p - y)
        W = n * (p * (1.0 - p))
        H = P_inv + (G.T * W) @ G
        v = v - np.linalg.solve(H, grad)
    # innovation cap per coordinate
    prior_sd = np.sqrt(np.diag(P_pred))
    delta = np.clip(v - m_pred, -innov_cap_sigma * prior_sd, innov_cap_sigma * prior_sd)
    m_post = m_pred + delta
    z = G @ m_post
    p = sigmoid(z)
    W = n * (p * (1.0 - p))
    H = P_inv + (G.T * W) @ G
    P_post = np.linalg.inv(H)
    P_post = 0.5 * (P_post + P_post.T)
    return GaussState(m_post, P_post)


def init_map_binomial(G: np.ndarray, y: np.ndarray, n: int, d: int,
                      prior_var: float = 10.0, newton_iters: int = 8) -> GaussState:
    """Round-0 MAP init under prior N(0, prior_var I)."""
    prior = GaussState(np.zeros(d), prior_var * np.eye(d))
    return update_binomial(prior, G, y, n, newton_iters=newton_iters, innov_cap_sigma=np.inf)


# --------------------------------------------------------------------------- #
# Update -- linear-Gaussian (exact KF; for test 6a)
# --------------------------------------------------------------------------- #
def update_linear(state: GaussState, H: np.ndarray, y: np.ndarray, R: np.ndarray) -> GaussState:
    """Exact Kalman update for y = H v + eps, eps ~ N(0, R)."""
    P = state.cov
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    m = state.mean + K @ (y - H @ state.mean)
    d = P.shape[0]
    P_post = (np.eye(d) - K @ H) @ P
    return GaussState(m, 0.5 * (P_post + P_post.T))


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #
def onsupport_basis(P_A: np.ndarray, tol: float = 0.5) -> np.ndarray:
    """Orthonormal basis (d x r) of the on-support subspace (eigvecs of P_A ~ 1)."""
    w, V = np.linalg.eigh(P_A)
    return V[:, w > tol]


def coverage_projection(state: GaussState, U: np.ndarray, v_true: np.ndarray,
                        z: float = 1.645) -> dict:
    """Per-direction 90% CI coverage in the columns of U. Returns truth/mean/sd and hits."""
    a = U.T @ v_true
    b = U.T @ state.mean
    s = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", U.T, state.cov, U.T), 0.0))
    hit = np.abs(a - b) <= z * s
    return {"true": a, "mean": b, "sd": s, "hit": hit}


def nees(state: GaussState, v_true: np.ndarray, U: np.ndarray | None = None) -> float:
    """Normalized estimation error squared (optionally restricted to subspace U)."""
    e = state.mean - v_true
    if U is not None:
        e = U.T @ e
        S = U.T @ state.cov @ U
    else:
        S = state.cov
    return float(e @ np.linalg.solve(S, e))


def norm_angle_intervals(state: GaussState, ref_dir: np.ndarray, rng: np.random.Generator,
                         U: np.ndarray | None = None, ndraws: int = 2000,
                         lo: float = 5.0, hi: float = 95.0) -> dict:
    """90% intervals for ||v|| and angle(vhat, ref_dir) by sampling the posterior.

    If ``U`` (on-support basis) is given, norm/angle are computed on the on-support
    projection -- the unmeasured novel dimensions carry prior-driven variance that
    would otherwise dominate the sampled norm (their true value is 0). Since v_true
    and ref_dir live in S_hist ⊆ span(U), this leaves the truth unchanged.
    """
    draws = rng.multivariate_normal(state.mean, state.cov, size=ndraws)
    rd = ref_dir / np.linalg.norm(ref_dir)
    if U is not None:
        draws = draws @ U          # (ndraws, r) on-support coordinates
        rd = U.T @ rd
        rd = rd / np.linalg.norm(rd)
    norms = np.linalg.norm(draws, axis=1)
    cos = np.clip((draws @ rd) / norms, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return {
        "norm_mean": float(norms.mean()),
        "norm_ci": (float(np.percentile(norms, lo)), float(np.percentile(norms, hi))),
        "angle_mean": float(ang.mean()),
        "angle_ci": (float(np.percentile(ang, lo)), float(np.percentile(ang, hi))),
    }


def nees_divergence_threshold(dof: int, q: float = 0.999) -> float:
    return float(chi2.ppf(q, dof))


# --------------------------------------------------------------------------- #
# TV drift statistic
# --------------------------------------------------------------------------- #
def total_variation(means: np.ndarray) -> float:
    """TV over the full vector: sum_k ||m_k - m_{k-1}||."""
    means = np.asarray(means)
    return float(np.sum(np.linalg.norm(np.diff(means, axis=0), axis=1)))


def total_variation_direction(means: np.ndarray) -> float:
    """Direction-only TV: sum_k ||mhat_k - mhat_{k-1}|| (blind to pure norm drift)."""
    means = np.asarray(means)
    dirs = means / np.linalg.norm(means, axis=1, keepdims=True)
    return float(np.sum(np.linalg.norm(np.diff(dirs, axis=0), axis=1)))


# --- Net (endpoint) displacement statistics ------------------------------- #
# Path-length TV accumulates the filter's per-round noise (~1.6 over 60 rounds even
# under S0), which swamps slow drift (S2/S4/S7). Net displacement cancels that noise
# over the run and is the robust drift-magnitude statistic (AMENDMENTS wave 3 C4).
def net_displacement(means: np.ndarray) -> float:
    """||m_K - m_0||: total drift of the estimate over the run."""
    means = np.asarray(means)
    return float(np.linalg.norm(means[-1] - means[0]))


def net_lognorm_change(means: np.ndarray) -> float:
    """|log||m_K|| - log||m_0|||: net norm-channel drift (detects S2)."""
    means = np.asarray(means)
    return float(abs(np.log(np.linalg.norm(means[-1])) - np.log(np.linalg.norm(means[0]))))


def net_angle_change(means: np.ndarray) -> float:
    """angle(m_K, m_0) in degrees: net direction-channel drift (blind to pure norm)."""
    means = np.asarray(means)
    m0, mK = means[0], means[-1]
    c = np.clip((mK @ m0) / (np.linalg.norm(mK) * np.linalg.norm(m0)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))
