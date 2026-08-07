"""C2. Drift generator (spec v0.2 Section 3, C2).

The generator mutates the identifiable rater state ``v = beta * w`` once per round,
driven by an event calendar plus optional per-round processes. All direction work
is done on ``vhat = v/||v||`` and ``||v||`` separately, then recomposed; ``beta`` is
held fixed and ``w = v/beta`` written back (the beta/w factorization is not
identifiable anyway, Prop 2).

Naming (binding, v0.2): ``vhat`` is the unit direction ONLY. This module never uses
``vhat`` for anything else.

Event functions are pure ``v_old -> v_new`` maps given a ``DriftContext`` and RNG,
so the ratio identities in test 5 (rho mechanics) and test 9 (norm-jump exactness)
can be checked directly. A ``Schedule`` sequences them over rounds.

A0 vs A1: the A0 presets (single-teacher and the S3 population) run fully here. The
A1-only presets (S7*, S8*, SC*, SN) require the achievable-gap projector ``P_ach``
and gap eigenvectors that are built in C8/M6; their event *functions* live here and
are unit-testable, but their end-to-end execution waits for the A1 loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .teacher import Population, Teacher

_ZERO = 1e-12


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class DriftContext:
    """Geometry the drift events need: which coordinates are historical vs novel,
    and (A1 only) the achievable-gap projector, its eigenvectors, and null basis."""

    d: int
    hist_idx: np.ndarray
    nov_idx: np.ndarray
    P_ach: np.ndarray | None = None          # A1: rank-6 achievable-gap projector
    gap_eigvecs: np.ndarray | None = None     # A1: columns = gap-cov eigenvectors (desc)
    null_basis: np.ndarray | None = None      # A1: (d, 2) basis of null(S_ach)

    @classmethod
    def a0(cls, d: int = 24, hist_dim: int = 16) -> "DriftContext":
        return cls(
            d=d,
            hist_idx=np.arange(hist_dim),
            nov_idx=np.arange(hist_dim, d),
        )

    @classmethod
    def a1(cls, P_ach: np.ndarray, gap_eigvecs: np.ndarray,
           null_basis: np.ndarray) -> "DriftContext":
        """A1: random drift axes live in the achievable-gap subspace S_ach
        (amendment A6); no historical/novel coordinate split."""
        d = P_ach.shape[0]
        return cls(d=d, hist_idx=np.arange(d), nov_idx=np.arange(0),
                   P_ach=P_ach, gap_eigvecs=gap_eigvecs, null_basis=null_basis)


def _rand_unit_in_subspace(
    indices: np.ndarray, d: int, rng: np.random.Generator, orthogonal_to: np.ndarray | None = None
) -> np.ndarray:
    """A random unit vector supported on ``indices`` (zero elsewhere), optionally
    orthogonalized against ``orthogonal_to`` first."""
    for _ in range(10):
        v = np.zeros(d)
        v[indices] = rng.standard_normal(len(indices))
        if orthogonal_to is not None:
            u = np.asarray(orthogonal_to, dtype=float)
            v = v - (v @ u) / (u @ u) * u
        n = np.linalg.norm(v)
        if n > 1e-8:
            return v / n
    raise RuntimeError("failed to draw a nondegenerate unit vector in subspace")


def _rand_unit_hist(ctx: DriftContext, rng: np.random.Generator,
                    orthogonal_to: np.ndarray | None = None) -> np.ndarray:
    """Random unit vector in the schedule's 'historical' drift subspace:
    S_hist coordinates in A0, the achievable-gap span S_ach in A1 (amendment A6)."""
    if ctx.P_ach is None:
        return _rand_unit_in_subspace(ctx.hist_idx, ctx.d, rng, orthogonal_to)
    for _ in range(10):
        v = ctx.P_ach @ rng.standard_normal(ctx.d)
        if orthogonal_to is not None:
            u = np.asarray(orthogonal_to, dtype=float)
            v = v - (v @ u) / (u @ u) * u
        n = np.linalg.norm(v)
        if n > 1e-8:
            return v / n
    raise RuntimeError("failed to draw a nondegenerate unit vector in P_ach span")


# --------------------------------------------------------------------------- #
# Event functions (pure v_old -> v_new)
# --------------------------------------------------------------------------- #
def direction_jump(v: np.ndarray, ctx: DriftContext, rng: np.random.Generator,
                   angle_deg: float, rho_target: float) -> np.ndarray:
    """Rotate vhat by ``angle_deg`` about an axis mixing S_hist and S_nov per
    ``rho_target``. Norm preserved (axis is orthogonal to vhat). In A1
    (``ctx.P_ach`` set) the axis lives in S_ach and rho_target must be 1."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    vhat = v / norm
    theta = np.radians(angle_deg)
    rt = float(np.clip(rho_target, 0.0, 1.0))
    if ctx.P_ach is not None:
        if rt != 1.0:
            raise ValueError("A1 direction_jump requires rho_target = 1 (no novel subspace)")
        u = _rand_unit_hist(ctx, rng, orthogonal_to=vhat)
    else:
        u_h = _rand_unit_in_subspace(ctx.hist_idx, ctx.d, rng, orthogonal_to=vhat)
        u_n = _rand_unit_in_subspace(ctx.nov_idx, ctx.d, rng)  # orthogonal to vhat in S_hist
        u = np.sqrt(rt) * u_h + np.sqrt(1.0 - rt) * u_n
        u = u / np.linalg.norm(u)
    vhat_new = np.cos(theta) * vhat + np.sin(theta) * u
    return norm * vhat_new


def direction_jump_about_axis(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate vhat by ``angle_deg`` toward a GIVEN axis (S8: gap-covariance
    eigenvector). The axis is orthogonalized against vhat first; the S8 guard
    (reject near-parallel axes, ||P_perp axis|| > 0.1) is the caller's job."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    vhat = v / norm
    u = np.asarray(axis, dtype=float)
    u = u - (u @ vhat) * vhat
    u = u / np.linalg.norm(u)
    theta = np.radians(angle_deg)
    return norm * (np.cos(theta) * vhat + np.sin(theta) * u)


def rotate_toward(v: np.ndarray, target: np.ndarray, eta_deg: float, sign: int,
                  clamp: bool = True) -> np.ndarray:
    """S7 per-round rotation: rotate vhat by ``eta_deg`` toward (sign=+1) or away
    from (sign=-1) the target direction, in the span(vhat, target) plane. The
    toward case CLAMPS at alignment (lead-approved default): once vhat reaches the
    target it stays there. Norm preserved."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    vhat = v / norm
    t = np.asarray(target, dtype=float)
    t = t / np.linalg.norm(t)
    cosang = float(np.clip(vhat @ t, -1.0, 1.0))
    ang = np.degrees(np.arccos(cosang))
    if ang < 1e-9:  # aligned: 'toward' stays; 'away' has no defined plane -- stay
        return v
    u = t - cosang * vhat
    u = u / np.linalg.norm(u)          # in-plane direction toward target
    step = eta_deg if sign > 0 else -eta_deg
    if clamp and sign > 0:
        step = min(step, ang)          # do not overshoot the target
    th = np.radians(step)
    return norm * (np.cos(th) * vhat + np.sin(th) * u)


def gradient_coupled_step(v: np.ndarray, ctx: DriftContext, gbar: np.ndarray,
                          eta_deg: float, sign: int) -> np.ndarray:
    """E6: rotate vhat by ``eta_deg`` toward (sign=+1) / away (sign=-1) from the
    policy's improvement direction ``gbar``, projected into the achievable set and
    orthogonal to vhat. Norm held constant. Returns v unchanged if the projected
    axis is degenerate (``|E[phi]_k - E[phi]_{k-1}|`` was ~0 or aligned with vhat)."""
    v = np.asarray(v, dtype=float)
    P = ctx.P_ach if ctx.P_ach is not None else np.eye(ctx.d)
    axis = P @ np.asarray(gbar, dtype=float)
    vhat = v / np.linalg.norm(v)
    axis = axis - (axis @ vhat) * vhat
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return v
    return direction_jump_about_axis(v, sign * axis / n, eta_deg)


def random_rotate_step(v: np.ndarray, ctx: DriftContext, rng: np.random.Generator,
                       eta_deg: float) -> np.ndarray:
    """E6 magnitude-matched control: rotate vhat by ``eta_deg`` about a random axis
    in the achievable set, orthogonal to vhat. Norm held constant."""
    axis = _rand_unit_hist(ctx, rng, orthogonal_to=v / np.linalg.norm(v))
    return direction_jump_about_axis(v, axis, eta_deg)


def churn_reset(v: np.ndarray, ctx: DriftContext, rng: np.random.Generator) -> np.ndarray:
    """E6 churn: replace vhat with a fresh random direction in the achievable set,
    keeping ||v|| (models a rater-pool turnover that resets the captured rotation)."""
    norm = float(np.linalg.norm(v))
    return norm * _rand_unit_hist(ctx, rng)


def null_translation(v: np.ndarray, ctx: DriftContext, rng: np.random.Generator,
                     magnitude: float) -> np.ndarray:
    """SN: translate v along null(S_ach) -- drift that is invisible to every
    achievable segment gap AND harmless (constant-per-step reward shift)."""
    if ctx.null_basis is None:
        raise ValueError("null_translation requires ctx.null_basis (A1 only)")
    coef = rng.standard_normal(ctx.null_basis.shape[1])
    u = ctx.null_basis @ coef
    u = u / np.linalg.norm(u)
    return np.asarray(v, dtype=float) + magnitude * u


def novel_translation(v: np.ndarray, ctx: DriftContext, rng: np.random.Generator,
                      magnitude: float) -> np.ndarray:
    """Translate v by ``magnitude`` along a random novel axis. rho_step = 0 exactly."""
    v = np.asarray(v, dtype=float)
    u_nov = _rand_unit_in_subspace(ctx.nov_idx, ctx.d, rng)
    return v + magnitude * u_nov


def norm_scale(v: np.ndarray, factor: float) -> np.ndarray:
    """Scale ||v|| by ``factor`` (direction unchanged). Covers norm_jump / norm_decay."""
    return float(factor) * np.asarray(v, dtype=float)


def family_swap(v: np.ndarray, ctx: DriftContext, rng: np.random.Generator,
                cos_target: float, norm_factor: float) -> np.ndarray:
    """Resample direction at cosine ``cos_target`` to the old one (perturbation in
    S_hist), then scale norm by ``norm_factor``. Judge/version swap."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    vhat = v / norm
    w_perp = _rand_unit_hist(ctx, rng, orthogonal_to=vhat)
    c = float(np.clip(cos_target, -1.0, 1.0))
    vhat_new = c * vhat + np.sqrt(1.0 - c * c) * w_perp
    return (norm * norm_factor) * vhat_new


def dir_diffusion(v: np.ndarray, ctx: DriftContext, rng: np.random.Generator,
                  sigma: float) -> np.ndarray:
    """Per-round direction diffusion in S_hist: vhat <- normalize(vhat + sigma eps_perp)."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    vhat = v / norm
    if ctx.P_ach is not None:
        eps = ctx.P_ach @ rng.standard_normal(ctx.d)
    else:
        eps = np.zeros(ctx.d)
        eps[ctx.hist_idx] = rng.standard_normal(len(ctx.hist_idx))
    eps_perp = eps - (eps @ vhat) * vhat
    vhat_new = vhat + sigma * eps_perp
    vhat_new = vhat_new / np.linalg.norm(vhat_new)
    return norm * vhat_new


def norm_diffusion(v: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Per-round lognormal norm diffusion: ||v|| <- ||v|| exp(sigma xi - sigma^2/2)."""
    v = np.asarray(v, dtype=float)
    xi = rng.standard_normal()
    return v * float(np.exp(sigma * xi - 0.5 * sigma * sigma))


# --- A1 named-criterion events (operate on w in feature coords; need P_ach) ---
def criterion_scale(v: np.ndarray, feature: int, factor: float) -> np.ndarray:
    """Scale one feature's weight. Does not renormalize; incidental norm change logged."""
    v = np.array(v, dtype=float)
    v[feature] *= float(factor)
    return v


def rubric_edit(v: np.ndarray, edits: list[tuple[int, float]]) -> np.ndarray:
    """Apply several ``criterion_scale`` edits at once."""
    v = np.array(v, dtype=float)
    for feature, factor in edits:
        v[feature] *= float(factor)
    return v


def tradeoff_rotation_matrix(ctx: DriftContext, feature_a: int, feature_b: int,
                             angle_deg: float) -> np.ndarray:
    """The (d x d) linear map of ``tradeoff_rotation`` -- exposed so the informed
    estimator can apply the same known law to its posterior (mean AND covariance)."""
    if ctx.P_ach is None:
        raise ValueError("tradeoff_rotation requires ctx.P_ach (A1 only)")
    d = ctx.d
    e_a = np.zeros(d); e_a[feature_a] = 1.0
    e_b = np.zeros(d); e_b[feature_b] = 1.0
    a = ctx.P_ach @ e_a
    a = a / np.linalg.norm(a)
    b = ctx.P_ach @ e_b
    b = b - (b @ a) * a           # Gram-Schmidt
    b = b / np.linalg.norm(b)
    th = np.radians(angle_deg)
    B = np.column_stack([a, b])   # (d, 2)
    R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.eye(d) + B @ (R2 - np.eye(2)) @ B.T


def tradeoff_rotation(v: np.ndarray, ctx: DriftContext, feature_a: int, feature_b: int,
                      angle_deg: float) -> np.ndarray:
    """Rotate v's components in the plane spanned by the achievable projections of
    e_a, e_b (Gram-Schmidt'd), preserving in-plane norm. Requires ``P_ach``."""
    R = tradeoff_rotation_matrix(ctx, feature_a, feature_b, angle_deg)
    return R @ np.asarray(v, dtype=float)


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    t: int
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Schedule:
    """A named drift schedule: an event calendar + per-round processes.

    ``apply_round`` advances the rater state by one round and returns
    ``(new_v, event_flags)``. Realized rho / effective drift magnitude are computed
    by the caller (they need the achievable-rowspace projector), from the returned
    increment; ``apply_round`` reports which events fired.
    """

    name: str
    rung: str = "A0"
    events: list[Event] = field(default_factory=list)
    per_round: dict[str, Any] = field(default_factory=dict)

    def apply_round(self, k: int, v: np.ndarray, ctx: DriftContext,
                    rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
        v = np.asarray(v, dtype=float).copy()
        flags: list[str] = []

        # 1) calendar events at this round
        for ev in self.events:
            if ev.t != k:
                continue
            v = _dispatch_event(ev, v, ctx, rng)
            flags.append(ev.type)

        # 2) per-round processes
        pr = self.per_round
        decay = pr.get("norm_decay")
        if decay and decay.get("factor", 1.0) != 1.0:
            lo = decay.get("from", 0)
            hi = decay.get("to", 10**9)
            if (lo is None or k >= lo) and (hi is None or k <= hi):
                v = norm_scale(v, decay["factor"])
                flags.append("norm_decay")
        if pr.get("dir_diffusion_sigma", 0.0):
            v = dir_diffusion(v, ctx, rng, pr["dir_diffusion_sigma"])
            flags.append("dir_diffusion")
        if pr.get("norm_diffusion_sigma", 0.0):
            v = norm_diffusion(v, rng, pr["norm_diffusion_sigma"])
            flags.append("norm_diffusion")

        return v, flags

    def first_event_round(self) -> int:
        """First round at which the schedule induces any drift (the G_auc/R_auc
        window start). Discrete events use their round; per-round processes use the
        round they switch on. Stationary schedules return a large sentinel."""
        candidates: list[int] = [ev.t for ev in self.events]
        pr = self.per_round
        decay = pr.get("norm_decay")
        if decay and decay.get("factor", 1.0) != 1.0:
            candidates.append(int(decay.get("from", 0) or 0))
        if pr.get("dir_diffusion_sigma", 0.0) or pr.get("norm_diffusion_sigma", 0.0):
            candidates.append(1)
        s7 = pr.get("s7")
        if s7 and s7.get("eta_deg", 0.0) > 0.0:
            candidates.append(1)   # S7 drifts continuously from round 1
        return min(candidates) if candidates else 10**9


def _dispatch_event(ev: Event, v: np.ndarray, ctx: DriftContext,
                    rng: np.random.Generator) -> np.ndarray:
    p = ev.params
    if ev.type == "direction_jump":
        if "eig_rank" in p:  # S8: jump about a gap-covariance eigenvector
            if ctx.gap_eigvecs is None:
                raise ValueError("eig_rank jump requires ctx.gap_eigvecs (A1 only)")
            vhat = v / np.linalg.norm(v)
            rank = int(p["eig_rank"])
            axis = ctx.gap_eigvecs[:, rank]
            # S8 axis guard (lead-approved default): fall to next eigvec if near-parallel
            while np.linalg.norm(axis - (axis @ vhat) * vhat) <= 0.1:
                rank += 1
                axis = ctx.gap_eigvecs[:, rank]
            return direction_jump_about_axis(v, axis, p["angle_deg"])
        return direction_jump(v, ctx, rng, p["angle_deg"], p["rho_target"])
    if ev.type == "null_translation":
        return null_translation(v, ctx, rng, p["magnitude"])
    if ev.type == "novel_translation":
        return novel_translation(v, ctx, rng, p["magnitude"])
    if ev.type == "norm_jump":
        return norm_scale(v, p["factor"])
    if ev.type == "family_swap":
        return family_swap(v, ctx, rng, p["cos_target"], p["norm_factor"])
    if ev.type == "criterion_scale":
        return criterion_scale(v, p["feature"], p["factor"])
    if ev.type == "rubric_edit":
        return rubric_edit(v, p["edits"])
    if ev.type == "tradeoff_rotation":
        return tradeoff_rotation(v, ctx, p["feature_a"], p["feature_b"], p["angle_deg"])
    if ev.type == "mixture_redraw":
        raise ValueError("mixture_redraw is a population event; use apply_round_population")
    raise ValueError(f"unknown event type {ev.type!r}")


# --------------------------------------------------------------------------- #
# Population drift (S3 mixture_redraw)
# --------------------------------------------------------------------------- #
def mixture_redraw(pop: Population, ctx: DriftContext, rng: np.random.Generator,
                   fraction: float, kappa: float = 20.0, lognorm_sigma: float = 0.1) -> None:
    """Replace a ``fraction`` of raters with fresh draws: direction vMF(pop_mean, kappa)
    in S_hist, ||v|| ~ LogNormal(0, lognorm_sigma). Mutates ``pop`` in place."""
    m = len(pop.teachers)
    n_replace = int(round(fraction * m))
    if n_replace == 0:
        return
    mean_v = pop.v_mean()
    mean_dir = mean_v / np.linalg.norm(mean_v)
    # New raters' norms are RELATIVE to the current population scale (an absolute
    # LogNormal(0, s) only coincides with the A0 scale ||v||=1; at A1's ||v||~4.4
    # it would collapse the population norm).
    base_norm = float(np.mean([np.linalg.norm(t.v()) for t in pop.teachers]))
    idx = rng.choice(m, size=n_replace, replace=False)
    for i in idx:
        vhat = _sample_vmf(mean_dir, kappa, ctx, rng)
        norm = base_norm * float(np.exp(rng.normal(0.0, lognorm_sigma)))
        pop.teachers[i].w = (norm / pop.teachers[i].beta) * vhat


def _sample_vmf(mu: np.ndarray, kappa: float, ctx: DriftContext,
                rng: np.random.Generator) -> np.ndarray:
    """von Mises-Fisher sample around ``mu`` restricted to S_hist (approximate;
    tangent-Gaussian for the perturbation, exact enough at kappa=20)."""
    tangent = _rand_unit_hist(ctx, rng, orthogonal_to=mu)
    # concentration ~ 1/sqrt(kappa) angular spread
    angle = rng.normal(0.0, 1.0 / np.sqrt(kappa))
    return np.cos(angle) * mu + np.sin(angle) * tangent


# --------------------------------------------------------------------------- #
# Presets (spec v0.2 C2 table)
# --------------------------------------------------------------------------- #
def _nt_magnitude(angle_deg: float) -> float:
    """||delta|| of an ``angle_deg`` rotation at ||v||=1 is 2 sin(theta/2)."""
    return 2.0 * np.sin(np.radians(angle_deg) / 2.0)


def get_preset(name: str) -> Schedule:
    """Return the Schedule for a preset id. A0 event round = 30, A1 = 20 (spec table)."""
    P: dict[str, Callable[[], Schedule]] = {
        "S0": lambda: Schedule("S0", "A0"),
        "S1": lambda: Schedule("S1", "A0",
            [Event(30, "direction_jump", {"angle_deg": 25, "rho_target": 1.0})]),
        "S1m": lambda: Schedule("S1m", "A0",
            [Event(30, "direction_jump", {"angle_deg": 25, "rho_target": 0.5})]),
        "S1n": lambda: Schedule("S1n", "A0",
            [Event(30, "direction_jump", {"angle_deg": 25, "rho_target": 0.0})]),
        "NT": lambda: Schedule("NT", "A0",
            [Event(30, "novel_translation", {"magnitude": _nt_magnitude(25)})]),
        "S2": lambda: Schedule("S2", "A0", [],
            {"norm_decay": {"from": 20, "to": None, "factor": 0.985}}),
        "S2j": lambda: Schedule("S2j", "A0",
            [Event(30, "norm_jump", {"factor": 0.6})]),
        "S1p": lambda: Schedule("S1p", "A0",
            [Event(30, "direction_jump", {"angle_deg": 25, "rho_target": 1.0})]),
        "S4": lambda: Schedule("S4", "A0", [], {"dir_diffusion_sigma": 0.02}),
        "S5": lambda: Schedule("S5", "A0",
            [Event(30, "family_swap", {"cos_target": 0.7, "norm_factor": 1.3})]),
        # Population preset
        "S3": lambda: Schedule("S3", "A0",
            [Event(t, "mixture_redraw", {"fraction": 0.3}) for t in (10, 20, 30, 40, 50)]),
        # --- A1-only presets (executed in M6; event functions tested in M1) ---
        "S8t": lambda: Schedule("S8t", "A1",
            [Event(20, "direction_jump", {"angle_deg": 25, "rho_target": 1.0, "eig_rank": 0})]),
        "S8m": lambda: Schedule("S8m", "A1",
            [Event(20, "direction_jump", {"angle_deg": 25, "rho_target": 1.0, "eig_rank": 2})]),
        "S8b": lambda: Schedule("S8b", "A1",
            [Event(20, "direction_jump", {"angle_deg": 25, "rho_target": 1.0, "eig_rank": 5})]),
        "SC1": lambda: Schedule("SC1", "A1",
            [Event(20, "criterion_scale", {"feature": 2, "factor": 1.5})]),      # lava
        "SC2": lambda: Schedule("SC2", "A1",
            [Event(20, "tradeoff_rotation", {"feature_a": 1, "feature_b": 3, "angle_deg": 20})]),
        "SC3": lambda: Schedule("SC3", "A1",
            [Event(20, "rubric_edit", {"edits": [(2, 1.3), (3, 1.5), (5, 2.0)]})]),
    }
    if name not in P:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(P)}")
    return P[name]()


A0_PRESETS = ["S0", "S1", "S1m", "S1n", "NT", "S2", "S2j", "S1p", "S4", "S5", "S3"]
A1_PRESETS = ["S8t", "S8m", "S8b", "SC1", "SC2", "SC3"]  # + S7a/S7b/SN built in C8/M6


def _nt_magnitude_a1(angle_deg: float, v_norm: float) -> float:
    """||delta|| of an ``angle_deg`` rotation at ||v|| = v_norm."""
    return 2.0 * v_norm * np.sin(np.radians(angle_deg) / 2.0)


def get_preset_a1(name: str) -> Schedule:
    """A1 schedule presets (spec v0.2 C2 table: event time t=20, K=40).

    S7a/b carry a ``s7`` per_round entry (interpreted by the A1 loop: 25deg-off
    init + 1deg/round rotation toward/away w*, toward clamps at w*); S7null is the
    mandatory shared-init stationary twin. SN's magnitude matches a 25deg jump at
    ||v0|| = 3||w*|| ~ 4.44.
    """
    P: dict[str, Callable[[], Schedule]] = {
        "S0": lambda: Schedule("S0", "A1"),
        "S1": lambda: Schedule("S1", "A1",
            [Event(20, "direction_jump", {"angle_deg": 25, "rho_target": 1.0})]),
        "S2": lambda: Schedule("S2", "A1", [],
            {"norm_decay": {"from": 20, "to": None, "factor": 0.985}}),
        "S3": lambda: Schedule("S3", "A1",
            [Event(t, "mixture_redraw", {"fraction": 0.3}) for t in (10, 20, 30)]),
        "S5": lambda: Schedule("S5", "A1",
            [Event(20, "family_swap", {"cos_target": 0.7, "norm_factor": 1.3})]),
        # S7a converges toward w* and STOPS there (rotate_toward halts at alignment);
        # S7b diverges. Comparable-magnitude, opposite direction-relative-to-w*,
        # opposite harm -- the H0-killer rests on S7a's substantial drift yet negative
        # harm (a magnitude predictor cannot produce that sign), not on exact
        # magnitude equality (lead decision 2026-08-05: keep converge-and-stop).
        "S7a": lambda: Schedule("S7a", "A1", [],
            {"s7": {"eta_deg": 1.0, "sign": +1, "init_offset_deg": 25.0}}),
        "S7b": lambda: Schedule("S7b", "A1", [],
            {"s7": {"eta_deg": 1.0, "sign": -1, "init_offset_deg": 25.0}}),
        # S7bm: matched-displacement 'away' twin. S7a converges toward w* and stops
        # at ~25deg of net rotation; S7bm rotates AWAY the same 25deg (max_rounds=25)
        # then stops. Equal net displacement, opposite direction -> the clean
        # equal-magnitude/opposite-harm contrast (closes the E4 open item).
        "S7bm": lambda: Schedule("S7bm", "A1", [],
            {"s7": {"eta_deg": 1.0, "sign": -1, "init_offset_deg": 25.0, "max_rounds": 25}}),
        "S7null": lambda: Schedule("S7null", "A1", [],
            {"s7": {"eta_deg": 0.0, "sign": 0, "init_offset_deg": 25.0}}),
        "SN": lambda: Schedule("SN", "A1",
            [Event(20, "null_translation", {"magnitude": _nt_magnitude_a1(25.0, 4.44)})]),
        # E6 gradient-coupled 2x2: {coupled, random} x {persistent, churn-8}, 2deg/round,
        # norm held constant. The per_round `e6` dict drives the special drift path in
        # a1_run; `dir_diffusion_sigma` tells the informed tracker to expect ~2deg/round
        # of direction diffusion so coverage/detectability are measured fairly.
        "E6cp": lambda: Schedule("E6cp", "A1", [],
            {"e6": {"kind": "coupled", "eta_deg": 2.0, "churn": 0}, "dir_diffusion_sigma": 0.035}),
        "E6cc": lambda: Schedule("E6cc", "A1", [],
            {"e6": {"kind": "coupled", "eta_deg": 2.0, "churn": 8}, "dir_diffusion_sigma": 0.035}),
        "E6rp": lambda: Schedule("E6rp", "A1", [],
            {"e6": {"kind": "random", "eta_deg": 2.0, "churn": 0}, "dir_diffusion_sigma": 0.035}),
        "E6rc": lambda: Schedule("E6rc", "A1", [],
            {"e6": {"kind": "random", "eta_deg": 2.0, "churn": 8}, "dir_diffusion_sigma": 0.035}),
    }
    # rate-replication variants (review round 5): same 2x2 at eta = 1deg and 4deg,
    # tracker diffusion scaled ~proportionally (2deg -> 0.035).
    for eta, tag in ((1.0, "1"), (4.0, "4")):
        sig = 0.035 * eta / 2.0
        for kind, kc in (("coupled", "c"), ("random", "r")):
            for churn, pc in ((0, "p"), (8, "c")):
                nm = f"E6{kc}{pc}{tag}"
                P[nm] = (lambda n=nm, k=kind, ch=churn, e=eta, sg=sig:
                         Schedule(n, "A1", [], {"e6": {"kind": k, "eta_deg": e, "churn": ch},
                                                "dir_diffusion_sigma": sg}))
    if name in P:
        return P[name]()
    if name in A1_PRESETS:  # S8t/m/b, SC1-3 already defined at t=20 in get_preset
        return get_preset(name)
    raise KeyError(f"unknown A1 preset {name!r}")


A1_ALL = ["S0", "S1", "S2", "S3", "S5", "S7a", "S7b", "S7null",
          "S8t", "S8m", "S8b", "SC1", "SC2", "SC3", "SN"]

# Control (CRN twin) pairing for A1 harm outcomes: S7 arms compare against the
# shared-init stationary twin, NOT S0 (spec v0.2 Risks -- "comparing S7 runs
# against S0 is a bug"); everything else pairs with S0.
A1_CONTROL = {name: ("S7null" if name.startswith("S7") and name != "S7null" else "S0")
              for name in A1_ALL}
