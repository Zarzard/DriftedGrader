"""Tandem / active-passive harness for the endogeneity claim (H3).

After Ostrovski, Castro & Dabney, "The Difficulty of Passive Learning in Deep RL"
(NeurIPS 2021). Our closed-loop/open-loop distinction is their active/passive
tandem:

  ACTIVE : its RM drives the policy, which samples comparisons, which the drifting
           grader labels, which the RM refits on -- the full closed loop.
  PASSIVE: an independently-initialised RM trained on the ACTIVE's exact comparison
           stream (same pairs, same labels, same order, same #updates/round), but
           whose own policy NEVER drives sampling or the grader.

Tandem effect (H3): passive regret > active regret on identical data. Their central
result -- and ours -- is that this needs NON-LINEAR function approximation: a convex
LinearBT has a unique, init-independent optimum, so passive RM == active RM given
the same data and the gap is exactly zero. With an MLP the passive lands in a
different basin and over-generalises where the active's data (shaped by the active's
own policy) fails to cover it.

Variants:
  tandem="plain"  : passive independently initialised (their Tandem).
  tandem="forked" : fork at t0 with the ACTIVE's weights; the active policy is then
                    frozen (still generates data), passive keeps training (Forked
                    Tandem -- isolates D x F from initialisation).
"""
from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from .arms import make_arms
from .graders import make_grader
from .reward_models import LinearBT, MLPBT
from .policies import expected_features, sample_pairs
from . import metrics as M


def _make_rm(base, d, P, cfg, seed):
    if base == "linear":
        return LinearBT(d, P=P, reg=cfg.reg)
    return MLPBT(d, P=P, hidden=cfg.hidden, reg=cfg.reg, steps=cfg.steps, seed=seed)


def run_tandem(cfg, seed: int, base: str = "mlp", tandem: str = "plain",
               fork_round: int = 40) -> List[dict]:
    """Run the active closed loop and a passive learner on its data stream.
    Returns per-round rows with R_active, R_passive, w-geometry, VR_t."""
    ss = np.random.SeedSequence(seed)
    streams = ss.spawn(8)
    problem_rng = np.random.default_rng(streams[0])
    agent_rng = np.random.default_rng(streams[1])

    arms = make_arms(cfg.n, cfg.d, problem_rng, corr=cfg.corr)
    Phi, w0 = arms.Phi, arms.w0
    P = None  # MLP primary uses full phi
    grader = make_grader(cfg.drift, arms, cfg.betaT, problem_rng,
                         lam=cfg.lam, T=cfg.T, kappa=cfg.kappa)

    active = _make_rm(base, cfg.d, P, cfg, seed=1000 + seed)
    # plain: independent init; forked: passive starts as a copy of active at fork
    passive = _make_rm(base, cfg.d, P, cfg, seed=9000 + seed)

    all_pairs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    m = np.zeros(cfg.d)
    rows: List[dict] = []
    frozen_pi = None

    for t in range(cfg.T):
        w_t = grader.w.copy()

        # ---- data generation: active acts with its CURRENT policy -------
        pi_gen = M.policy_from_scores(active.predict(Phi), cfg.beta)
        if tandem == "forked" and t >= fork_round:
            if frozen_pi is None:
                frozen_pi = pi_gen.copy()
            pi_gen = frozen_pi

        pairs = sample_pairs(pi_gen, cfg.q, agent_rng)
        labels = grader.label(pairs, agent_rng)
        all_pairs.append(pairs)
        all_labels.append(labels)
        allp = np.concatenate(all_pairs)
        alll = np.concatenate(all_labels).astype(float)

        # ---- both refit on the identical cumulative buffer -------------
        # (active frozen after fork in forked mode)
        if not (tandem == "forked" and t >= fork_round):
            active.fit_buffer(Phi, allp, alll, agent_rng)
        if tandem == "forked" and t == fork_round and base != "linear":
            passive = _clone_mlp(active)
        passive.fit_buffer(Phi, allp, alll, agent_rng)

        # ---- evaluate BOTH policies POST-refit at the same round -------
        pi_active = M.policy_from_scores(active.predict(Phi), cfg.beta)
        pi_passive = M.policy_from_scores(passive.predict(Phi), cfg.beta)
        R_active, _ = M.regret_Rt(pi_active, Phi, w_t, cfg.beta)
        R_passive, _ = M.regret_Rt(pi_passive, Phi, w_t, cfg.beta)

        diffs = Phi[pairs[:, 0]] - Phi[pairs[:, 1]]
        m = (1 - cfg.alpha) * m + cfg.alpha * expected_features(pi_gen, Phi)

        rows.append(dict(
            t=t, seed=int(seed), base=base, tandem=tandem, lam=cfg.lam,
            beta=cfg.beta, R_active=R_active, R_passive=R_passive,
            tandem_gap=R_passive - R_active,
            w_dist=float(np.linalg.norm(w_t - w0)),
        ))

        # grader drifts on the ACTIVE's achieved profile
        grader.step(m)
        u_t = M.drift_direction(w_t, grader.w)
        rows[-1]["VR_t"] = M.variance_ratio(u_t, diffs) if u_t is not None else float("nan")

    return rows


def _clone_mlp(src: MLPBT) -> MLPBT:
    dst = MLPBT(src.d, P=src.P, hidden=src.h, reg=src.reg, steps=src.steps, seed=0)
    dst.W1 = src.W1.copy(); dst.b1 = src.b1.copy()
    dst.w2 = src.w2.copy(); dst.b2 = float(src.b2)
    dst._adam = None
    return dst
