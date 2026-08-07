"""E3 estimator-in-the-loop harness (spec v0.2 Section 4, E3).

Runs the C7 drift tracker online against a schedule's anchor probes and logs, per
round, everything the coverage analysis needs (log-only analysis, spec Section 1):
the truth v_true, posterior mean m_k, cov diagonal, on-support-eigenbasis
(truth/mean/sd) triples, NEES (full + on-support), and ||v||/angle intervals with
their truths.

Single-teacher schedules {S0,S1,S2,S4,S5} are covered here (the E3 gate is on these,
spec amendment B.. / v0.2 E3). S3 (population mixture) is reported descriptively and
is handled separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import drift, estimator as est
from ..anchors import build_battery_a0
from ..logging_io import RunLogger
from ..seeding import spawn_generators
from ..teacher import Population, Teacher

SINGLE_TEACHER = ["S0", "S1", "S2", "S4", "S5"]
ALL_SCHEDULES = SINGLE_TEACHER + ["S3"]   # S3 = population mixture (report-only cell)


def _probe_counts(battery, teacher, rng, n_probe):
    """Successes per anchor out of n_probe (vectorized single-teacher draws)."""
    y = np.empty(battery.gaps.shape[0])
    for j in range(battery.gaps.shape[0]):
        p = teacher.prob(battery.gaps[j])
        y[j] = np.sum(rng.random(n_probe) < p)
    return y


def _probe_counts_population(battery, pop, rng, n_probe):
    """Population probing: the rater is resampled PER LABEL (spec C5), so each
    anchor's count mixes raters -- the measurement is E_r[sigma(v_r . g)]."""
    y = np.empty(battery.gaps.shape[0])
    for j in range(battery.gaps.shape[0]):
        g = battery.gaps[j]
        y[j] = sum(pop.label(g, rng) for _ in range(n_probe))
    return y


def run_a0_filter(schedule_name: str, seed: int, mode: str = "known_calendar", K: int = 60,
                  d: int = 24, hist_dim: int = 16, n_probe: int = 50,
                  run_dir: str | Path | None = None, schedule=None) -> dict:
    """Run the drift tracker over one (schedule, seed, mode). Returns per-round arrays.
    ``schedule`` (pre-built Schedule) overrides the preset lookup (E4b magnitude sweep)."""
    ctx = drift.DriftContext.a0(d=d, hist_dim=hist_dim)
    rngs = spawn_generators(seed)
    sched = schedule if schedule is not None else drift.get_preset(schedule_name)
    is_pop = any(ev.type == "mixture_redraw" for ev in sched.events)

    w0 = np.zeros(d)
    w0[ctx.hist_idx] = rngs["teacher"].standard_normal(hist_dim)
    w0 /= np.linalg.norm(w0)
    if is_pop:
        pop = Population([Teacher(w=w0.copy(), s=0.0, beta=1.0) for _ in range(10)])
        v_current = pop.v_mean          # estimand: mixture mean (v0.2 E3, report-only)
        probe = lambda rng: _probe_counts_population(battery, pop, rng, n_probe)
    else:
        teacher = Teacher(w=w0.copy(), s=0.0, beta=1.0)
        v_current = teacher.v
        probe = lambda rng: _probe_counts(battery, teacher, rng, n_probe)

    battery = build_battery_a0(w0.copy(), ctx, rngs["probe"])
    G = np.asarray(battery.gaps)
    U = est.onsupport_basis(battery.P_A)

    y0 = probe(rngs["probe"])
    state = est.init_map_binomial(G, y0, n_probe, d)
    m0_norm = float(np.linalg.norm(state.mean))
    ref_dir = state.mean / m0_norm

    logger = RunLogger(run_dir) if run_dir is not None else None
    event_rounds = {ev.t for ev in sched.events}
    dof_on, dof_full = U.shape[1], d

    means = [state.mean.copy()]
    v0_true = v_current().copy()
    v_trues = []
    out = {k: [] for k in ("round", "nees_on", "nees_full", "cov_hit_on",
                           "norm_true", "norm_lo", "norm_hi", "angle_true", "angle_lo", "angle_hi")}

    for k in range(1, K + 1):
        if is_pop:
            for ev in sched.events:
                if ev.t == k:
                    drift.mixture_redraw(pop, ctx, rngs["drift"], ev.params["fraction"])
        else:
            v_new, _flags = sched.apply_round(k, teacher.v(), ctx, rngs["drift"])
            teacher.w = v_new / teacher.beta
        v_true = v_current()

        if mode == "informed":
            A, Q = est.informed_process(sched, k, m_norm=float(np.linalg.norm(state.mean)),
                                        m0_norm=m0_norm, d=d, P_A=battery.P_A)
            state = est.predict_informed(state, A, Q)
        else:
            Q = est.process_noise(mode, event=(k in event_rounds), m0_norm=m0_norm, d=d)
            state = est.predict(state, Q)
        yk = probe(rngs["probe"])
        state = est.update_binomial(state, G, yk, n_probe)
        means.append(state.mean.copy())

        cov = est.coverage_projection(state, U, v_true)
        na = est.norm_angle_intervals(state, ref_dir, rngs["probe"], U=U)
        nees_on = est.nees(state, v_true, U)
        nees_full = est.nees(state, v_true)

        v_trues.append(v_true.copy())
        out["round"].append(k)
        out["nees_on"].append(nees_on)
        out["nees_full"].append(nees_full)
        out["cov_hit_on"].append(float(np.mean(cov["hit"])))
        norm_true = float(np.linalg.norm(v_true))
        ang_true = float(np.degrees(np.arccos(np.clip(
            (v_true @ ref_dir) / norm_true, -1.0, 1.0))))
        for key, val in (("norm_true", norm_true), ("norm_lo", na["norm_ci"][0]),
                         ("norm_hi", na["norm_ci"][1]), ("angle_true", ang_true),
                         ("angle_lo", na["angle_ci"][0]), ("angle_hi", na["angle_ci"][1])):
            out[key].append(val)

        if logger is not None:
            logger.log_row(
                round=k, seed=seed, schedule=schedule_name, v_true=v_true,
                m_k=state.mean, cov_diag=np.diag(state.cov),
                # coverage projections (on-support eigenbasis) + intervals:
                anchor_p_hat=(yk / n_probe),
                D_crit={"onsupport_true": cov["true"].tolist(),
                        "onsupport_mean": cov["mean"].tolist(),
                        "onsupport_sd": cov["sd"].tolist(),
                        "nees_on": nees_on, "nees_full": nees_full,
                        "norm_true": norm_true, "norm_lo": na["norm_ci"][0],
                        "norm_hi": na["norm_ci"][1], "angle_true": ang_true,
                        "angle_lo": na["angle_ci"][0], "angle_hi": na["angle_ci"][1],
                        "mode": mode})
    if logger is not None:
        logger.close()

    res = {k: np.array(v) for k, v in out.items()}
    res["means"] = np.array(means)
    res["v_true_traj"] = np.array(v_trues)
    res["v0_true"] = v0_true
    res["dof_on"], res["dof_full"] = dof_on, dof_full
    res["tv_full"] = est.total_variation(res["means"])
    res["tv_dir"] = est.total_variation_direction(res["means"])
    return res
