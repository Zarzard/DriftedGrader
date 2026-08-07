"""C8 A1 PbRL round loop (spec v0.2 Section 3, C8).

One run = one (schedule, seed): K=40 rounds of collect segments -> pair -> label
with the drifting teacher -> refit RM -> soft VI policy -> probe anchors -> log
dual-reference harm. Harm outcomes are computed in analysis by pairing each run
with its CRN control at the same seed (S0, or S7null for the S7 arms).

Round-0 geometry (amendment A6): the achievable-gap subspace S_ach is estimated
from a reference pool of uniform-policy segments ("eps-greedy initial policy"
before any RM exists IS uniform -- lead-flagged default), giving P_ach (rank 6),
the gap-covariance eigenvectors (S8 axes), and the null basis (SN axis). All
random drift axes are drawn inside S_ach.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import drift, estimator as est
from ..anchors import build_battery_a1
from ..diagnostics import op_temp_inv
from ..envs.grid import GAMMA, TRUE_W, get_env, uniform_policy
from ..logging_io import RunLogger
from ..policy import (evaluate_policy, feature_expectation, optimal_return,
                      reward_from_w, soft_value_iteration)
from ..probe import d_dir, d_norm, probe_battery
from ..rm import RewardModel
from ..seeding import spawn_generators
from ..teacher import Population, Teacher
from .._math import logit

BETA_0 = 3.0
K_DEFAULT = 40
N_SEG = 200
N_PAIRS = 100
Q_DUP = 0.15
EPS_GREEDY = 0.2
N_PROBE = 50
N_REF_SEGS = 400
N_CAND_PAIRS = 20000
_CLIP = (0.02, 0.98)


def _reference_geometry(env, rng):
    """Round-0 reference pool -> candidate gaps, P_ach, gap eigvecs, null basis."""
    segs = np.array([env.rollout_segment(uniform_policy(), rng) for _ in range(N_REF_SEGS)])
    i = rng.integers(N_REF_SEGS, size=N_CAND_PAIRS)
    j = rng.integers(N_REF_SEGS, size=N_CAND_PAIRS)
    keep = i != j
    gaps = segs[i[keep]] - segs[j[keep]]

    # SVD of the gap matrix: achievable span + null space (threshold 1e-8 sig_max).
    # Economy SVD: for a tall (N x 8) matrix vt is still the complete 8x8 V.
    _, s, vt = np.linalg.svd(gaps, full_matrices=False)
    thresh = 1e-8 * s[0]
    rank = int(np.sum(s > thresh))
    V_ach = vt[:rank]                      # (rank, 8)
    P_ach = V_ach.T @ V_ach
    null_basis = vt[rank:].T               # (8, 8-rank)

    # gap-covariance eigenvectors within S_ach, descending variance (S8 axes)
    C = (gaps.T @ gaps) / len(gaps)
    w_eig, U = np.linalg.eigh(P_ach @ C @ P_ach)
    order = np.argsort(w_eig)[::-1]
    gap_eigvecs = U[:, order[:rank]]       # columns, desc variance
    return gaps, P_ach, gap_eigvecs, null_basis, rank


def _behavior_policy(policy):
    """eps-greedy mixture of the current policy with uniform (eps = 0.2)."""
    return (1.0 - EPS_GREEDY) * policy + EPS_GREEDY * uniform_policy()


def _d_crit(p_hat_t, p_hat_0, battery):
    out = {}
    lo, hi = _CLIP
    for name, idx in battery.contrast_idx.items():
        if len(idx) == 0:
            out[name] = None
            continue
        lt = logit(np.clip(p_hat_t[idx], lo, hi))
        l0 = logit(np.clip(p_hat_0[idx], lo, hi))
        out[name] = float(np.mean(battery.contrast_sign[name] * (lt - l0)))
    return out


def run_a1(schedule_name: str, seed: int, K: int = K_DEFAULT,
           run_dir: str | Path | None = None, tau_pol: float = 0.05,
           n_pairs: int = N_PAIRS, estimator_mode: str | None = "informed",
           schedule=None, arm: str = "accumulate", refresh_f: float = 0.7,
           rm_window: int = 10) -> dict:
    """``tau_pol`` and ``n_pairs`` are the spec's sanctioned A1-harm knobs
    (Risks section); defaults are the pre-registered values. ``estimator_mode``
    runs the C7 drift tracker inline ("informed" per amendment C5, or "blind",
    or None to skip); its m_k trajectory feeds E4's net-stat predictors (C4).
    ``schedule`` (a pre-built Schedule) overrides the preset lookup so E4 can sweep
    magnitudes; ``schedule_name`` is still used for logging + control pairing.
    ``arm`` selects the E5 data-curation arm (accumulate | window | refresh);
    all arms spend the same n_pairs teacher-labels/round (refresh splits them
    refresh_f fresh / (1-refresh_f) re-labeling the oldest retained gaps)."""
    env = get_env()
    rngs = spawn_generators(seed)
    sched = schedule if schedule is not None else drift.get_preset_a1(schedule_name)
    P, F = env.P, env.F

    # --- round-0 geometry + drift context ---------------------------------- #
    cand_gaps, P_ach, gap_eigvecs, null_basis, rank = _reference_geometry(env, rngs["stream"])
    ctx = drift.DriftContext.a1(P_ach, gap_eigvecs, null_basis)

    # --- teacher init ------------------------------------------------------ #
    v_star = BETA_0 * TRUE_W
    s7 = sched.per_round.get("s7")
    is_pop = any(ev.type == "mixture_redraw" for ev in sched.events)
    if s7 is not None:
        # 25deg-off init within S_ach; SAME drift-rng draws across S7a/b/null at a
        # given seed => shared init (the mandatory shared-init twin, spec Risks).
        v0 = drift.direction_jump(v_star, ctx, rngs["drift"],
                                  angle_deg=s7["init_offset_deg"], rho_target=1.0)
    else:
        v0 = v_star.copy()
    if is_pop:
        pop = Population([Teacher(w=v0 / BETA_0, s=0.0, beta=BETA_0) for _ in range(10)])
        v_current = pop.v_mean
        label = pop.label
    else:
        teacher = Teacher(w=v0 / BETA_0, s=0.0, beta=BETA_0)
        v_current = teacher.v
        label = teacher.label

    # --- frozen battery + baseline probe ----------------------------------- #
    battery = build_battery_a1(v0, cand_gaps, rngs["probe"])
    if is_pop:
        p_hat_0 = probe_battery(battery, rngs["probe"], sample_label=pop.label, n_probe=N_PROBE)
    else:
        p_hat_0 = probe_battery(battery, rngs["probe"], prob_fn=teacher.prob, n_probe=N_PROBE)

    # --- drift tracker (C7) ------------------------------------------------- #
    state = None
    means = []
    if estimator_mode is not None:
        y0 = np.rint(p_hat_0 * N_PROBE)
        state = est.init_map_binomial(np.asarray(battery.gaps), y0, N_PROBE, d=8)
        m0_norm = float(np.linalg.norm(state.mean))
        means.append(state.mean.copy())

    # --- gold-reference machinery ------------------------------------------ #
    r_star = reward_from_w(F, TRUE_W)
    _, pi_star = optimal_return(P, F, GAMMA, v_star, tau=tau_pol)   # v-scaled reference
    J_opt_wstar = evaluate_policy(P, r_star, GAMMA, pi_star)

    logger = RunLogger(run_dir) if run_dir is not None else None
    rm = RewardModel(d=8, arm=arm, window=rm_window)
    policy = uniform_policy()
    out = {k: [] for k in ("round", "gold_wstar", "gold_wt", "regret_wt", "proxy_return",
                           "e_shift", "e_epi", "D_dir", "D_norm", "eff_drift_mag",
                           "op_agree")}

    e6 = sched.per_round.get("e6")
    mu_hist: list[np.ndarray] = []
    for k in range(1, K + 1):
        # 1) drift
        if e6 is not None:
            if is_pop:
                raise ValueError("E6 is single-teacher")
            if e6["churn"] and k % e6["churn"] == 0:
                v_new = v0.copy()   # fresh raters from the original pool: resets the
                                    # captured rotation back to the baseline grader v0
                                    # (the "rotation-resets-capture" mechanism)
            elif e6["kind"] == "random":
                v_new = drift.random_rotate_step(teacher.v(), ctx, rngs["drift"], e6["eta_deg"])
            elif e6["kind"] == "coupled" and len(mu_hist) >= 2:
                gbar = mu_hist[-1] - mu_hist[-2]           # policy improvement direction
                v_new = drift.gradient_coupled_step(teacher.v(), ctx, gbar, e6["eta_deg"], sign=+1)
            else:
                v_new = teacher.v()                        # coupled but no gbar yet
            teacher.w = v_new / teacher.beta
        elif s7 is not None:
            if s7["eta_deg"] > 0.0 and k <= s7.get("max_rounds", 10 ** 9):
                if is_pop:
                    raise ValueError("S7 is single-teacher")
                v_new = drift.rotate_toward(teacher.v(), v_star, s7["eta_deg"], s7["sign"],
                                            clamp=s7.get("clamp", True))
                teacher.w = v_new / teacher.beta
        elif is_pop:
            for ev in sched.events:
                if ev.t == k:
                    drift.mixture_redraw(pop, ctx, rngs["drift"], ev.params["fraction"])
        else:
            v_new, _flags = sched.apply_round(k, teacher.v(), ctx, rngs["drift"])
            teacher.w = v_new / teacher.beta
        v_t = v_current()
        w_t = v_t / BETA_0

        # 2) segments: 80% eps-greedy current policy, 20% uniform
        beh = _behavior_policy(policy)
        segs = np.array([env.rollout_segment(beh if rngs["stream"].random() < 0.8
                                             else uniform_policy(), rngs["stream"])
                         for _ in range(N_SEG)])

        # 3) pairs + labels. All arms spend n_pairs teacher-labels/round; the
        # refresh arm splits them into fresh operational pairs + re-labels of the
        # oldest retained gaps under the CURRENT teacher (matched-budget baseline).
        n_fresh = int(round(refresh_f * n_pairs)) if arm == "refresh" else n_pairs
        ii = rngs["stream"].integers(N_SEG, size=n_fresh)
        jj = (ii + 1 + rngs["stream"].integers(N_SEG - 1, size=n_fresh)) % N_SEG
        fresh_gaps = segs[ii] - segs[jj]
        fresh_labels = np.array([label(g, rngs["labels"]) for g in fresh_gaps], dtype=int)
        round_gaps, round_labels = fresh_gaps, fresh_labels
        if arm == "refresh" and n_pairs - n_fresh > 0:
            old = rm.oldest_retained_gaps(n_pairs - n_fresh)
            if old is not None and len(old) > 0:
                relbl = np.array([label(g, rngs["labels"]) for g in old], dtype=int)
                round_gaps = np.vstack([fresh_gaps, old])
                round_labels = np.concatenate([fresh_labels, relbl])
        # double-label agreement diagnostic on the fresh operational pairs
        n_dup = int(round(Q_DUP * n_fresh))
        dup_idx = rngs["stream"].choice(n_fresh, size=n_dup, replace=False)
        dup_labels = np.array([label(fresh_gaps[i], rngs["labels"]) for i in dup_idx], dtype=int)
        agree = float(np.mean(fresh_labels[dup_idx] == dup_labels)) if n_dup else float("nan")

        # 4) RM refit on the arm's retained pairs; 5) policy improvement
        rm.add_round(round_gaps, round_labels)
        w_hat = rm.fit()
        r_hat = reward_from_w(F, w_hat)
        policy = soft_value_iteration(P, r_hat, GAMMA, tau=tau_pol)
        if e6 is not None:  # policy's behavior direction, for next round's coupled drift
            mu_hist.append(feature_expectation(P, F, GAMMA, policy))
        ti_std, ti_raw = op_temp_inv(fresh_gaps, fresh_labels, w_hat)

        # 6) probe anchors
        if is_pop:
            p_hat_t = probe_battery(battery, rngs["probe"], sample_label=pop.label, n_probe=N_PROBE)
        else:
            p_hat_t = probe_battery(battery, rngs["probe"], prob_fn=teacher.prob, n_probe=N_PROBE)
        Dd = d_dir(p_hat_t, p_hat_0, battery.dir_idx)
        Dn = d_norm(p_hat_t, p_hat_0, battery.z0, battery.norm_idx)
        Dc = _d_crit(p_hat_t, p_hat_0, battery)
        if state is not None:
            if estimator_mode == "informed":
                state = est.informed_predict_a1(state, sched, k, m0_norm, ctx, v_star)
            else:
                state = est.predict(state, est.process_noise(
                    estimator_mode, event=False, m0_norm=m0_norm, d=8))
            state = est.update_binomial(state, np.asarray(battery.gaps),
                                        np.rint(p_hat_t * N_PROBE), N_PROBE)
            means.append(state.mean.copy())

        # 7) dual-reference harm + decomposition (exact evaluations)
        gold_wstar = evaluate_policy(P, r_star, GAMMA, policy)
        r_wt = reward_from_w(F, w_t)
        _, pi_vt = optimal_return(P, F, GAMMA, v_t, tau=tau_pol)   # v-scaled current optimum
        gold_wt = evaluate_policy(P, r_wt, GAMMA, policy)
        regret_wt = evaluate_policy(P, r_wt, GAMMA, pi_vt) - gold_wt
        J_wstar_pivt = evaluate_policy(P, r_star, GAMMA, pi_vt)
        e_shift = J_opt_wstar - J_wstar_pivt
        e_epi = J_wstar_pivt - gold_wstar
        proxy_return = evaluate_policy(P, r_hat, GAMMA, policy)
        eff = float(np.linalg.norm(P_ach @ (v_t - v0)))

        for key, val in (("round", k), ("gold_wstar", gold_wstar), ("gold_wt", gold_wt),
                         ("regret_wt", regret_wt), ("proxy_return", proxy_return),
                         ("e_shift", e_shift), ("e_epi", e_epi), ("D_dir", Dd),
                         ("D_norm", Dn), ("eff_drift_mag", eff), ("op_agree", agree)):
            out[key].append(val)

        if logger is not None:
            logger.log_row(round=k, seed=seed, schedule=schedule_name, v_true=v_t,
                           m_k=(state.mean if state is not None else None),
                           cov_diag=(np.diag(state.cov) if state is not None else None),
                           anchor_p_hat=p_hat_t, D_dir=Dd, D_norm=Dn, D_crit=Dc,
                           eff_drift_mag=eff, op_agree=agree, op_flip=1.0 - agree,
                           op_temp_inv_std=ti_std, op_temp_inv_raw=ti_raw,
                           gold_wstar=gold_wstar, gold_wt=gold_wt, regret_wt=regret_wt,
                           proxy_return=proxy_return, e_shift=e_shift, e_epi=e_epi,
                           n_labels_cum=rm.n_pairs)
    if logger is not None:
        logger.close()

    res = {k: np.array(v) for k, v in out.items()}
    res["first_event"] = sched.first_event_round()
    res["rank_ach"] = rank
    res["purity"] = battery.contrast_purity
    if means:
        M = np.array(means)
        res["means"] = M
        res["net_disp"] = est.net_displacement(M)
        res["net_lognorm"] = est.net_lognorm_change(M)
        res["net_angle"] = est.net_angle_change(M)
        res["tv_full"] = est.total_variation(M)
    return res
