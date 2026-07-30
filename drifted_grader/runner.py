"""Single-run closed loop: config -> per-round rows -> parquet.

Loop order (plan Section 7.2), load-bearing:
    pi -> sample/label/fit -> update EMA m -> LOG ROW -> grader.step(m)
so every row's diagnostics and its w_t refer to the same round. u_t (drift
direction) needs w_{t+1}, so A_t / proj_angle are computed just after step and
backfilled into the same row.

Primary metric is R_t (regret vs the current grader at matched pressure,
Section 3.6), computed via metrics.py with range-normalised scores. J_init/
J_final/J_avg are logged DESCRIPTIVE-ONLY.

Two RNG streams from `seed`: problem (Phi/w0/P/instruments, shared across
conditions at a seed) and agent (policy noise, sampling, RM bootstrap;
selectable via agent_stream so DRIFT-REPLAY feeds a decoupled fresh agent).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import warnings
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from .arms import make_arms
from .graders import make_grader
from .policies import make_policy, expected_features, sample_pairs, gibbs_policy
from .reward_models import make_reward_model, make_projection, EnsembleBT
from . import metrics as M


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", _REPO_DIR, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "uncommitted"
    except Exception:
        return "nogit"


GIT_SHA = _git_sha()


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class Config:
    stage: str = "A"
    drift: str = "SAT"           # NONE|SAT|SCALE|EXO|REPLAY
    lam: float = 0.2
    beta: float = 0.1            # KL-Gibbs pressure (smaller => more pressure)
    betaT: float = 10.0          # grader inverse temperature (positive!)
    d: int = 6
    k: int = 6                   # RM projection dim; k>=d => full phi (MLP primary).
                                 # k<d = structural-misspec ablation (linear only)
    n: int = 256
    T: int = 100
    alpha: float = 0.5
    L: int = 0                   # agent estimate lag (Stage A)
    sigma: float = 0.0           # agent estimate noise (Stage A)
    q: int = 64                  # comparisons per round
    policy: str = "gibbs"        # gibbs|bestofn
    n_bo: int = 16
    rm: str = "none"             # none|linear|ensemble
    reg: float = 1e-3
    K: int = 5                   # ensemble heads
    hidden: int = 64             # MLP hidden units
    steps: int = 80              # MLP online-refit Adam steps
    kappa: float = 4.0           # DriftScale sharpness
    corr: float = 0.0            # feature-correlation knob
    buffer_regime: str = "all"   # all|recency
    window: int = 10
    agent_stream: int = 0        # REPLAY uses 1 to decouple the fresh agent
    n_probe: int = 64            # fixed probe set (D4)
    q_anchor: int = 64           # anchor comparisons (D5)
    q_eval: int = 256            # held-out eval pairs (D1)
    schema: str = "reduced"      # reduced|full  (logging only, not hashed)

    def hash(self) -> str:
        d = {kk: vv for kk, vv in asdict(self).items() if kk != "schema"}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def config_hash(cfg: Config) -> str:
    return cfg.hash()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-np.sum(q * np.log(q)))


def _simplex_entropy(w: np.ndarray) -> float:
    q = w[w > 1e-12]
    return float(-np.sum(q * np.log(q)))


_SAFE = dict(over="raise", divide="raise", invalid="raise", under="ignore")


def _psi_snapshot(rm, use_rm: bool) -> list:
    """Head-0 psi (ensemble) or psi (linear); [] when no RM."""
    if not use_rm:
        return []
    if isinstance(rm, EnsembleBT):
        return rm.heads[0].psi.tolist()
    return rm.psi.tolist()


# --------------------------------------------------------------------------
# main run
# --------------------------------------------------------------------------
def run(cfg: Config, seed: int, record_w_traj: bool = False,
        replay_w_traj=None) -> List[dict]:
    ss = np.random.SeedSequence(seed)
    streams = ss.spawn(16)
    problem_rng = np.random.default_rng(streams[0])
    agent_rng = np.random.default_rng(streams[1 + cfg.agent_stream])

    arms = make_arms(cfg.n, cfg.d, problem_rng, corr=cfg.corr)
    Phi, w0 = arms.Phi, arms.w0
    P = make_projection(cfg.d, cfg.k, problem_rng)          # per-seed, fixed
    A_null = M.A_null(cfg.d)

    grader = make_grader(cfg.drift, arms, cfg.betaT, problem_rng,
                         lam=cfg.lam, T=cfg.T, kappa=cfg.kappa, w_traj=replay_w_traj)

    use_rm = cfg.rm != "none"
    rm = (make_reward_model(cfg.rm, cfg.d, P=P, K=cfg.K, reg=cfg.reg,
                            hidden=cfg.hidden, steps=cfg.steps)
          if use_rm else None)

    probe_idx = problem_rng.choice(cfg.n, size=min(cfg.n_probe, cfg.n), replace=False)
    anchor_pairs = np.stack([problem_rng.choice(cfg.n, size=cfg.q_anchor),
                             problem_rng.choice(cfg.n, size=cfg.q_anchor)], axis=1)
    anchor_p0 = grader.prob_first(anchor_pairs).copy()

    buffer_pairs: List[np.ndarray] = []
    buffer_labels: List[np.ndarray] = []
    m = np.zeros(cfg.d)
    w_hist: List[np.ndarray] = []
    ef_hist: List[np.ndarray] = []
    rows: List[dict] = []
    prev_probe_z: Optional[np.ndarray] = None
    R_running_sum = 0.0

    with np.errstate(**_SAFE):
        for t in range(cfg.T):
            w_t = grader.w.copy()
            w_hist.append(w_t)

            # ---- agent scoring vector over arms --------------------------
            if use_rm:
                r_vec = rm.predict(Phi)
            else:
                idx = max(0, t - cfg.L)
                what = w_hist[idx] + cfg.sigma * agent_rng.standard_normal(cfg.d)
                r_vec = Phi @ what

            # ---- policy (range-normalised scores; beta = pure pressure) --
            if cfg.policy == "gibbs":
                pi = M.policy_from_scores(r_vec, cfg.beta)
            else:  # bestofn
                pi = make_policy("bestofn", r_vec, n_bo=cfg.n_bo)

            # ---- comparisons this round (for A_t design matrix + RM/D1) --
            pairs = sample_pairs(pi, cfg.q, agent_rng)
            diffs_round = Phi[pairs[:, 0]] - Phi[pairs[:, 1]]
            v_min, _ = M.least_informed_direction(diffs_round)

            d1_acc = float("nan")
            if use_rm:
                labels = grader.label(pairs, agent_rng)
                buffer_pairs.append(pairs)
                buffer_labels.append(labels)
                if cfg.buffer_regime == "recency":
                    vp, vl = buffer_pairs[-cfg.window:], buffer_labels[-cfg.window:]
                else:
                    vp, vl = buffer_pairs, buffer_labels
                allp = np.concatenate(vp, axis=0)
                alll = np.concatenate(vl, axis=0)
                rm.fit_buffer(Phi, allp, alll.astype(float), agent_rng)
                # D1: held-out accuracy on fresh current-grader on-policy pairs
                ev = sample_pairs(pi, cfg.q_eval, agent_rng)
                ev_lab = grader.label(ev, agent_rng)
                r_ev = rm.predict(Phi)
                pred = (r_ev[ev[:, 0]] > r_ev[ev[:, 1]]).astype(int)
                d1_acc = float(np.mean(pred == ev_lab))

            # ---- achieved profile EMA ------------------------------------
            ef = expected_features(pi, Phi)
            ef_hist.append(ef)
            m = (1 - cfg.alpha) * m + cfg.alpha * ef

            # ---- primary metrics -----------------------------------------
            R_t, range_t = M.regret_Rt(pi, Phi, w_t, cfg.beta)
            R_running_sum += R_t
            J_init = float(w0 @ ef)
            J_proxy = float(r_vec @ pi)
            kl_ref = float(np.log(cfg.n) - _entropy(pi))

            # descriptive detector signals available now
            d2 = float("nan")
            if isinstance(rm, EnsembleBT):
                d2 = float(pi @ rm.disagreement(Phi))
            d4 = float("nan")
            if use_rm:
                pz = M.rangenorm(rm.predict(Phi[probe_idx]))
                if prev_probe_z is not None:
                    d4 = float(np.sqrt(np.mean((pz - prev_probe_z) ** 2)))
                prev_probe_z = pz
            anchor_pt = grader.prob_first(anchor_pairs)
            d5 = float(np.mean(np.abs(anchor_pt - anchor_p0)))

            row = dict(
                git_sha=GIT_SHA, config_hash=cfg.hash(), seed=int(seed),
                stage=cfg.stage, schema=cfg.schema, drift_model=cfg.drift,
                lam=cfg.lam, beta=cfg.beta, betaT=cfg.betaT, d=cfg.d, k=cfg.k,
                n=cfg.n, L=cfg.L, sigma=cfg.sigma, buffer_regime=cfg.buffer_regime,
                t=t,
                R_t=R_t, R_bar_running=float(R_running_sum / (t + 1)),
                range_t=range_t, A_t=float("nan"), A_null=A_null,
                VR_t=float("nan"),
                J_init=J_init, J_final=float("nan"), J_avg=float("nan"),
                J_proxy=J_proxy,
                w_dist_from_w0=float(np.linalg.norm(w_t - w0)),
                w_step=float("nan"), w_entropy=_simplex_entropy(w_t),
                cond_Phi=arms.cond_Phi, proj_angle=float("nan"),
                D1_heldout_acc=d1_acc, D2_ens_disagree=d2, D3_kl_ref=kl_ref,
                D4_temporal_rm_drift=d4, D5_anchor_stat=d5,
                D6_kappa=float("nan"), D6_rho=float("nan"), D6_R=float("nan"),
                D7_regret_oracle=R_t,
                n_comparisons_total=int(sum(len(l) for l in buffer_labels)),
                rm_fit_loss=(float(rm.fit_loss) if use_rm else float("nan")),
                # vectors (trimmed by schema at write time)
                w_t=w_t.tolist(), m_t=m.tolist(),
                psi_hat=_psi_snapshot(rm, use_rm),
                ef_t=ef.tolist(),
            )
            rows.append(row)

            grader.step(m)                                   # w_t -> w_{t+1}

            # ---- backfill drift-direction metrics (need w_{t+1}) ---------
            u_t = M.drift_direction(w_t, grader.w)
            row["w_step"] = float(np.linalg.norm(grader.w - w_t))
            row["A_t"] = M.A_t(u_t, v_min) if u_t is not None else float("nan")
            row["VR_t"] = M.variance_ratio(u_t, diffs_round) if u_t is not None else float("nan")
            row["proj_angle"] = M.proj_angle(u_t, P) if u_t is not None else float("nan")

    # ---- post-hoc descriptive reference rewards -------------------------
    w_arr = np.array(w_hist)
    w_final, w_mean = w_arr[-1], w_arr.mean(axis=0)
    for row, ef in zip(rows, ef_hist):
        row["J_final"] = float(w_final @ np.array(ef))
        row["J_avg"] = float(w_mean @ np.array(ef))

    if record_w_traj:
        rows[0]["_w_traj"] = [w.tolist() for w in w_hist]
    # attach P once (row 0) for provenance / C6 proj-angle covariate
    rows[0]["P_flat"] = P.flatten().tolist()
    return rows


# --------------------------------------------------------------------------
# parquet io
# --------------------------------------------------------------------------
_VECTOR_COLS = ["w_t", "m_t", "psi_hat", "ef_t"]
_REDUCED_DROP = ["w_t", "m_t", "psi_hat", "ef_t"]  # keep only final w_T/m_T


def rows_to_frame(rows: List[dict], schema: str):
    import pandas as pd
    rows = [dict(r) for r in rows]
    for r in rows:
        r.pop("_w_traj", None)
    if schema == "reduced":
        T = len(rows)
        for i, r in enumerate(rows):
            wt, mt = r.get("w_t"), r.get("m_t")
            for c in _REDUCED_DROP:
                r.pop(c, None)
            r["w_T"] = wt if i == T - 1 else None
            r["m_T"] = mt if i == T - 1 else None
    df = pd.DataFrame(rows)
    # downcast float64 scalar columns to float32
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def write_parquet(rows: List[dict], path: str, schema: str = "reduced"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = rows_to_frame(rows, schema)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path
