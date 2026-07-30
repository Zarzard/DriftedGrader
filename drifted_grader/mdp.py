"""Tabular MDP with preference-based RL, a bootstrapped (fitted-Q) value learner,
and preference drift (plan Stage E). Built to test whether VALUE BOOTSTRAPPING --
the ingredient the Tandem-RL paper (Ostrovski et al. 2021) found necessary and the
bandit lacks -- resurrects a closed-loop / endogeneity effect.

Pieces:
  TabularMDP        : P(s'|s,a), phi(s,a) in R^d, rho0, gamma.
  drift grader      : w_t on the d-simplex, drifting on the agent's achieved
                      state-action feature profile (reuses the SAT/SCALE/EXO/REPLAY
                      formulas); labels preferences over (s,a) pairs by w_t.phi.
  reward model      : LinearBT / MLPBT on phi(s,a) (reused from reward_models; the
                      (s,a) pairs are flattened to indices).
  FittedQ agent     : Q_theta(s,a) via linear / MLP function approx, trained by
                      model-free fitted-Q on SAMPLED transitions:
                        y = r_hat(s,a) + gamma * max_a' Q(s',a').
                      The max-over-own-Q is the bootstrap (B); coverage gaps in the
                      sampled data + function approx (F) drive over-estimation.

Regret R_t = [ V*(w_t) - V^pi(w_t) ] / range_t, greedy policy vs the value-iteration
optimum under the CURRENT grader's true reward w_t.phi. N7 analog: an agent given
the true reward recovers the optimum -> R_t = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reward_models import LinearBT, MLPBT


# --------------------------------------------------------------------------
# MDP
# --------------------------------------------------------------------------
@dataclass
class TabularMDP:
    P: np.ndarray            # (S, A, S)
    phi: np.ndarray          # (S, A, d) non-negative
    rho0: np.ndarray         # (S,)
    gamma: float
    w0: np.ndarray           # (d,) initial grader weights
    per_dim_max: np.ndarray  # (d,)

    @property
    def S(self): return self.phi.shape[0]
    @property
    def A(self): return self.phi.shape[1]
    @property
    def d(self): return self.phi.shape[2]

    def phi_flat(self):
        return self.phi.reshape(self.S * self.A, self.d)


def make_mdp(n_s, n_a, d, rng, gamma=0.95, conc=0.4):
    # transitions: each (s,a) -> a somewhat concentrated distribution over s'
    P = np.stack([[rng.dirichlet(conc * np.ones(n_s)) for _ in range(n_a)]
                  for _ in range(n_s)])
    phi = rng.uniform(0, 1, (n_s, n_a, d))
    # standardise per feature over (s,a), shift non-negative (match arms geometry)
    flat = phi.reshape(-1, d)
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-12)
    flat = flat - flat.min(0)
    phi = flat.reshape(n_s, n_a, d)
    w0 = rng.dirichlet(np.ones(d))
    rho0 = rng.dirichlet(np.ones(n_s))
    per_dim_max = phi.reshape(-1, d).max(0)
    return TabularMDP(P=P, phi=phi, rho0=rho0, gamma=gamma, w0=w0,
                      per_dim_max=per_dim_max)


# --------------------------------------------------------------------------
# planning / evaluation under a given weight vector w (TRUE reward w.phi)
# --------------------------------------------------------------------------
def q_star(mdp: TabularMDP, w, iters=1000, tol=1e-10):
    r = mdp.phi @ w                       # (S,A)
    Q = np.zeros((mdp.S, mdp.A))
    for _ in range(iters):
        V = Q.max(1)
        Qn = r + mdp.gamma * (mdp.P @ V)
        if np.max(np.abs(Qn - Q)) < tol:
            Q = Qn; break
        Q = Qn
    return Q


def policy_value(mdp: TabularMDP, w, greedy, iters=2000, tol=1e-10):
    """Value of a deterministic greedy policy (array of actions) under w.phi."""
    r = mdp.phi @ w
    idx = np.arange(mdp.S)
    Pg = mdp.P[idx, greedy]               # (S, S)
    rg = r[idx, greedy]                   # (S,)
    V = np.zeros(mdp.S)
    for _ in range(iters):
        Vn = rg + mdp.gamma * (Pg @ V)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn; break
        V = Vn
    return V


def _q_worst(mdp: TabularMDP, w, iters=1000, tol=1e-10):
    r = mdp.phi @ w; Q = np.zeros((mdp.S, mdp.A))
    for _ in range(iters):
        V = Q.min(1); Qn = r + mdp.gamma * (mdp.P @ V)
        if np.max(np.abs(Qn - Q)) < tol:
            Q = Qn; break
        Q = Qn
    return Q


def regret(mdp: TabularMDP, w, greedy):
    """Normalised regret of a greedy policy vs the optimum, scaled by the
    achievable return spread (optimal - worst policy) so R in [0, 1]."""
    Vs = q_star(mdp, w).max(1)
    Vw = _q_worst(mdp, w).min(1)
    Vpi = policy_value(mdp, w, greedy)
    ret_opt = float(mdp.rho0 @ Vs); ret_pi = float(mdp.rho0 @ Vpi)
    ret_worst = float(mdp.rho0 @ Vw)
    denom = (ret_opt - ret_worst) + 1e-9
    return max(0.0, (ret_opt - ret_pi) / denom)


# --------------------------------------------------------------------------
# fitted-Q value learner (bootstrapping = max over own Q)
# --------------------------------------------------------------------------
class _LinQ:
    def __init__(self, d, seed):
        self.th = np.random.default_rng(seed).standard_normal(d) * 0.01
    def predict(self, X): return X @ self.th
    def regress(self, X, y, **kw):
        self.th = np.linalg.lstsq(X, y, rcond=None)[0]


class _MLPQ:
    def __init__(self, d, seed, h=64):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((h, d)) * np.sqrt(1 / d); self.b1 = np.zeros(h)
        self.w2 = rng.standard_normal(h) * np.sqrt(1 / h); self.b2 = 0.0
    def _fwd(self, X):
        H = np.tanh(X @ self.W1.T + self.b1); return H @ self.w2 + self.b2, H
    def predict(self, X): return self._fwd(X)[0]
    def regress(self, X, y, steps=60, lr=3e-2):
        for _ in range(steps):
            pred, H = self._fwd(X); e = (pred - y) / len(y)
            dw2 = H.T @ e; db2 = e.sum(); dZ = (e[:, None] * self.w2) * (1 - H ** 2)
            self.W1 -= lr * (dZ.T @ X); self.b1 -= lr * dZ.sum(0)
            self.w2 -= lr * dw2; self.b2 -= lr * db2


class FittedQ:
    """Model-free fitted-Q with function approximation over phi(s,a)."""
    def __init__(self, mdp: TabularMDP, base="mlp", seed=0, hidden=64):
        self.mdp = mdp; self.base = base
        self.Phi = mdp.phi_flat()                       # (S*A, d)
        self.net = (_MLPQ(mdp.d, seed, hidden) if base == "mlp"
                    else _LinQ(mdp.d, seed))

    def Q(self):
        return self.net.predict(self.Phi).reshape(self.mdp.S, self.mdp.A)

    def greedy(self):
        return self.Q().argmax(1)

    def policy(self, beta):
        q = self.Q() / beta
        q = q - q.max(1, keepdims=True)
        e = np.exp(q); return e / e.sum(1, keepdims=True)

    def fit(self, S, Aa, SP, rhat_sa, outer=25):
        """S,Aa,SP: sampled transition arrays; rhat_sa: (S,A) reward from RM."""
        X = self.mdp.phi[S, Aa]
        r = rhat_sa[S, Aa]
        for _ in range(outer):
            Q = self.Q()
            Vp = Q[SP].max(1)                            # bootstrap: max over own Q
            y = r + self.mdp.gamma * Vp
            self.net.regress(X, y)


# --------------------------------------------------------------------------
# drift grader over (s,a) preferences
# --------------------------------------------------------------------------
def _sigmoid(x):
    out = np.empty_like(x, float); p = x >= 0
    out[p] = 1 / (1 + np.exp(-x[p])); e = np.exp(x[~p]); out[~p] = e / (1 + e)
    return out


class MDPGrader:
    """w_t drifts on the achieved state-action feature profile; labels (s,a)-pair
    comparisons by w_t.phi with inverse temperature betaT."""
    def __init__(self, mdp, drift="SAT", betaT=10.0, lam=0.2, T=100,
                 rng=None, w_traj=None):
        self.mdp = mdp; self.drift = drift.upper(); self.betaT = betaT
        self.lam = lam; self.w = mdp.w0.copy(); self.t = 0
        self.per_dim_max = mdp.per_dim_max
        self.phi_flat = mdp.phi_flat()
        if self.drift == "EXO":
            w_end = rng.dirichlet(np.ones(mdp.d))
            fr = np.linspace(0, 1, T); self.sched = [(1 - f) * mdp.w0 + f * w_end for f in fr]
        if self.drift == "REPLAY":
            self.w_traj = [np.asarray(w, float) for w in w_traj]; self.w = self.w_traj[0].copy()

    def label(self, pairs, rng):
        s = self.phi_flat @ self.w                      # (S*A,)
        logit = self.betaT * (s[pairs[:, 0]] - s[pairs[:, 1]])
        return (rng.random(len(pairs)) < _sigmoid(logit)).astype(np.int64)

    def step(self, m):
        self.t += 1
        if self.drift == "SAT":
            mt = np.clip(m / np.maximum(self.per_dim_max, 1e-12), 0, 1)
            wn = self.w * np.exp(-self.lam * mt); self.w = wn / wn.sum()
        elif self.drift == "EXO":
            self.w = self.sched[min(self.t, len(self.sched) - 1)].copy()
        elif self.drift == "REPLAY":
            self.w = self.w_traj[min(self.t, len(self.w_traj) - 1)].copy()
        # NONE: unchanged


# --------------------------------------------------------------------------
# rollouts + closed loop
# --------------------------------------------------------------------------
def rollout(mdp, agent, eps, n_ep, H, rng):
    """Roll out n_ep episodes with eps-greedy on the agent's Q. Returns sampled
    transitions (S,A,SP), the round's achieved feature profile, and the (s,a)
    visitation counts (flattened index)."""
    Q = agent.Q()
    Ss, As, SPs = [], [], []
    occ = np.zeros(mdp.S * mdp.A)
    for _ in range(n_ep):
        s = rng.choice(mdp.S, p=mdp.rho0)
        for _ in range(H):
            a = rng.integers(mdp.A) if rng.random() < eps else int(Q[s].argmax())
            sp = rng.choice(mdp.S, p=mdp.P[s, a])
            Ss.append(s); As.append(a); SPs.append(sp); occ[s * mdp.A + a] += 1
            s = sp
    S, A, SP = np.array(Ss), np.array(As), np.array(SPs)
    feat = (mdp.phi[S, A].mean(0) if len(S) else np.zeros(mdp.d))  # achieved profile
    occ = occ / max(1, occ.sum())
    return S, A, SP, feat, occ


def _sample_pairs(occ, q, rng):
    n = len(occ)
    p = occ if occ.sum() > 0 else np.full(n, 1 / n)
    i = rng.choice(n, q, p=p); j = rng.choice(n, q, p=p)
    mk = i != j
    return np.stack([i[mk], j[mk]], 1)


def _make_rm(base, d, seed, hidden, steps):
    return (MLPBT(d, hidden=hidden, steps=steps, seed=seed) if base == "mlp"
            else LinearBT(d))


def run_active(mdp, drift, base, seed, T=60, betaT=10.0, lam=0.2, beta=0.1,
               eps=0.1, H=15, n_ep=8, q=64, alpha=0.5, hidden=64, steps=60,
               agent_outer=8, w_traj=None, rm_seed=1000, q_seed=1):
    """Closed loop: fitted-Q agent drives rollouts, grader drifts on achieved
    features, RM fit on (s,a) preferences. Returns (rows, streams, w_hist)."""
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + q_seed)
    Phi = mdp.phi_flat()
    grader = MDPGrader(mdp, drift, betaT, lam, T, rng, w_traj)
    rm = _make_rm(base, mdp.d, rm_seed + seed, hidden, steps)
    agent = FittedQ(mdp, base, seed=rm_seed + seed, hidden=hidden)
    all_pairs, all_lab = [], []
    tr_S, tr_A, tr_SP = [], [], []
    m = np.zeros(mdp.d); rows = []; w_hist = []
    for t in range(T):
        w_t = grader.w.copy(); w_hist.append(w_t)
        S, A, SP, feat, occ = rollout(mdp, agent, eps, n_ep, H, rng)
        tr_S.append(S); tr_A.append(A); tr_SP.append(SP)
        pairs = _sample_pairs(occ, q, arng); labels = grader.label(pairs, arng)
        all_pairs.append(pairs); all_lab.append(labels)
        rm.fit_buffer(Phi, np.concatenate(all_pairs),
                      np.concatenate(all_lab).astype(float), arng)
        rhat = rm.predict(Phi).reshape(mdp.S, mdp.A)
        agent.fit(np.concatenate(tr_S), np.concatenate(tr_A), np.concatenate(tr_SP),
                  rhat, outer=agent_outer)
        R = regret(mdp, w_t, agent.greedy())
        m = (1 - alpha) * m + alpha * feat
        rows.append(dict(t=t, R_active=R, w=w_t.tolist()))
        grader.step(m)
    streams = dict(pairs=all_pairs, labels=all_lab, tr_S=tr_S, tr_A=tr_A, tr_SP=tr_SP)
    return rows, streams, w_hist


def run_passive_on(mdp, streams, w_hist, base, seed, beta=0.1, hidden=64, steps=60,
                   agent_outer=8, rm_seed=9000):
    """Passive learner: RM + fitted-Q trained on the ACTIVE's cumulative streams
    each round (independent init), policy never drives. Returns R_passive per round."""
    Phi = mdp.phi_flat()
    rm = _make_rm(base, mdp.d, rm_seed + seed, hidden, steps)
    agent = FittedQ(mdp, base, seed=rm_seed + seed, hidden=hidden)
    arng = np.random.default_rng(seed + 12345)
    T = len(w_hist); Rp = []
    for t in range(T):
        rm.fit_buffer(Phi, np.concatenate(streams["pairs"][:t + 1]),
                      np.concatenate(streams["labels"][:t + 1]).astype(float), arng)
        rhat = rm.predict(Phi).reshape(mdp.S, mdp.A)
        agent.fit(np.concatenate(streams["tr_S"][:t + 1]),
                  np.concatenate(streams["tr_A"][:t + 1]),
                  np.concatenate(streams["tr_SP"][:t + 1]), rhat, outer=agent_outer)
        Rp.append(regret(mdp, np.array(w_hist[t]), agent.greedy()))
    return np.array(Rp)
