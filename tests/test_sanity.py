"""Sanity suite (plan Section 6). A failing check blocks downstream analysis.
Each is an asserted test with a numeric tolerance. Thresholds fixed on dev seeds
and justified in EXPERIMENT_LOG.

Run:  python -m pytest tests/test_sanity.py -v
"""
import warnings

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader, DriftNone, DriftSat, DriftScale
from drifted_grader.reward_models import LinearBT, MLPBT, make_projection
from drifted_grader.metrics import policy_from_scores, regret_Rt, A_null
from drifted_grader.policies import expected_features
from drifted_grader.runner import run, Config


# ==========================================================================
# helpers
# ==========================================================================
def _fixed_fit(k, d=6, n=256, q=100_000, seed=0, betaT=10.0, shuffle=False):
    rng = np.random.default_rng(seed)
    arms = make_arms(n, d, rng)
    P = make_projection(d, k, rng)
    g = DriftNone(arms, betaT)
    i = rng.integers(0, n, q); j = rng.integers(0, n, q)
    mask = i != j; i, j = i[mask], j[mask]
    lab = g.label(np.stack([i, j], 1), rng).astype(float)
    if shuffle:
        lab = rng.permutation(lab)
    rm = LinearBT(d, P, reg=1e-3)
    rm.fit(arms.Phi[i] - arms.Phi[j], lab)
    return arms, P, rm


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ==========================================================================
# Correctness
# ==========================================================================
def test_P1_rm_recovery_full_basis():
    """P1: k=d, 1e5 comparisons -> cos(psi_hat, w) > 0.99. Verifies BT fitting."""
    for seed in range(3):
        arms, P, rm = _fixed_fit(k=6, seed=seed)
        assert _cos(rm.psi, arms.w0) > 0.99, f"seed {seed}: cos={_cos(rm.psi, arms.w0)}"


def test_P1b_rm_recovery_projected():
    """P1b: k<d recovers P w (pseudo-projection). cos(psi_hat, P w) > 0.94."""
    for seed in range(3):
        arms, P, rm = _fixed_fit(k=3, seed=seed)
        Pw = P @ arms.w0
        assert _cos(rm.psi, Pw) > 0.94, f"seed {seed}: cos={_cos(rm.psi, Pw)}"


def test_P2_policy_optimality():
    """P2: given true w and beta -> 0, Gibbs regret vs argmax < 1e-3."""
    for seed in range(3):
        rng = np.random.default_rng(seed)
        arms = make_arms(256, 6, rng)
        true = arms.Phi @ arms.w0
        pi = policy_from_scores(true, beta=1e-3)
        regret = float(true.max() - pi @ true)
        assert regret < 1e-3, f"seed {seed}: regret={regret}"


def test_P3_corrected_misspecification_overopt():
    """P3 (corrected, Section 3.5/6): misspecified RM (k<d) over-optimises.
    R_bar deepens monotonically as k shrinks; rises with pressure; k=d stays low."""
    betas = [2.0, 0.5, 0.1, 0.02, 0.005]
    q = 512
    Rbar_by_k = {}
    for k in [1, 2, 3, 4, 6]:
        allR = []
        rise = []
        for seed in range(8):
            arms, P, rm = _fixed_fit(k=k, q=q, seed=100 + seed)
            rhat = rm.predict(arms.Phi)
            Rs = [regret_Rt(policy_from_scores(rhat, b), arms.Phi, arms.w0, b)[0]
                  for b in betas]
            allR.extend(Rs)
            rise.append(Rs[-1] - Rs[0])   # high-pressure minus low-pressure
        Rbar_by_k[k] = np.mean(allR)
        if k < 6:
            assert np.mean(rise) > 0, f"k={k}: regret should rise with pressure"
    # monotone deepening as k shrinks
    ks = [6, 4, 3, 2, 1]
    vals = [Rbar_by_k[k] for k in ks]
    assert all(vals[i] <= vals[i + 1] + 1e-3 for i in range(len(vals) - 1)), \
        f"R_bar not monotone in k: {Rbar_by_k}"
    # misspecified >> correct
    assert Rbar_by_k[3] > 5 * Rbar_by_k[6], f"k=3 not >> k=6: {Rbar_by_k}"


def test_P3b_mlp_overoptimises_where_linear_clean():
    """P3b (Tandem-RL reframe): non-linear FA over-optimises where the correctly-
    specified linear RM is provably clean. Fixed grader, on-policy collection,
    regret at high eval pressure: MLP >> linear."""
    from drifted_grader.graders import sigmoid
    lin_hi, mlp_hi = [], []
    for seed in range(5):
        rng = np.random.default_rng(6000 + seed)
        arms = make_arms(256, 6, rng); Phi, w0 = arms.Phi, arms.w0
        pi = policy_from_scores(Phi @ w0, 0.1)                 # collect on-policy
        i = rng.choice(256, 512, p=pi); j = rng.choice(256, 512, p=pi)
        mk = i != j; i, j = i[mk], j[mk]
        p = sigmoid(10.0 * ((Phi @ w0)[i] - (Phi @ w0)[j]))
        lab = (rng.random(len(i)) < p).astype(float)
        pairs = np.stack([i, j], 1)
        lin = LinearBT(6).fit(Phi[i] - Phi[j], lab)
        mlp = MLPBT(6, hidden=64, steps=400, seed=seed).fit_buffer(Phi, pairs, lab)
        lin_hi.append(regret_Rt(policy_from_scores(lin.predict(Phi), 0.03), Phi, w0, 0.03)[0])
        mlp_hi.append(regret_Rt(policy_from_scores(mlp.predict(Phi), 0.03), Phi, w0, 0.03)[0])
    assert np.mean(lin_hi) < 0.03, f"linear not clean: {np.mean(lin_hi)}"
    assert np.mean(mlp_hi) > 3 * np.mean(lin_hi), \
        f"MLP not over-optimising: mlp={np.mean(mlp_hi)} lin={np.mean(lin_hi)}"


def test_tandem_linear_gap_zero():
    """Harness correctness: with a convex LinearBT, the passive learner recovers
    the same RM as the active (unique optimum), so tandem_gap == 0 exactly. The
    non-zero gap must come from non-linear FA, not the harness."""
    from drifted_grader.tandem import run_tandem
    cfg = Config(drift="SAT", lam=0.4, beta=0.05, d=6, T=25, q=64)
    rows = run_tandem(cfg, 0, base="linear", tandem="plain")
    gap = np.max(np.abs([r["tandem_gap"] for r in rows]))
    assert gap < 1e-9, f"linear tandem gap should be 0, got {gap}"


def test_P5_drift_monotonicity_and_simplex():
    """P5: under SAT, the highest-m-tilde dimension's weight (weakly) decreases,
    and the simplex constraint holds to 1e-10."""
    rng = np.random.default_rng(0)
    arms = make_arms(256, 6, rng)
    g = DriftSat(arms, 10.0, lam=0.4)
    m = np.zeros(6)
    for t in range(40):
        pi = policy_from_scores(arms.Phi @ g.w, 0.05)
        m = 0.5 * m + 0.5 * expected_features(pi, arms.Phi)
        m_tilde = np.clip(m / np.maximum(arms.per_dim_max, 1e-12), 0, 1)
        w_before = g.w.copy()
        g.step(m)
        assert abs(g.w.sum() - 1.0) < 1e-10, f"simplex violated: {g.w.sum()}"
        i_star = int(np.argmax(m_tilde))
        if m_tilde.ptp() > 1e-6:   # only when there is spread
            assert g.w[i_star] <= w_before[i_star] + 1e-12, \
                f"t={t}: top-m dim weight increased"


# ==========================================================================
# Nulls
# ==========================================================================
def test_N1_lambda_zero_no_compounding():
    """N1: lambda=0 (NONE) -> R_t is a flat misspec floor, no compounding
    (median 2nd-half slope ~ 0). The floor LEVEL varies across seeds because the
    random projection P aligns with w0 differently -- that seed-dependence is the
    C6 covariate, not instability -- so we test the slope, not the variance."""
    slopes = []
    for seed in range(8):
        cfg = Config(stage="B", drift="NONE", lam=0.0, beta=0.05, k=3,
                     rm="ensemble", T=100)
        rows = run(cfg, seed)
        R = np.array([r["R_t"] for r in rows])
        slopes.append(np.polyfit(np.arange(50, 100), R[50:], 1)[0])
        assert R.mean() < 0.30, f"seed {seed}: NONE floor implausibly high {R.mean()}"
    assert abs(np.median(slopes)) < 1e-3, f"NONE compounds: slope={np.median(slopes)}"


def test_N2_scale_argmax_invariance():
    """N2 core: DRIFT-SCALE is order-preserving -> argmax of the grader's
    comparison scores equals argmax of w0.phi at every theta (H2 mechanism)."""
    rng = np.random.default_rng(0)
    arms = make_arms(256, 6, rng)
    g = DriftScale(arms, 10.0, lam=0.5, kappa=4.0)
    base_argmax = int(np.argmax(arms.Phi @ arms.w0))
    m = np.zeros(6)
    for t in range(50):
        assert int(np.argmax(g.comp_scores())) == base_argmax, \
            f"t={t}: SCALE changed the argmax (not order-preserving)"
        pi = policy_from_scores(arms.Phi @ arms.w0, 0.05)
        m = 0.5 * m + 0.5 * expected_features(pi, arms.Phi)
        g.step(m)


def test_N2_scale_no_regret_stageA():
    """N2: under DRIFT-SCALE, Stage-A regret stays at the floor (agent tracking
    w0 is optimal since ordering is preserved)."""
    cfg = Config(stage="A", drift="SCALE", lam=0.5, beta=0.05, rm="none")
    rows = run(cfg, seed=0)
    Rbar = np.mean([r["R_t"] for r in rows])
    assert Rbar < 0.02, f"SCALE produced regret: R_bar={Rbar}"


def test_N3_d1_reduces_to_scale():
    """N3: d=1 -> SAT has nothing to reweight -> w constant -> R_t ~ 0."""
    cfg = Config(stage="A", drift="SAT", lam=0.8, beta=0.05, d=1, k=1, rm="none")
    rows = run(cfg, seed=0)
    Rbar = np.mean([r["R_t"] for r in rows])
    assert Rbar < 1e-6, f"d=1 SAT produced drift regret: R_bar={Rbar}"


def test_N4_label_shuffle_recovers_nothing():
    """N4: shuffled labels -> RM recovers nothing. psi_hat collapses to ~0, so
    its norm is a small fraction of a genuine fit's (cosine is meaningless for a
    near-zero vector, so we test the norm ratio)."""
    for seed in range(3):
        _, _, rm_real = _fixed_fit(k=3, q=20_000, seed=seed, shuffle=False)
        _, _, rm_shuf = _fixed_fit(k=3, q=20_000, seed=seed, shuffle=True)
        n_real = np.linalg.norm(rm_real.psi)
        n_shuf = np.linalg.norm(rm_shuf.psi)
        assert n_shuf < 0.1 * n_real, \
            f"seed {seed}: shuffled norm {n_shuf:.3f} vs real {n_real:.3f}"


def test_N5_endogenous_coupling_implemented():
    """N5: the closed loop actually couples m -> w. Feeding different achieved
    profiles must yield different next-weights (else REPLAY==closed-loop)."""
    rng = np.random.default_rng(0)
    arms = make_arms(256, 6, rng)
    g1 = DriftSat(arms, 10.0, lam=0.4)
    g2 = DriftSat(arms, 10.0, lam=0.4)
    m_a = np.zeros(6); m_a[0] = arms.per_dim_max[0]     # mastered dim 0
    m_b = np.zeros(6); m_b[3] = arms.per_dim_max[3]     # mastered dim 3
    g1.step(m_a); g2.step(m_b)
    assert np.linalg.norm(g1.w - g2.w) > 1e-3, "w does not respond to m"


def test_N6_correct_spec_cannot_overoptimise():
    """N6: k=d, well-covered comparisons -> R_bar <= 0.02 across the pressure
    sweep. The RM class is not silently generating the effect."""
    betas = [2.0, 0.5, 0.1, 0.02, 0.005]
    allR = []
    for seed in range(8):
        arms, P, rm = _fixed_fit(k=6, q=1024, seed=300 + seed)
        rhat = rm.predict(arms.Phi)
        allR.extend([regret_Rt(policy_from_scores(rhat, b), arms.Phi, arms.w0, b)[0]
                     for b in betas])
    assert np.mean(allR) <= 0.02, f"correct-spec R_bar={np.mean(allR)}"


def test_N7_oracle_tracking_floor():
    """N7: agent handed w_t exactly -> R_t < 1e-6 under EVERY drift incl SAT
    lam=0.8. Gates Stage B."""
    for drift in ["NONE", "SAT", "SCALE", "EXO"]:
        for lam in [0.0, 0.8]:
            for beta in [0.5, 0.02]:
                rng = np.random.default_rng(7)
                arms = make_arms(256, 6, rng)
                g = make_grader(drift, arms, 10.0, rng, lam=lam, T=60)
                m = np.zeros(6)
                for t in range(60):
                    pi = policy_from_scores(arms.Phi @ g.w, beta)   # handed w_t
                    R, _ = regret_Rt(pi, arms.Phi, g.w, beta)
                    assert R < 1e-6, f"{drift} lam={lam} beta={beta} t={t}: R={R}"
                    m = 0.5 * m + 0.5 * expected_features(pi, arms.Phi)
                    g.step(m)


# ==========================================================================
# Invariances / infra
# ==========================================================================
def test_C1_affine_invariance():
    """C1: grader BT logits are shift-invariant; scale is absorbed by the
    range-fix (BT logit scales by a, range scales by a, ratio unchanged)."""
    rng = np.random.default_rng(0)
    arms = make_arms(256, 6, rng)
    Phi, w = arms.Phi, arms.w0
    pairs = np.stack([rng.integers(0, 256, 200), rng.integers(0, 256, 200)], 1)
    base = w @ (Phi[pairs[:, 0]] - Phi[pairs[:, 1]]).T
    # shift invariance
    shifted = w @ ((Phi + 3.7)[pairs[:, 0]] - (Phi + 3.7)[pairs[:, 1]]).T
    assert np.allclose(base, shifted), "BT not shift-invariant"
    # scale absorbed by range renormalisation
    a = 2.5
    Phi_s = a * Phi
    logit_s = w @ (Phi_s[pairs[:, 0]] - Phi_s[pairs[:, 1]]).T
    range_s = (Phi_s @ w).ptp()
    range_0 = (Phi @ w).ptp()
    assert np.allclose(base, logit_s * (range_0 / range_s)), "scale correction wrong"


def test_A_null_matches_montecarlo():
    """A_t null: exact Gamma formula matches Monte Carlo E|u.v| to MC error."""
    for d in [3, 6, 10]:
        rng = np.random.default_rng(0)
        U = rng.standard_normal((200_000, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
        V = rng.standard_normal((200_000, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        mc = np.mean(np.abs(np.sum(U * V, axis=1)))
        assert abs(A_null(d) - mc) < 5e-3, f"d={d}: exact={A_null(d)} mc={mc}"


def test_reproducibility_byte_identical():
    """Identical (cfg, seed) reproduces identical output."""
    cfg = Config(stage="B", drift="SAT", lam=0.2, beta=0.05, k=3, rm="ensemble", T=30)
    r1 = run(cfg, seed=3)
    r2 = run(cfg, seed=3)
    for a, b in zip(r1, r2):
        assert a["R_t"] == b["R_t"] and a["A_t"] == b["A_t"] and a["J_init"] == b["J_init"]


def test_numerical_stability_betaT_sweep():
    """Section 7.4: no over/invalid/divide warnings across betaT in {2,5,10,50}."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for betaT in [2.0, 5.0, 10.0, 50.0]:
            cfg = Config(stage="B", drift="SAT", lam=0.2, beta=0.02, betaT=betaT,
                         k=3, rm="ensemble", T=20)
            run(cfg, seed=0)
