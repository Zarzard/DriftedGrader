<!-- Transcribed 2026-08-05 from user-provided attachment; encoding (mojibake) fixed, content otherwise verbatim. -->

# Tier-A Experimental Design: Grader Drift in Preference Learning
## Executable specification for a coding agent

*v0.1 — August 2026. Companion to `grader_drift_identifiability_note.md` (referenced below as "the note"). Section references like §4 point to the note.*

---

## 0. Purpose and scope

Build a simulation suite that (a) demonstrates the operational-stream attribution failure (§4), (b) validates the anchor instrument and its split rule (§5), (c) validates a state-space drift estimator with calibrated uncertainty, (d) runs the headline regression: does anchor-measured drift predict gold-return harm, stratified by on-support fraction rho, and (e) establishes the matched-budget refresh baseline.

Two rungs:

- **A0 (measurement rung):** no RL. Items are feature vectors; "policy improvement" is simulated by a shrinking pair-gap distribution. Isolates instrument + estimator questions from RL confounds. Most experiments live here.
- **A1 (loop rung):** tabular PbRL on a gridworld. Real closed loop: policy -> segments -> drifting teacher labels -> reward model -> policy. Provides the RL replication and the gold-gap outcome.

**Out of scope for this build:** Tier-B continuous control, any drift-correction algorithm, LLM judges, human data, active query selection. Do not add features beyond this spec.

---

## 1. Conventions

- Python 3.11. Dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`, `pyyaml`. No GPU, no deep-learning frameworks. Everything must run on CPU; a full A0 run should take < 1 min, a full A1 run < 10 min.
- Repo layout:

```
graderdrift/
  src/graderdrift/
    teacher.py        # C1
    drift.py          # C2
    streams.py        # C3 (A0 item streams)
    anchors.py        # C4
    probe.py          # C5
    diagnostics.py    # C6
    estimator.py      # C7
    envs/grid.py      # C8 (A1)
    rm.py             # C8
    policy.py         # C8
    experiments/      # e1.py ... e6.py, each a thin driver
    analysis/         # aggregate.py, figures.py, power.py
  configs/            # yaml per experiment + schedule presets
  tests/              # proposition-derived unit tests (Section 6)
  runs/               # output tree (gitignored)
```

- Every run is fully specified by one YAML config + one integer seed. One master seed spawns named `numpy.random.Generator` children per component (`teacher`, `drift`, `stream`, `labels`, `probe`, `policy`, `rm`); document the spawning so runs are reproducible component-wise.
- Logging: one JSONL row per round per run at `runs/{exp}/{schedule}/{seed}/log.jsonl`, plus a copy of the resolved config. Row schema (superset; unused fields null):

```
{round, seed, schedule,
 v_true, v_hat_mean, v_hat_cov_diag, v_hat_loglik,
 anchor_p_hat: [..], D_dir, D_norm,
 rho_cum, rho_step,
 op_agree, op_flip, op_temp_inv,
 gold_return, proxy_return, e_shift, e_epi,
 n_labels_cum, event_flags: [..]}
```

- `analysis/aggregate.py` walks `runs/` and emits tidy CSVs; `analysis/figures.py` produces the named figures. Analysis never reads anything except the JSONL logs.
- Functions pure where possible; all stochastic calls take an explicit `Generator`.

---

## 2. Core math the code implements

- Utility: `u_t(x) = w_t . phi(x) + s_t`. Labels: `P_t(x > y) = sigma(beta_t (u_t(x) - u_t(y))) = sigma(v_t . dphi)` with `v_t = beta_t w_t`, `dphi = phi(x) - phi(y)`, `sigma(z) = 1/(1+exp(-z))`, `sigma'(z) = sigma(z)(1-sigma(z))`.
- The simulator carries `(w_t, s_t, beta_t)` explicitly and exposes `v_t`; `s_t` exists solely so the unit tests can verify Prop 1 (it must never affect comparison labels).
- Norm-sensitivity optimum: `|z.sigma'(z)|` maximized at `|z*| = 1.5434`, `p* = sigma(z*) = 0.8237`.
- On-support fraction: given realized battery gaps `{dphi_j}` stacked as `Phi_A`, let `P_A` project onto `rowspace(Phi_A)` (via SVD, tolerance 1e-10). For a drift increment `delta_k = v_k - v_{k-1}`: `rho_step = ||P_A delta_k||^2 / ||delta_k||^2`; `rho_cum` uses `delta = v_k - v_0`.

---

## 3. Components

### C1. Teacher and population (`teacher.py`)

`Teacher` holds `(w, s, beta)` and exposes:
- `v()` -> `beta * w`.
- `label(dphi, rng)` -> Bernoulli(sigma(v.dphi)).
- `label_from_scores(phi_x, phi_y, t_x, t_y, rng)` -> uses full `u` including `s` at possibly different states -> exists only for the Prop-1/Cor-1 unit tests.

`Population`: list of `M` teachers + weights. A comparison is labeled by one sampled rater (per-rater severity cancellation holds by construction). Default `M = 1`; churn schedules use `M = 10`, uniform weights.

Defaults (A0): `d = 24`. Historical subspace `S_hist = span(e_1..e_16)`, novel `S_nov = span(e_17..e_24)`. `w_0`: random unit vector **inside S_hist** (this makes rho-targeting exact; see C2). `beta_0 = 1.0`, so `||v_0|| = 1`. `s_0 = 0` with arbitrary drift allowed in tests.

### C2. Drift generator (`drift.py`)

State transition applied once per round, driven by an **event calendar** plus optional per-round processes. Config schema:

```yaml
schedule:
  name: S1
  events:
    - {t: 30, type: direction_jump, angle_deg: 25, rho_target: 1.0}
    - {t: 30, type: norm_jump, factor: 0.6}          # example
    - {t: 30, type: family_swap, cos_target: 0.7, norm_factor: 1.3}
    - {t: 10, type: mixture_redraw, fraction: 0.3}
  per_round:
    dir_diffusion_sigma: 0.0      # radians-scale
    norm_diffusion_sigma: 0.0     # lognormal sigma
    norm_decay: {from: null, to: null, factor: 1.0}
    gradient_coupled: null        # E6 only: {eta_deg: 2.0, sign: +1}
```

Implementations (all operate on `vhat = v/||v||` and `||v||`; then recompose; `w <- v/beta` keeping `beta` fixed unless the event says otherwise — the factorization is not identified anyway, Prop 2):

- `direction_jump(angle theta, rho_target rho*)`: rotation axis `u perp vhat` built as `u = normalize(sqrt(rho*).u_h + sqrt(1-rho*).u_n)`, `u_h` random unit in `S_hist intersect vhat-perp`, `u_n` random unit in `S_nov`. Then `vhat <- cos(theta).vhat + sin(theta).u`. Because `v_0 in S_hist` and the battery is built from historical items (C4), realized rho of the jump ~ rho*; always log realized rho, never trust the target.
- `norm_jump(factor)` / `norm_decay`: multiply `||v||` (decay: by `factor` each round in `[from, to]`).
- `family_swap(cos_target, norm_factor)`: resample `vhat_new` with `cos(vhat_new, vhat_old) = cos_target` (component in a random orthogonal direction, drawn within `S_hist` unless `rho_target` given), set `||v|| <- norm_factor.||v||`. Models a judge/version swap.
- `mixture_redraw(fraction)`: population mode; replace that fraction of raters with fresh draws `vhat ~ vMF(vhat_pop_mean, kappa=20)` in `S_hist`, `||v|| ~ LogNormal(0, 0.1)`.
- Diffusion per round: `vhat <- normalize(vhat + sigma_dir.eps_perp)`, `eps_perp = (I - vhat.vhat^T)eps`, `eps ~ N(0, I_d)` restricted to `S_hist` unless configured otherwise; `||v|| <- ||v||.exp(sigma_norm xi - sigma_norm^2/2)`.
- `gradient_coupled(eta_deg, sign)` (E6): rotate `vhat` by `eta` per round toward (`sign=+1`) or away from `normalize(P_{vhat-perp} gbar)`, where `gbar = normalize(E_{pi_k}[phi] - E_{pi_{k-1}}[phi])` supplied by the A1 loop; skip if `||E_{pi_k}[phi] - E_{pi_{k-1}}[phi]|| < 1e-3`.

**Schedule presets** (used throughout; K = 60 rounds A0, 40 rounds A1):

| id | content |
|---|---|
| S0 | stationary |
| S1 | direction_jump 25deg at t=30, rho*=1.0 |
| S1n | direction_jump 25deg at t=30, rho*=0.0 (novel-subspace jump) |
| S1m | direction_jump 25deg at t=30, rho*=0.5 |
| S2 | norm_decay factor 0.985/round from t=20 |
| S3 | mixture_redraw 30% every 10 rounds (M=10) |
| S4 | dir_diffusion_sigma 0.02/round |
| S5 | family_swap at t=30, cos 0.7, norm x1.3 |
| S2j | norm_jump x0.6 at t=30 (pure scale, for E2 factorial) |
| S1p | direction_jump 25deg at t=30, rho*=1.0, **no norm change** (pure rotation, E2) |

### C3. A0 item streams (`streams.py`)

Operational pairs at round k: `dphi ~ N(0, c_k^2 Sigma_k)` where the gap scale `c_k = c_0 . gamma^k`, `c_0 = 2.0`, `gamma = 0.95` (the JND concentration knob). `Sigma_k`: starts as identity on `S_hist`, zero on `S_nov`; from round `t_nov = 20`, novel-dimension variance ramps linearly to `0.5.c_k^2` by round 40 (policy exploring new behavior). `n_pairs = 100` per round. A duplicate-label fraction `q_dup = 0.15` of pairs is labeled twice (independent label draws; in population mode, two independently sampled raters) — feeds C6.

### C4. Anchor battery (`anchors.py`)

Built **once at round 0, then frozen**. Candidate pool: `N_cand = 5000` pairs with `dphi` drawn from the round-0 historical distribution (`N(0, c_0^2 I)` on `S_hist`, exactly zero on `S_nov`). Compute `z_j = v_0.dphi_j`.

- **Direction panel** (`m_dir = 16`): candidates with `|p_j - 0.5| <= 0.05`. Greedy D-optimal: start `A = 1e-6.I`; repeatedly add the candidate maximizing `log det(A + sigma'(z_j)dphi_j dphi_j^T)`.
- **Norm panel** (`m_norm = 8`): candidates with `p_j in [0.78, 0.86] union [0.14, 0.22]`; pick the 8 with `|z_j|` closest to 1.5434, balanced 4/4 in sign.
- Store `Phi_A` (24 x d), compute `P_A` once. By construction `rowspace(Phi_A) subset S_hist`; assert its rank (expected 16) and log it.
- A1 variant: the candidate pool is `dphi` between segment pairs collected at round 0 from a reference mixture (uniform-random policy 50% + eps-greedy initial policy 50%); same selection rules against `v_0`.

### C5. Probing protocol (`probe.py`)

Every round, after drift and before anything else uses the teacher: draw `n_probe = 50` labels per anchor from the current teacher state (population mode: rater resampled per label). Anchors never enter RM training data. Log `phat_j(t)` and the two panel summaries:

- `D_dir(t) = mean_{j in dir} |phat_j(t) - phat_j(0)|`
- `D_norm(t) = mean_{j in norm} sign(z_j(0)) . (logit(clip(phat_j(t))) - logit(clip(phat_j(0))))`, clip to `[0.02, 0.98]`. Negative = slide toward 0.5.

### C6. Operational diagnostics (`diagnostics.py`) — the "naive detector" family

Computed **only** from the operational stream (never from anchors):

- `op_agree(k)`: agreement rate on the `q_dup` double-labeled pairs.
- `op_flip(k)`: `1 - op_agree` (kept separate for plotting convention).
- `op_temp_inv(k)`: fit a scalar `tau` by 1-D MLE on the round's fresh labels given the current reward model direction `what_k` (A0: given `vhat_hat` from a running logistic fit on operational data; A1: the RM): maximize `Sum log sigma(tau.(what_k.dphi_i).(2y_i-1))`. Report `1/tau_hat` (effective temperature).

These are the kappa-hat-analogues; E1 exists to show they move under S0.

### C7. State-space estimator (`estimator.py`)

Iterated EKF / Laplace filter. State `v in R^d`.

- Dynamics: `v_{k+1} = v_k + q_k`, `q_k ~ N(0, Q_k)`.
  - `known_calendar` mode: `Q_k = sigma_diff^2 I + 1[event at k].sigma_jump^2 I`, with `sigma_diff = 0.01`, `sigma_jump = 0.5` (config).
  - `blind` mode: constant `Q_k = sigma_blind^2 I`, `sigma_blind = 0.05`.
- Measurement at k: for each anchor, `y_jk ~ Binomial(n_probe, sigma(v.dphi_j))`. Update by iterated EKF (3 Newton iterations on the Laplace objective; observed information for the Hessian).
- Init: `vhat_0` = MAP fit to the round-0 probe with prior `N(0, 10.I)`; `Sigma_0` from the Laplace Hessian.
- Outputs per round: mean, covariance, marginal log-lik. Derived: 90% CIs for each coordinate of `P_A v` and of `v`; intervals for `||v||` and `angle(vhat, vhat_0)` by sampling 2000 draws from the filtered Gaussian.
- Primary drift statistic for E4: `TV = Sum_k ||vhat_k - vhat_{k-1}||` minus the S0 noise floor (mean TV over the E1 S0 runs at the same config; compute once, store in `analysis/noise_floor.json`).

### C8. A1 PbRL loop (`envs/grid.py`, `rm.py`, `policy.py`)

- **Env:** 10x10 gridworld, actions {up, down, left, right, stay}, slip 0.05, episode = segment of `H = 12` steps from random spawn. Cell types: goal (1 cell), lava (6), water (8), grass (rest). Features `phi(s) in R^8`: [goal_reached, -normalized_manhattan_to_goal, lava, water, grass, wall_bump, stay_action, 1 (step cost)]. Segment features: unweighted sum over steps. `d = 8`; **no rho stratification in A1** — the battery will effectively span the space; A1 replicates the on-support story only. State this in the paper as a deliberate division of labor between rungs.
- **True reward:** `w* = [+1.0, +0.3, -1.0, -0.3, 0, -0.1, -0.02, -0.02]`. Gold return `J_{w*}(pi)`: exact policy evaluation, gamma = 0.97, uniform start distribution.
- **Teacher:** `w_0 = w*`, `beta_0 = 3.0` (reliable initial labels). Drift schedules re-used; jumps in random planes within R^8.
- **Round loop (K = 40):** (1) collect `n_seg = 200` segments: 80% current policy eps-greedy (eps=0.2), 20% uniform policy; (2) form `n_pairs = 100` random pairs; (3) label with current teacher; double-label `q_dup`; (4) fit RM: logistic regression (L2, lambda=1e-3, `scipy.optimize`) on all accumulated pairs -> `what_k`; (5) policy: soft value iteration on `rhat = what_k.phi(s)` -> 80 sweeps, `pi propto exp(Q/tau_pol)`, `tau_pol = 0.05`; (6) probe anchors; (7) log gold return, proxy return, diagnostics.
- **Error decomposition logging (Axis-2 hook):** `e_mis = 0` by construction (RM family contains teacher). Per round compute the `v_t`-optimal policy (soft VI on `v_t.phi`) and log `e_shift = J_{w*}(pi_{w*}) - J_{w*}(pi_{v_t})` and `e_epi = J_{w*}(pi_{v_t}) - J_{w*}(pi_{what_k})`.
- **Gold gap (harm outcome):** run every (schedule, seed) **paired with an S0 control at the same seed** using common random numbers for env and stream components (labels differ necessarily). `G_auc = Sum_{k >= first_event} [J_{w*}^{S0}(k) - J_{w*}^{sched}(k)] / (K - first_event)`; also `G_final` at k = K.

---

## 4. Experiments

Each experiment = config file + driver + analysis function + acceptance criterion. Run order: E1 -> E2 -> E3 -> E4 pilot -> power -> E4 full -> E5. E6 is specced but gated.

### E1 — The stationarity null (Figure 1)

- **Setup:** A0, schedule S0, 20 seeds. Also A1/S0, 10 seeds.
- **Prediction (Prop 3):** `op_agree` falls / `op_temp_inv` rises as `c_k` shrinks; both anchor panels flat.
- **Analysis:** per-run OLS slope over rounds 5-60 for each operational stat (report standardized effect + CI across seeds). Anchors: slope of `D_dir` and `D_norm`; equivalence criterion |slope| < 0.001/round (<= 0.06 absolute drift over the run) with 90% CI inside the bound.
- **Acceptance:** operational-stat slopes significantly != 0 in predicted direction; anchor slopes pass equivalence. Figure 1: two-panel plot, operational stats (left) vs `D_dir`, `D_norm` (right), mean +/- seed band.

### E2 — Split-rule validation (the exact-zero cell)

- **Setup:** A0, 2x2 factorial: schedules {S1p pure rotation, S2j pure norm} x responses {`D_dir`, `D_norm`}, 20 seeds each.
- **Predictions:** (i) `D_dir` under S2j is **exactly zero in expectation** (scaling does nothing at z = 0) — the sharp falsifiable cell; sampling noise only. (ii) `D_dir` under S1p large. (iii) `D_norm` responds to both (rotation moves norm-anchor z too) but with the S2j response matching the theoretical `Delta-z = log(0.6)` per-anchor prediction within CI.
- **Analysis:** post-event mean of each response per run; cell means +/- CI; for cell (i), equivalence test against the pure-sampling-noise null (simulated by relabeling round-30 state).
- **Acceptance:** cell (i) statistically indistinguishable from sampling noise; ordering of other cells as predicted.

### E3 — Estimator calibration

- **Setup:** A0, schedules {S0, S1, S2, S3, S4, S5}, 20 seeds each; estimator in both modes.
- **Metrics:** empirical coverage of nominal-90% per-coordinate CIs, separately for `P_A v` and full `v`; NEES; norm/angle interval coverage.
- **Acceptance:** `known_calendar` coverage for `P_A v` in [85%, 95%] pooled and per-schedule. `blind` mode: report; expected failure localized at jump rounds (this *is* the Glicko point — a finding, not a bug). Full-`v` coverage expected ~ prior-driven on the novel component; report, don't gate.
- **Deliverable:** coverage table + one figure of filtered `||vhat||`/angle trajectories vs truth with bands, S1 and S5.

### E4 — Headline regression: does anchor-measured drift predict harm?

- **Pilot:** A1, schedules {S1, S2, S3, S4, S5} x 10 seeds (each with paired S0 control). Compute per-run `(X = TV, Y = G_auc)`. Estimate `sd(Y)`, within-schedule `corr(X, Y)`, and the S0 noise floor.
- **Power:** with pilot numbers, compute runs-per-stratum to distinguish `r = 0.8` from `r = 0.5` (one-sided alpha = 0.05, power 0.8) via Fisher z: `n = 3 + ((z_.95 + z_.80)/(atanh r1 - atanh r0))^2` ~ 24 with the default targets; recompute from pilot-informed targets (`analysis/power.py`). If n <= 40/stratum, proceed; else escalate to the human (reduce strata or magnitudes grid).
- **Full run:** rho-stratification lives in **A0-with-harm**: attach a harm proxy to A0 by defining `Y_A0 = integral angle(v_t, v_0) dk` weighted... **No.** Keep harm real: full E4 runs on **two designs**: (a) A1, magnitude grid — schedules {S1, S2, S5} x magnitude {small, medium, large: 10deg/25deg/45deg; decay 0.995/0.985/0.97; swap cos 0.9/0.7/0.5} x n seeds (all high-rho; A1 battery spans). Regression of `G_auc` on `TV`, report r + CI, plus Spearman. (b) A0 support-stratified detection-vs-truth: schedules {S1, S1m, S1n} x magnitudes x n seeds; here Y = true drift size `||v_K - v_0||` and X = `TV`; demonstrate estimated-vs-true slope ~ rho (attenuation curve). Together: (a) shows measured drift predicts harm where measurable; (b) quantifies exactly how measurement degrades off-support. This pairing is the paper's core claim structure.
- **Acceptance:** pre-registered: in (a), pooled r >= 0.6 with 90% CI excluding 0.3. In (b), monotone attenuation of recovered-drift fraction in rho, with the rho->0 cell near zero.

### E5 — Matched-budget refresh baseline (the skeptic's null)

- **Setup:** A1, schedules {S1, S2, S5} x arms x 20 seeds. Total label budget per round fixed at `B = 100` pairs for every arm. Arms: (a) accumulate-all (default); (b) window: keep only last `W = 10` rounds of labels; (c) refresh: each round spend `f.B` fresh operational pairs and `(1-f).B` **re-labeling** the oldest retained pairs with the *current* teacher, `f = 0.7`, retention cap 10 rounds.
- **Outcome:** `G_auc` per arm; report Delta vs arm (a) with paired-seed CIs.
- **Acceptance:** none (descriptive) — this quantifies how much of drift harm the dumb baseline already removes at matched budget. Whatever the answer, it publishes; it is the comparison any future correction method must beat.

### E6 — Gradient-coupled drift (gated; build only after E1-E5 accepted)

- **Setup:** A1, 2x2: drift {gradient_coupled eta=2deg/round sign=+1, random-direction rotation magnitude-matched per round} x teacher persistence {single persistent teacher, churn every 8 rounds} x 20 seeds. Norm held constant.
- **Outcomes:** `G_auc`; estimator coverage trajectory; anchor detectability (`TV` recovered / true drift).
- **Hypotheses (pre-registered):** coupled x persistent shows the largest `G_auc` and the largest coverage degradation; churn attenuates coupled-drift harm more than random-drift harm (the rotation-resets-capture mechanism).
- **Acceptance:** report whichever way it falls; the differential is the adversarial signature either way.

---

## 5. Statistical appendix

- All CIs: seed-level nonparametric bootstrap (10k resamples) unless closed-form is exact.
- Equivalence tests: TOST at the stated bounds.
- Correlation comparisons: Fisher z. Worked default: distinguishing r=0.8 from r=0.5 -> n ~ 24/stratum.
- Paired designs (E4a, E5, E6) always use common-random-number S0 pairing; report paired differences.
- Multiple comparisons: within an experiment, Holm over its pre-registered test family; families are per-experiment, not global.

## 6. Unit tests (proposition-derived; write these FIRST)

1. **Prop 1:** arbitrary `s_t` trajectory (including huge jumps) leaves the exact label probability of every comparison unchanged; and `label_from_scores` across times does depend on `s_t - s_t'` (Cor 1).
2. **Prop 2:** `(beta, w)` and `(beta/c, cw)` produce identical label probabilities for all pairs.
3. **Prop 3:** scaling equivalence — label law under `(c.v, gaps G)` equals law under `(v, c.G)` (exact probability comparison on a fixed pair set).
4. **Split rule:** finite-difference `dp/deps` for norm perturbation is 0 at p=0.5 anchors and maximal near p=0.8237 (numeric check against `z sigma'(z)`); direction perturbation prefactor maximal at p=0.5.
5. **rho correctness:** constructed delta inside `rowspace(Phi_A)` -> rho=1 +/- 1e-9; delta in `S_nov` -> rho=0 +/- 1e-9; 50/50 mix -> 0.5.
6. **Estimator sanity:** on a linear-Gaussian reduction (replace sigma with identity link + Gaussian noise), EKF coverage = nominal within MC error; and on the logistic model with S0, 90% CI empirical coverage in [85, 95] over 200 short runs.
7. **Battery construction:** direction panel phat's in [0.45, 0.55]; norm panel in bands; rank(Phi_A) = 16 in A0; battery bytes identical across the run (frozen).
8. **Gold eval:** analytic check on a 2-state MDP.

## 7. Implementation order and definitions of done

| M | build | DoD |
|---|---|---|
| M0 | scaffold, config/seed/logging plumbing, tests 1-3 red-green | `pytest` green; one dummy run writes valid JSONL |
| M1 | C1 teacher/population + C2 drift + presets | tests 1-3, 5-precursor pass; drift trajectories plot correctly for all presets |
| M2 | C3 streams + C4 anchors + rho | tests 5, 7 pass |
| M3 | C5 probe + C6 diagnostics | E1 dry-run on 2 seeds produces sane curves |
| M4 | **E1 full + Figure 1** | acceptance criteria met; figure reviewed by human |
| M5 | C7 estimator + E3 | test 6 passes; E3 table produced; human review |
| M6 | C8 env/RM/policy + decomposition logging | test 8; A1/S0 gold-return curve rises and plateaus; over-optimization sanity: A1/S1 shows nonzero `G_auc` (if not, see Risks) |
| M7 | E2 + E4 pilot | E2 acceptance; pilot CSV + power output; **human review gate before full E4** |
| M8 | E4 full (a)+(b) | pre-registered analyses produced exactly as specified |
| M9 | E5 | arm comparison table + figure |
| — | E6 | only on explicit human go-ahead |

## 8. Risks and sanctioned knobs

- **A1 harm too small** (linear RM tracks drifted teacher too fast): sanctioned knobs, in order — halve `n_pairs` to 50; sharpen `tau_pol` to 0.02; add feature masking to the RM (drop 2 features -> `e_mis != 0` arm, log it as such). Do not invent other mechanisms.
- **Anchors saturating after large drift** (p -> 0/1, information dies): expected at 45deg jumps; report per-anchor `sigma'(zhat)` and flag panels with mean sigma' < 0.05; do not adapt the battery mid-run (freezing is the point). If E4-large cells are information-starved, that is a finding about instrument dynamic range — report it.
- **EKF divergence at jumps in blind mode:** cap innovation at 6 sigma per coordinate per update; if NEES > chi2_{0.999} for 3 consecutive rounds, log a divergence flag; do not silently reinitialize.
- **Degenerate D-optimal panel** (greedy picks near-duplicates): assert min pairwise angle between selected `dphi_j` > 5deg; if violated, resample pool with larger `N_cand`.

## 9. Explicit non-goals

No correction algorithms (E5's arms are data curation, not estimation-weighted training). No active learning. No neural reward models. No mid-run battery adaptation. No additional diagnostics beyond C6. No Tier-B. Any extension request goes to the human first.
