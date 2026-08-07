<!-- Transcribed 2026-08-05 from user-provided attachment; encoding (mojibake) fixed, content otherwise verbatim. This is the binding spec; supersedes v0.1. -->

# Tier-A Experimental Design: Grader Drift in Preference Learning
## Executable specification for a coding agent — v0.2

*v0.2 — August 2026. Supersedes v0.1 entirely. Companion: `grader_drift_identifiability_note.md` (v0.2) — "the note", §-references point there. All changes are logged in `AMENDMENTS.md`; every amendment predates confirmatory runs. **Any run executed under v0.1 definitions is void and must be re-run.***

---

## 0. Purpose and scope

Build a simulation suite that (a) demonstrates the operational-stream attribution failure (§4), (b) validates the anchor instrument and its split rule (§5), (c) validates a state-space drift estimator with calibrated uncertainty, (d) runs the headline analysis: how well do anchor-derived readouts predict harm — via a predictor hierarchy, under two harm references, with on-support attenuation quantified — and (e) establishes the matched-budget refresh baseline under both harm references.

Two rungs:

- **A0 (measurement rung):** no RL. Items are feature vectors; "policy improvement" is a shrinking pair-gap distribution. Isolates instrument + estimator questions.
- **A1 (loop rung):** tabular PbRL on a gridworld. Real closed loop with named criteria; provides the harm outcomes and criterion-level attribution.

**Harm is reference-dependent and both references are logged everywhere** (§C8): the frozen reference w* (corruption model: movement away from the initial standard is damage) and the current standard w_t (renegotiation model: staleness relative to today's rater is damage).

**Out of scope:** Tier-B, drift-correction algorithms, LLM judges, human data, active query selection, predictor tiers beyond H0–H3. Do not add features beyond this spec.

---

## 1. Conventions

- Python 3.11; `numpy scipy matplotlib pandas pytest pyyaml`. CPU only. A0 run < 1 min; A1 run < 10 min.
- Repo layout as v0.1 (`src/graderdrift/{teacher,drift,streams,anchors,probe,diagnostics,estimator}.py`, `envs/grid.py`, `rm.py`, `policy.py`, `experiments/`, `analysis/`, `configs/`, `tests/`, `runs/`), plus `AMENDMENTS.md` at repo root — a dated log (date, source, change, which pre-registered quantity it altered). It ships with the paper.
- **Notation (binding):** `vhat` means the unit direction `v/||v||` and nothing else. The filter posterior mean is **`m_k`** (never `vhat`).
- Seeding: one master seed spawns named `numpy.random.Generator` children (`teacher, drift, stream, labels, probe, policy, rm`). Paired runs (see C8) share all streams except where stated.
- Logging: one JSONL row per round per run; schema (superset, unused = null):

```
{round, seed, schedule,
 v_true, m_k, cov_diag, loglik,
 anchor_p_hat: [..], D_dir, D_norm, D_crit: {contrast: value},
 rho_step_real, rho_cum_real, eff_drift_mag,
 op_agree, op_flip, op_temp_inv_std, op_temp_inv_raw,
 gold_wstar, gold_wt, regret_wt, proxy_return, e_shift, e_epi,
 n_labels_cum, event_flags: [..]}
```

---

## 2. Core math the code implements

- `u_t(x) = w_t.phi(x) + s_t`; labels `P_t(x>y) = sigma(v_t.dphi)`, `v_t = beta_t w_t`, `sigma(z) = 1/(1+e^{-z})`, `sigma' = sigma(1-sigma)`.
- Norm-sensitivity optimum: `|z sigma'(z)|` max at `|z*| = 1.5434`, `p* = 0.8237`.
- On-support fraction: `P_A` = projector onto `rowspace(Phi_A)` (SVD, tol 1e-10). `rho_step = ||P_A delta_k||^2/||delta_k||^2` for `delta_k = v_k - v_{k-1}`; `rho_cum` with `delta = v_k - v_0`. **Always log realized rho; targets are labels only.**
- Recovered-norm identity: `||P_A delta|| = sqrt(rho).||delta||`. Attenuation curves are in **sqrt(rho)**: an ideal estimator at rho = 0.5 recovers 0.72 of drift magnitude, not 0.5.
- Rotation floor: rotating `vhat` by theta about an axis in the battery's null space still realizes `rho = (1-cos theta)/2` (0.0076 / 0.0468 / 0.1464 at 10deg/25deg/45deg), because the increment has a component along `vhat` itself. **rho = 0 exactly requires a pure translation into the null space** (`novel_translation` event), not a rotation.
- Norm-jump prediction on an anchor: `Delta z_j = (factor - 1).z_j(0)` (a logit difference; not `log(factor)`).

---

## 3. Components

### C1. Teacher and population (`teacher.py`)

As v0.1: `Teacher(w, s, beta)`; `v() = beta.w`; `label(dphi, rng)`; `label_from_scores(...)` for the Prop-1/Cor-1 tests only. `Population` of `M` teachers; per-rater severity cancellation by construction. Defaults A0: `d = 24`, `S_hist = span(e_1..e_16)`, `S_nov = span(e_17..e_24)`, `w_0` random unit **in S_hist**, `beta_0 = 1.0`, `s_0 = 0`. A1: see C8 (`beta_0 = 3.0`, `w_0 = w*`, `||v_0|| ~ 4.44`).

### C2. Drift generator (`drift.py`)

Event calendar + per-round processes, applied once per round. Event types:

- `direction_jump(angle_deg, rho_target)` — axis `u = normalize(sqrt(rho*).u_h + sqrt(1-rho*).u_n)`, `u_h in S_hist ∩ vhat-perp`, `u_n in S_nov`; `vhat <- cos(theta) vhat + sin(theta) u`. Log realized rho (floor per §2 applies).
- `novel_translation(magnitude)` — `v <- v + m.u_nov`, `u_nov` random unit in `S_nov`. Realizes `rho_step = 0` exactly. Semantics: the rater acquires a new criterion.
- `norm_jump(factor)` / `norm_decay(from, to, factor)` — multiply `||v||`.
- `family_swap(cos_target, norm_factor)` — resample direction at given cosine to old (within `S_hist` unless `rho_target` given); rescale norm. Judge/version swap.
- `mixture_redraw(fraction)` — population mode, as v0.1.
- Per-round diffusion (`dir_diffusion_sigma`, `norm_diffusion_sigma`) as v0.1.
- `gradient_coupled(eta_deg, sign)` — E6 only, as v0.1; axes projected into `Shat_ach` in A1.
- **A1 named-criterion events** (operate on `w` in feature coordinates; do not renormalize; log the incidental norm change): `criterion_scale(feature, factor)`; `tradeoff_rotation(feature_a, feature_b, angle_deg)` — Gram-Schmidt the two projected axes `P_ach e_a, P_ach e_b`, rotate `v`'s components in that plane, norm-in-plane preserved; `rubric_edit(list of (feature, factor))`.
- **A1 constraint:** all *random* drift axes are drawn inside `Shat_ach` (C8), so nominal = effective. Named-criterion events act naturally on `w`; the effective (projected) magnitude `||P_ach delta||` is logged and used in analyses.

**Schedule presets** (K = 60 A0, 40 A1; A1 event time t=20 unless noted):

| id | rung | content |
|---|---|---|
| S0 | both | stationary |
| S1 | both | direction_jump 25deg @30 (A0) / @20 (A1), rho*=1.0 |
| S1m / S1n | A0 | as S1 with rho*=0.5 / rho*=0.0-labeled (floor documented) |
| NT | A0 | novel_translation @30, magnitudes matched to ||delta|| of 10deg/25deg/45deg jumps |
| S2 | both | norm_decay 0.985/round from t=20 |
| S2j | A0 | norm_jump x0.6 @30 (E2 factorial) |
| S1p | A0 | pure rotation 25deg @30, rho*=1.0, norm untouched (E2 factorial) |
| S3 | both | mixture_redraw 30% every 10 rounds (M=10) |
| S4 | A0 | dir_diffusion 0.02/round |
| S5 | both | family_swap cos 0.7, x1.3 @30/@20 |
| S7a / S7b | A1 | teacher **initialized 25deg off w*** (rotation within Shat_ach); then 1deg/round rotation toward (a) / away (b) from w*; norm fixed. Matched |eta|. |
| S8t / S8m / S8b | A1 | direction_jump 25deg @20 about `u = normalize(P_{vhat-perp} e_i)` for the top / 3rd / 6th eigenvector of the achievable-gap covariance (within Shat_ach); norm fixed |
| SC1 | A1 | criterion_scale(lava, x1.5) @20 |
| SC2 | A1 | tradeoff_rotation(goal_proximity, water, 20deg) @20 |
| SC3 | A1 | rubric_edit[(lava,x1.3),(water,x1.5),(wall_bump,x2.0)] @20 |
| SN | A1 | appendix: translation purely in null(Shat_ach) — predicts identical label probabilities (identical realized labels under CRN) and zero harm |

### C3. A0 item streams (`streams.py`)

As v0.1: `dphi ~ N(0, c_k^2 Sigma_k)`, `c_k = 2.0.0.95^k`; `Sigma_k` identity on `S_hist`, novel-dim variance ramping from round 20 to `0.5.c_k^2` by round 40; `n_pairs = 100`; duplicate fraction `q_dup = 0.15`.

### C4. Anchor battery (`anchors.py`)

Built once at round 0, then **frozen**; battery bytes identical across the run.

- **A0:** candidate pool `N_cand = 5000` from the round-0 historical distribution. Direction panel `m_dir = 16` (|p-0.5| <= 0.05, greedy D-optimal on `sigma'(z)dphi dphi^T`); norm panel `m_norm = 8` (p in [0.78,0.86] union [0.14,0.22], |z| nearest 1.5434, 4/4 sign-balanced). Assert `rank(Phi_A) = 16`; min pairwise angle between selected gaps > 5deg.
- **A1:** candidate pool = segment pairs from the round-0 reference mixture. Direction panel `m_dir = 12`, norm panel `m_norm = 6` (same rules against `v_0`), **plus criterion-contrastive panel**: for each contrast `c in {lava-vs-grass, water-vs-grass, wall_bump, stay_action, goal_proximity}`, select `m_c = 4` pairs maximizing purity `|dphi_c|/||dphi||` subject to `p_j in [0.25, 0.75]`; log per-anchor purity; flag any contrast with mean purity < 0.6 (sanctioned knob: widen band to [0.20, 0.80], then reduce `m_c`). Assert `rank(Phi_A) <= 6` and report it.

### C5. Probing (`probe.py`)

Every round, post-drift, pre-training-use: `n_probe = 50` labels per anchor from the current teacher state; anchors never enter RM data. Readouts: `D_dir`, `D_norm` as v0.1; per-contrast `D_c(t)` = mean signed logit change over the contrast's anchors, sign-aligned with the contrast direction.

### C6. Operational diagnostics (`diagnostics.py`)

From the operational stream only:

- `op_agree`, `op_flip` as v0.1.
- `op_temp_inv_std` (**primary**): standardize margins within round — `m_i = (what_k.dphi_i)/std_round(what_k.dphi)` — fit scalar tau by 1-D MLE of `sigma(tau m_i (2y_i-1))`; report `1/tau_hat`. Under S0 this rises ~ `1/c_k`: the aliasing statistic.
- `op_temp_inv_raw` (secondary): same fit on unstandardized margins. With observed gaps and a fitted direction, tau is identified ~ `||v||` and stays flat under S0. **Log both**; the contrast is itself a finding — normalization is exactly what re-introduces the JND confound, and normalized statistics are what dashboards typically monitor.

### C7. State-space estimator (`estimator.py`)

Iterated EKF/Laplace, state `v in R^d`, posterior mean **`m_k`**, covariance `Sigma_k`.

- Dynamics `v_{k+1} = v_k + q_k`. Modes: `known_calendar` (**headline**; the calendar *is* known in simulation — instrument-validation framing): `Q_k = sigma_diff^2 I + 1[event].sigma_jump^2 I`. `blind` (**robustness appendix only**): `Q_k = sigma_blind^2 I`.
- **Scales are relative:** `sigma_diff = 0.01.||m_0||`, `sigma_jump = 0.5.||m_0||`, `sigma_blind = 0.05.||m_0||`, with `||m_0||` from the round-0 MAP (~1 in A0, ~4.4 in A1).
- Measurements: anchor binomials through sigma; 3 Newton iterations; innovation cap 6 sigma/coordinate; NEES divergence flag as v0.1.
- **Primary drift statistic:** `TV = Sum_k ||m_k - m_{k-1}||` over the **full vector**, minus the mode- and rung-specific S0 noise floor (`analysis/noise_floor.json`, computed from E1 S0 runs). Regression test: an S2 run's TV must exceed the floor — a direction-only implementation fails this by construction.
- Derived intervals (norm, angle, `P_A v` coordinates) by 2000 Gaussian draws, as v0.1.

### C8. A1 PbRL loop

- **Env:** 10x10 grid, 5 actions, slip 0.05, `H = 12`. Features `phi(s) in R^8`: `[on_goal_cell (per-step, non-absorbing), -normalized_manhattan_to_goal, lava, water, grass, wall_bump, stay_action, 1]`. Cell types partition the grid, so segment gaps annihilate {constant, uniform cell-type shift}: **achievable-gap rank = 6**.
- **Round 0:** collect the reference pool; SVD the gap matrix (threshold 1e-8.sigma_max); assert rank 6; store `P_ach` and the gap covariance eigenvectors (for S8). Log `eff_drift_mag = ||P_ach(v_k - v_0)||` every round.
- **True reward** `w* = [+1.0, +0.3, -1.0, -0.3, 0, -0.1, -0.02, -0.02]`; `||v_0|| = 3||w*|| ~ 4.44`.
- **Round loop** as v0.1 (200 segments 80/20 policy/uniform; 100 random pairs; RM = L2 logistic on accumulated pairs; soft VI, `tau_pol = 0.05`; probe; log).
- **Dual-reference harm, logged every round:** `gold_wstar = J_{w*}(pi_k)` (exact evaluation, gamma = 0.97); `gold_wt = J_{w_t}(pi_k)`; `regret_wt = J_{w_t}(pi*_{w_t}) - J_{w_t}(pi_k)` (the `pi*_{w_t}` solve is already needed for `e_shift`). Decomposition logs `e_shift`, `e_epi` as v0.1.
- **Pairing:** every drifted run has a stationary twin at the same seed with common random numbers. The twin **shares the initialization**: for S7a/b the control is stationary-at-25deg-off (call it S7-null), not S0. Frozen-reference harm: `G_auc = mean_{k >= first_event}[gold_wstar^{ctrl}(k) - gold_wstar^{sched}(k)]`; current-standard harm: `R_auc = mean_{k >= first_event} regret_wt(k)` (no twin needed; report the twin's `R_auc` as its floor).

---

## 4. Experiments

Run order: E1 -> E2 -> E3 -> E4 pilot -> power -> E4 full -> E5. E6 gated.

### E1 — Stationarity null (Figure 1)

A0/S0 x20 seeds; A1/S0 x10.
**Predictions:** `op_agree` falls; `op_temp_inv_std` rises ~ `1/c_k`; `op_temp_inv_raw` flat; both anchor panels flat.
**Analysis:** per-run OLS slopes, rounds 5-60. Equivalence bounds (TOST, 90% CI inside): anchor summaries |slope| < 0.001/round; `1/tau_hat_raw` |slope| < 0.002/round.
**Acceptance:** std-temperature and agreement slopes significant in predicted directions; raw-temperature and anchors pass equivalence. Figure 1: operational stats (three lines, left) vs anchor panels (right).

### E2 — Split rule (the exact-zero cell)

A0, {S1p, S2j} x {`D_dir`, `D_norm`}, 20 seeds/cell.
**Predictions:** (i) `D_dir` under S2j = 0 in expectation — equivalence vs pure sampling noise (noise distribution simulated by relabeling the round-30 state); (ii) `D_dir` under S1p large; (iii) `D_norm` under S2j matches the **per-anchor** prediction `Delta z_j = -0.4.z_j(0)` within CI, using each anchor's round-0 probe MLE for `z_j(0)`.
**Acceptance:** (i) passes equivalence; (iii) per-anchor CIs cover the prediction for >= 7/8 anchors; cell ordering as predicted.

### E3 — Estimator calibration

A0, {S0,S1,S2,S3,S4,S5} x 20 seeds; both modes run, `known_calendar` gated, `blind` reported in a robustness appendix (expected failure at jump rounds — the Glicko point, a finding).
**Acceptance:** `known_calendar` coverage of nominal-90% CIs for `P_A v` in [85, 95] pooled and per-schedule. Full-`v` coverage reported, not gated.

### E4 — Headline: what predicts harm, and how measurement attenuates

**(a) A1 harm hierarchy.** Schedules: {S1, S2, S5} x magnitudes {10deg/25deg/45deg; decay 0.995/0.985/0.97; swap cos 0.9/0.7/0.5} (9) + S7a, S7b (2) + S8t/m/b (3) + SC1-3 (3) = **17 schedules** x n seeds, each with its paired control.
Outcome: `Y = G_auc` (primary, frozen reference); `Y' = R_auc` (secondary, current standard).
**Predictor hierarchy** (all from estimator outputs `m_k`, `known_calendar`):
- H0: `TV` alone (expected to fail on the S2/S7 mix — that is the point; reported, not gated).
- H1: H0 + norm/direction split — total log-norm change `Sum|Delta log||m_k|||` and angular path length `Sum angle(m_k, m_{k-1})`.
- H2 (reference-aware tier; reference available in sim, not in deployment — label it as such): H1 + signed alignment change `Delta align = angle(m_0, w*) - angle(m_K, w*)`.
- H3 (oracle ceiling): true `e_shift`, `e_epi` integrals.
**Analysis:** adjusted R^2 per tier with seed-level bootstrap CIs; report incremental R^2. Named contrasts: S7a vs S7b — `|TV_a - TV_b|` CI includes 0 **and** `G(S7a) < 0 < G(S7b)` with paired CIs excluding 0; S8 — `G(S8t) > G(S8m) > G(S8b)`, top-vs-bottom significant.
**Pre-registered gates:** H2 adjusted R^2 >= 0.5 **and** >= 0.8 x H3's; both S7 and S8 contrasts as stated.
**(b) A0 attenuation.** Cells: rotations {10deg, 25deg, 45deg} x rho* {1.0, 0.5, 0.0-labeled} + NT at the three matched magnitudes. X-axis: realized `sqrt(rho_cum)`. Y: recovered fraction `||m_K - m_0|| / ||v_K - v_0||`.
**Acceptance:** slope of recovered-fraction on sqrt(rho) has CI including 1; NT cells within noise floor + 2.se of zero; rotation floors land on the curve (they are data points, not failures).
**Pilot & power:** pilot = 5 schedules {S1-25deg, S2-mid, S5-mid, S7a, S8t} x 10 seeds — variance estimates, noise floors, bootstrap CI width on H2 R^2. Escalation rule: if the H2 R^2 CI width > 0.2, double seeds (cap 40); if still > 0.2, stop and escalate to the human. Fisher-z machinery retained for the S7 TV-equality contrast (default worked n ~ 24/cell for r-contrasts stands as the reference calculation).

### E5 — Matched-budget refresh, dual-reference sign flip

A1, arms {(a) accumulate-all, (b) window W=10, (c) refresh f=0.7} x {S1, S2, S5} x 20 seeds, budget B = 100 pairs/round all arms.
**Outcomes:** `G_auc` and `R_auc` per arm.
**Headline table:** arm ranking under each reference. **Prediction:** ranking flips — (a) best under frozen reference (old labels are the clean ones), (c) best under current standard (tracking the live rater is the point). Descriptive; publishes either way; this is the corruption-vs-renegotiation dichotomy made quantitative.

### E6 — Gradient-coupled drift (gated; explicit human go-ahead required)

As v0.1, with axes in `Shat_ach`, norm constant, scales per C7. Hypotheses unchanged.

### Criterion-attribution readout (piggybacks on E4a, no new runs)

On SC1-SC3 runs: confusion matrix — rows = drifted criterion (truth), columns = argmax `|D_c|` post-event — reported **with** mean contrast purity per row. Acceptance (soft): diagonal-dominant on SC1/SC2 at purity >= 0.6; SC3 reported as the multi-criterion case.

---

## 5. Statistical appendix

Seed-level bootstrap (10k) for all CIs incl. adjusted-R^2; TOST at stated bounds; Fisher z for correlation contrasts; paired designs always report paired differences; Holm within each experiment's pre-registered family.

## 6. Unit tests (write first; proposition- and amendment-derived)

1. **Prop 1 / Cor 1:** arbitrary `s_t` drift leaves comparison probabilities bit-identical; `label_from_scores` across times depends on `s_t - s_{t'}`.
2. **Prop 2:** `(beta, w)` vs `(beta/c, cw)` identical.
3. **Prop 3:** scaling equivalence `(c.v, G)` == `(v, c.G)` exact.
4. **Split rule numerics:** norm-sensitivity 0 at p=0.5, max near p=0.8237; direction prefactor max at p=0.5.
5. **rho mechanics:** delta in rowspace -> rho=1; pure-novel-axis **rotation** by theta -> rho=(1-cos theta)/2 +/- 1e-6; `novel_translation` -> rho=0 exact; `||P_A delta|| = sqrt(rho)||delta||` identity.
6. **Estimator:** linear-Gaussian reduction coverage nominal; logistic S0 coverage in [85,95] over 200 short runs; **S2 TV regression test** — full-vector TV > noise floor (direction-only TV fails).
7. **Battery:** panel bands respected; A0 rank 16; A1 rank <= 6; frozen bytes; min pairwise angle > 5deg; contrastive purity logged and >= threshold or knob-flagged.
8. **Temperature aliasing:** S0 with shrinking `c_k` -> `1/tau_hat_std` rises ~ `1/c_k` (regression slope on 1/c_k, R^2 > 0.9), `tau_hat_raw` flat within tolerance.
9. **Norm-jump exactness:** exact-probability `Delta z_j = (f-1)z_j(0)` to machine precision.
10. **A1 rank:** gap-matrix rank 6; null space = span{constant, uniform cell-type shift}; SN schedule under CRN yields identical realized labels and zero harm.
11. **Gold eval:** analytic 2-state MDP check.

## 7. Implementation order and definitions of done

| M | build | DoD |
|---|---|---|
| M0 | scaffold, plumbing, tests 1-3 | pytest green; dummy run writes valid JSONL |
| M1 | C1 + C2 incl. novel_translation, named-criterion events, all presets | tests 1-3, 5, 9 pass; trajectory plots for every preset |
| M2 | C3 + C4 + rho | tests 5, 7 |
| M3 | C5 + C6 (both temperature variants) | E1 dry-run sane |
| M4 | **E1 full + Figure 1** | acceptance incl. test 8; **human review gate** |
| M5 | C7 (m_k, full-vector TV, relative scales, mode gating) + E3 | tests 6; E3 table; human review |
| M6 | C8 (per-step goal, Shat_ach, dual-reference logs, contrastive panel) | tests 10, 11; A1/S0 gold curve rises; A1/S1 shows `G_auc > 0` (else Risks) |
| M7 | E2 (per-anchor prediction) + E4 pilot | E2 acceptance; pilot CSV + power output; **human review gate before full E4** |
| M8 | E4 full (a)+(b) + criterion confusion matrix | pre-registered analyses exactly as specified |
| M9 | E5 dual-reference table | table + figure |
| — | E6 | human go-ahead only |

## 8. Risks and sanctioned knobs

- **A1 harm too small:** halve `n_pairs` -> 50; sharpen `tau_pol` -> 0.02; feature-mask RM (drop 2 features; log as the `e_mis != 0` arm). Nothing else.
- **Contrastive purity low:** widen p-band to [0.20, 0.80]; then reduce `m_c`. Always report purity.
- **Anchors saturating post-drift:** report per-anchor `sigma'(zhat)`; flag panels with mean sigma' < 0.05; never adapt the battery mid-run. Information starvation at 45deg cells is a dynamic-range finding.
- **S7 baseline:** the 25deg-off initialization changes baseline harm; the S7-null twin (stationary at the same init) is mandatory — comparing S7 runs against S0 is a bug.
- **EKF at jumps in blind mode:** innovation cap + NEES flag; no silent re-init.

## 9. Explicit non-goals

No correction algorithms; no active learning; no neural RMs; no mid-run battery adaptation; no diagnostics beyond C6; no predictor tiers beyond H0-H3; no Tier-B. Extensions go to the human first.

## 10. Changelog (v0.1 -> v0.2)

Full entries with rationale in `AMENDMENTS.md`. Summary: (review fixes) standardized-vs-raw temperature pair and E1 rewrite; per-anchor norm-jump prediction `(f-1)z0`; sqrt(rho) attenuation; realized-rho axes + rotation floor + `novel_translation`; full-vector TV and `m_k` notation; A1 rank-6 identifiable subspace, per-step goal indicator, SN appendix schedule; known_calendar headline + relative noise scales. (Wave-1 amendments) dual-reference harm logging everywhere; S7 toward/away controls with shared-init twins; S8 behavioral-relevance schedules; E4 predictor hierarchy replacing pooled-r gate; E5 reframed as the sign-flip table; named-criterion schedules SC1-3; criterion-contrastive panel + confusion matrix + purity reporting. All v0.1 runs void.
