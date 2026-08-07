# Pre-registration amendments — Tier-A grader-drift suite

Amendments to `md_files/tier_a_experiment_spec.md` v0.1 and
`md_files/grader_drift_identifiability_note.md` v0.1, made **before any
confirmatory data was collected**. Recording them is pre-registration hygiene:
amendments made ahead of the data are legitimate; hiding them would not be. This
file ships with the paper.

Each entry: date · source · what changed · rationale · which pre-registered
number/definition it altered · gate before which it must be applied.

---

## Wave 1 — Pre-implementation audit (2026-08-05)

Source: pre-implementation audit of the v0.1 spec (5-lens review + direct
numerical verification). Seven items. All numeric claims below were checked in
numpy/scipy before acceptance.

### A1. `op_temp_inv` redefined on standardized margins  · **FATAL** · gate: before M4/M5
- **Was:** E1 pre-registers that `op_temp_inv` (1/τ̂ from a 1-D temperature MLE
  on operational labels, given the fitted direction and *observed* gaps) *rises*
  as the gap scale `c_k` shrinks under S0.
- **Problem:** with observed gaps and a jointly-fitted well-specified margin the
  temperature is *identified* — τ̂ is consistent for ‖v‖ regardless of `c_k`
  (verified: median τ̂ ≈ 1.0 across `c ∈ {2, 1, 0.5, 0.25}`; only its variance
  grows). The rising-temperature phenomenon requires the *scale-normalized*
  statistic practitioners actually monitor.
- **Change:** primary `op_temp_inv` fits τ on within-round model margins
  **divided by their round std**; then `1/τ̂ ∝ 1/c_k` and E1's prediction holds
  (verified: `1/τ̂` rises 0.50 → 0.96 → 2.06 → 3.72 as `c` halves). The **raw
  (unstandardized) variant is kept as a secondary logged column** — the pair is
  the finding: raw stays flat (jointly-fitted margin absorbs scale), standardized
  rises (normalization creates the aliasing).
- **Altered pre-registration:** E1 prediction & acceptance for `op_temp_inv`.
- **Note v0.2:** §4 sentence "any fitted Bradley-Terry temperature rises"
  amended to — *temperature fitted against scale-normalized scores rises; a
  jointly-fitted margin is calibrated and flat; the aliasing lives precisely in
  the normalized-scale statistics practitioners typically monitor.*
- **New unit test:** S0 with shrinking `c_k` → standardized `1/τ̂` rises ∝ `1/c_k`;
  raw τ̂ ≈ 1 flat.

### A2. E2(iii) norm-jump prediction: per-anchor `Δz`, not `log(0.6)`  · **MAJOR** · gate: before E2
- **Was:** E2 pre-registers the S2j (×0.6 norm jump) `D_norm` response matching
  `Δz = log(0.6) ≈ −0.511` per anchor.
- **Problem:** `D_norm` is a difference of *logits*; a norm jump `z → 0.6 z` gives
  `Δlogit = (0.6−1)·z_j(0)`, not `log(0.6)`. At panel center `|z₀|=1.5434` that is
  **−0.617**, and the panel spans `p ∈ [0.78, 0.86]` so a single value is only the
  center.
- **Change:** predict per-anchor `Δz_j = (factor−1)·z_j(0)` using each anchor's
  round-0 probe estimate of `z_j(0)`; CI check against the per-anchor values.
- **Altered pre-registration:** E2(iii) numeric target.
- **New unit test:** exact-probability computation of `Δz` under a norm jump
  matches `(f−1)·z₀` to machine precision.

### A3. E4(b) attenuation is `√ρ`, not `ρ`  · **MAJOR** · gate: before any E4(b) run
- **Was:** E4(b) pre-registers estimated-vs-true recovered-drift slope ≈ ρ.
- **Problem:** the estimator sees `P_A δ`, whose *norm* is `√ρ · ‖δ‖` because ρ is
  defined on *squared* norms. Recovered fraction at ρ=0.5 is ~0.71, not 0.5.
- **Change:** plot recovered fraction against realized **`√ρ`**; slope ≈ 1.
  Acceptance updated (ρ=0.5 cell expects ~0.71).
- **Altered pre-registration:** E4(b) prediction & acceptance.

### A4. Realized ρ floor for rotations + new `novel_translation` event  · **MAJOR** · gate: before E4(b) / M1 drift
- **Was:** S1n labeled "ρ*=0.0"; E4(b) acceptance "the ρ→0 cell near zero".
- **Problem:** a rotation increment `δ = (cosθ−1)v̂ + sinθ·u` always retains an
  on-support component along `v̂ ∈ S_hist` (battery rowspace): realized
  `ρ = (1−cosθ)/2` even at ρ*=0 → 0.008/0.047/0.146 at 10°/25°/45°. Combined with
  A3 the 45° "novel" cell still recovers ~38% of drift norm.
- **Change:** (i) x-axis is realized *logged* ρ everywhere; `rho_target` values are
  labels only. (ii) Add a `novel_translation` event `v ← v + m·u_nov` (`u_nov`
  unit in `S_nov`), which achieves `ρ_step = 0` **exactly** and is a better model
  of a judge swap that adds a rubric dimension ("rater acquires a new criterion").
  It takes over the ρ≈0 cells; **S1n demoted** to a documented mid-low-ρ point.
- **Altered pre-registration:** S1n role; E4(b) ρ≈0 cell definition.
- **New unit test:** pure-novel-axis rotation realizes `ρ = (1−cosθ)/2 ±` tol;
  `novel_translation` realizes `ρ = 0` exactly.

### A5. TV notation collision resolved; TV is on the full vector  · **MAJOR** · gate: before M4/M5
- **Was:** C7 writes `TV = Σ‖v̂_k − v̂_{k−1}‖`, but `v̂` elsewhere denotes the
  *unit direction* `v/‖v‖`.
- **Problem:** a direction-only TV is blind to pure norm drift (S2, which is in
  E4(a)'s schedule set), so it would tank the pre-registered pooled `r ≥ 0.6`;
  E4(b)'s `Y = ‖v_K − v₀‖` is full-vector too.
- **Change:** rename the filter posterior mean to **`m_k`** throughout C7 (reserve
  `v̂` for unit direction); define `TV = Σ_k ‖m_k − m_{k−1}‖` on the **full
  estimated vector**, minus the S0 noise floor.
- **Altered pre-registration:** definition of the headline predictor `TV`.
- **New regression test:** an S2 (pure norm-decay) run must produce TV
  significantly above the noise floor — a direction-only implementation fails
  this by construction.

### A6. A1 feature rank deficiency (6/8) handled explicitly  · **MAJOR** · gate: before M6
- **Was:** A1 features `[goal_reached, −norm_manhattan, lava, water, grass,
  wall_bump, stay, 1]` summed over fixed `H=12` segments; drift in "random planes
  within R^8"; C4 asserts battery rank.
- **Problem:** the constant feature and the cell-type sum are identically zero in
  every segment gap (verified: gap-matrix rank = 6/8, null space =
  {constant feature, uniform cell-type shift}). So "random-plane" drift is partly
  invisible *and harmless* (a constant-per-step reward shift does not move the
  optimal policy), diluting E4(a)'s X–Y relation and logging ρ<1 despite full
  effective coverage.
- **Change:** (i) pin `goal_reached` as a **per-step, non-absorbing** "on goal
  cell" indicator; (ii) at round 0 estimate the achievable-gap rowspace
  empirically (SVD of the reference segment-pool gap matrix, threshold 1e−8) and
  **restrict all A1 drift** — jump axes and diffusion — to it; (iii) log
  **effective drift magnitude** = projection onto that subspace; (iv) update C4's
  A1 rank assertion to expect **6**. Plus a deliberate appendix schedule with
  drift *purely in the null space* (predicting bit-identical labels, zero harm) —
  invisible-and-harmless drift demonstrated, not accidentally contaminating the
  grid.
- **Altered pre-registration:** C4 A1 rank assertion; A1 drift-plane definition;
  effective vs nominal drift magnitude in E4(a).
- **New unit test:** gap-matrix rank = 6; null space spanned by
  {constant feature, uniform cell-type shift}.

### A7. Headline TV from `known_calendar`; process-noise scales as fractions of ‖m₀‖  · **MAJOR** · gate: before E3
- **Was:** E4 does not state which estimator mode feeds TV; `σ_diff=0.01`,
  `σ_jump=0.5`, `σ_blind=0.05` are absolute (tuned at A0 ‖v‖=1).
- **Problem:** absolute scales are mismatched at A1's ‖v₀‖≈4.4; and the headline
  predictor's mode was unspecified.
- **Change:** headline `TV` comes from **`known_calendar`** mode (the calendar is
  known in simulation; instrument-validation is the honest framing); **`blind`
  mode moves to a robustness appendix**. All process-noise scales become fractions
  of ‖m₀‖: `σ_diff = 0.01·‖m₀‖`, `σ_jump = 0.5·‖m₀‖`, `σ_blind = 0.05·‖m₀‖`
  (correct at both A0 ‖v‖=1 and A1 ≈4.4 without per-rung tuning).
- **Altered pre-registration:** E4 headline-predictor mode; C7 noise scales.

### Process
- Sequence by gate: A1 & A5 before M4/M5 sign-off; A7 before E3; A2 before E2;
  A3/A4 before any E4(b) run; A6 before M6.
- Anything already executed under old definitions (E1 dry runs, pilots) is re-run
  — all cheap at Tier A.
- Downgraded during verification (kept here for honesty): the earlier worry that
  `blind` mode "cannot absorb" an A1 jump in 10 rounds (it recovers ~87% in a
  well-conditioned battery — the real issue is scale, folded into A7), and that
  S3's population mixture biases coverage (Jensen bias ≈ −0.001 at κ=20 /
  LogNormal(0,0.1) — negligible; only `v_true := mixture mean` needs defining).

---

## Wave 2 — Consolidated spec v0.2 (2026-08-05)

Source: lead-authored consolidation. Folds Wave 1 (the seven audit fixes above)
together with five families from the earlier design discussion into a single
coherent diff. Binding docs: `md_files/tier_a_experiment_spec_v0_2.md` and
`grader_drift_identifiability_note_v0_2.md`. **All v0.1 runs are void.** Wave-1
items A1–A7 are all reflected in v0.2 (op_temp_inv std/raw + E1 rewrite; E2
per-anchor `Δz`; √ρ attenuation; `novel_translation` + realized-ρ axes + S1n
demotion + NT schedule; `m_k` + full-vector TV + S2 regression test; A1 rank-6
`Ŝ_ach` + per-step goal + SN schedule; `known_calendar` headline + ‖m₀‖-relative
scales). The five new families:

### B1. Dual-reference harm logging everywhere  · gate: M6
- **Change:** log both harm references every round — `gold_wstar = J_{w*}(π_k)`
  (frozen reference; corruption model: movement away from the initial standard is
  damage) and `gold_wt = J_{w_t}(π_k)` + `regret_wt = J_{w_t}(π*_{w_t}) − J_{w_t}(π_k)`
  (current standard; renegotiation model: staleness relative to today's rater is
  damage). Harm outcomes: `G_auc` (frozen) and `R_auc` (current). The `π*_{w_t}`
  solve is already required for `e_shift`, so no extra machinery.
- **Altered pre-registration:** replaces the single `G_auc` outcome of v0.1 E4/E5
  with a two-reference outcome; the corruption-vs-renegotiation distinction is now
  a measured quantity, not a framing choice.

### B2. S7 toward/away controls with shared-init twins  · gate: M6/E4
- **Change:** new A1 schedules S7a/S7b — teacher initialized 25° off w* (rotation
  within `Ŝ_ach`), then 1°/round rotation toward (a) / away (b) from w*, norm
  fixed, matched |η|. **Mandatory control is the S7-null twin** (stationary at the
  same 25°-off init), NOT S0 — comparing S7 against S0 is a bug (Risks §).
- **Purpose / prediction:** `|TV_a − TV_b|` CI includes 0 (equal-magnitude drift)
  **while** `G(S7a) < 0 < G(S7b)` — beneficial vs harmful drift of identical size.
  This is the case that defeats a magnitude-only predictor and motivates the
  reference-aware H2 tier.
- **Altered pre-registration:** new schedules + the pre-registered S7 contrast in
  E4(a).

### B3. S8 behavioral-relevance eigen-direction schedules  · gate: M6/E4
- **Change:** new A1 schedules S8t/S8m/S8b — 25° `direction_jump` about the top /
  3rd / 6th eigenvector of the achievable-gap covariance (within `Ŝ_ach`, rank 6),
  norm fixed.
- **Prediction:** `G(S8t) > G(S8m) > G(S8b)`, top-vs-bottom significant — drift
  along behaviorally high-variance directions causes more harm than drift along
  low-variance ones, at fixed drift magnitude.
- **Altered pre-registration:** new schedules + the pre-registered S8 monotonicity
  contrast in E4(a).

### B4. E4 predictor hierarchy H0–H3 replaces the pooled-r gate  · gate: E4
- **Change:** v0.1's "pooled r ≥ 0.6 of G_auc on TV" is replaced by a nested
  predictor hierarchy from estimator outputs `m_k` (`known_calendar`): H0 = `TV`
  alone (expected to fail on the S2/S7 mix — reported, not gated); H1 = + norm/
  direction split; H2 (reference-aware, sim-only) = + signed alignment change
  `Δalign = angle(m_0,w*) − angle(m_K,w*)`; H3 = oracle ceiling (true `e_shift`,
  `e_epi`). Metric: adjusted R² per tier with seed-level bootstrap CIs +
  incremental R². **Gate:** H2 adjusted R² ≥ 0.5 AND ≥ 0.8 × H3.
- **Altered pre-registration:** the E4(a) acceptance criterion entirely. Rationale:
  a single correlation can't express "magnitude alone is insufficient; you need
  the reference" — which is the S7 finding. The hierarchy makes that claim
  testable as incremental R².

### B5. Criterion-contrastive panel + confusion matrix + purity  · gate: M6/E4
- **Change:** A1 battery gains a criterion-contrastive panel (`m_c = 4` per
  contrast over {lava-vs-grass, water-vs-grass, wall_bump, stay_action,
  goal_proximity}, purity `|Δφ_c|/‖Δφ‖`, band `p ∈ [0.25,0.75]`); per-contrast
  readout `D_c`; named-criterion schedules SC1 (criterion_scale lava×1.5), SC2
  (tradeoff_rotation goal_proximity↔water 20°), SC3 (rubric_edit multi). Analysis:
  a criterion-attribution confusion matrix (truth = drifted criterion, prediction
  = argmax `|D_c|`), reported **with** mean per-row purity. Soft acceptance:
  diagonal-dominant on SC1/SC2 at purity ≥ 0.6.
- **Note:** contrasts are *differences* by design — individual cell-type levels are
  not separately identifiable (they lie partly in `null(Ŝ_ach)` = {constant,
  uniform cell-type shift}), so only lava-vs-grass / water-vs-grass etc. are
  meaningful. Consistent with A6.
- **Altered pre-registration:** adds the (soft) criterion-attribution readout to
  E4(a); no change to the harm gates.

### Version bump
Note → v0.2: §4 Remark 4.1 (which temperature statistics alias) and §5a two
quantitative footnotes (√ρ recovered-norm scaling; rotation ρ-floor).

## Wave 3 — Measurement-design amendments from the M4 (E1) build (2026-08-05)

Source: implementing + dry-running E1 (A0/S0, 20 seeds) surfaced that the
per-round temperature statistic is not estimable once gaps get small. All
pre-confirmatory (fixed before the confirmatory E-runs are declared accepted).
**Awaiting lead sign-off on C3 before E1 is marked accepted.**

### C1. Schema migration to v0.2 (applied) · gate: M0/all
- The M0 `ROW_FIELDS` logging schema was still v0.1; migrated to the v0.2 row
  schema (`m_k`, `op_temp_inv_std`/`_raw`, `gold_wstar`/`gold_wt`/`regret_wt`,
  `D_crit`, `rho_step_real`/`rho_cum_real`, `eff_drift_mag`, ...). Additive-only.

### C2. E1 temperature-statistic window narrowed 5-60 → 5-35 · gate: M4/E1
- **Problem:** `op_temp_inv_{std,raw}` are per-round 1-D temperature MLEs; once
  `c_k < ~0.33` (round ~35) the 100 operational labels are near-coin-flips and the
  MLE is unidentified — boundary-hit fraction (1/tau at the optimizer bound) jumps
  from 0% at r35 to 10% at r40, 33-50% by r50-60. This is Prop-3 information
  collapse taken to the limit, not an estimator bug.
- **Change:** analyze the temperature statistics over rounds **5-35** (c_k ≥ 0.33,
  zero boundary hits). Anchor summaries stay 5-60 (computable throughout). Figure 1
  shades the info-starved region. Worth one paper sentence: the operational
  detector doesn't just alias, it *dies* as gaps shrink — a stronger instrument-
  necessity argument.

### C3. E1 aggregation robustified + `op_temp_inv_raw` acceptance changed · gate: M4/E1 · SIGNED OFF (lead, 2026-08-05: "Do 1")
- **Problem:** the per-round temperature MLE is heavy-tailed (occasional boundary
  spikes even within 5-35), so the pre-registered mean-of-OLS-slopes is
  outlier-dominated (raw mean slope +0.19), and the pre-registered
  `|slope| < 0.002/round` equivalence bound for `op_temp_inv_raw` is unachievable
  for a per-round MLE at n_pairs=100 (even the robust median slope is +0.013).
- **Change (proposed):** (i) aggregate per-seed slopes by their **median** with a
  bootstrap CI of the median; (ii) replace `op_temp_inv_raw`'s absolute equivalence
  test with the **std-vs-raw divergence test** — the scientific claim (note §4) is
  that the two *diverge* under a constant rater. Acceptance: std median-slope CI
  lies entirely above raw's, and raw's level stays near ‖v‖=1.
- **Result (20 seeds):** op_agree PASS (median slope −0.0052, CI<0); op_temp_inv_std
  PASS (median +0.094, CI>0, level 0.58→3.11); anchors PASS equivalence (D_dir slope
  ~3e-5 CI⊂±0.001, level 0.07-0.08; D_norm slope ~7e-5, level −0.07..0.13);
  divergence PASS (std CI [0.071,0.155] vs raw [0.007,0.036], SEPARATED, 7.2× ratio;
  raw level 0.76-1.36 ≈ ‖v‖=1). Figure: `md_files/figures/fig1_stationarity_null.png`.
- **Altered pre-registration:** E1 analysis window + aggregator + the raw-temperature
  acceptance criterion. The anchor equivalence and directional (agree↓, std↑) tests
  are unchanged and pass.

### C4. Net-displacement drift statistics added beside path-length TV · gate: E4 (M7) · SIGNED OFF (lead, 2026-08-05: "Accept both")
- **Problem:** the pre-registered primary drift statistic TV = Σ‖m_k − m_{k−1}‖ is a
  path length, so it accumulates the filter's per-round estimation jitter: the A0
  S0 noise floor is TV = 2.16 ± 0.19 over 60 rounds (known_calendar, 20 seeds).
  Against that floor, an S1 25° jump (true ‖δ‖ = 0.43) is only a ~2σ exceedance,
  and slow-drift schedules (S2/S4/S7-style) drown entirely.
- **Change (proposed):** log the net (endpoint) statistics alongside TV —
  `net_disp = ‖m_K − m_0‖` (floor 0.12 ± 0.03 → the same S1 jump is a ~10σ
  exceedance), `net_lognorm = |log‖m_K‖ − log‖m_0‖|` (norm channel, floor
  0.063 ± 0.044), `net_angle = angle(m_K, m_0)` (direction channel, floor
  5.5° ± 1.2°). E4's H0/H1 predictor tiers use the net family; TV stays logged and
  reported. Endpoint forms cancel round-to-round noise; path forms do not.
- **Altered pre-registration:** the E4 predictor definitions (H0: net_disp instead
  of TV; H1: net_lognorm + net_angle instead of the path-length split). Noise-floor
  numbers above are from E3's S0 cell and stored in `runs/e3/noise_floor.json`.

### C5. `informed` calendar mode (generator-matched Q) · gate: M5/E3 · SIGNED OFF (lead, 2026-08-05: "Accept both") — E3 ACCEPTED
- **Problem:** the pre-registered known_calendar mode (fixed σ_diff=0.01‖m₀‖ +
  event-triggered σ_jump) STRUCTURALLY fails the per-schedule coverage gate: with
  one constant σ_diff, static/jumpy schedules over-cover (S0 0.967, S1 0.961,
  S5 0.956 — variance pumped into a non-moving state) while continuous-drift
  schedules under-cover (S2 0.819, S4 0.746, NEES/dof 2.31 — true per-round drift
  exceeds the floor). No global σ_diff fixes both directions; only the pooled gate
  passes (0.890).
- **Change (proposed):** define the calendar-aware mode as knowing the drift
  process **law** (types, times, magnitudes — as a platform knows its own
  rubric-edit/judge-swap/churn processes), not just event times: deterministic norm
  decay → known mean map m←factor·m; known diffusion → Q matched to its per-round
  covariance on the battery span; jumps → σ_jump at their round; mixture redraws →
  variance from the redraw law; numerical floor σ=3e-4‖m₀‖ (tuned: S0 coverage
  0.931 in a 6-seed sweep; 0.001 gave 0.951, marginally over). E3's gate applies to
  `informed`; fixed-Q known_calendar and blind are reported alongside (the fixed-Q
  failure pattern is itself a finding: NEES localizes exactly where process-noise
  magnitude is mismatched).
- **Status:** smoke-tested (S2: 0.819→0.998 w/ the norm band tracking the decay
  truth; S4: 0.746→0.943); full 20-seed table pending batch job 14518914. Both
  tables to be shown at the M5 review.

## Wave 4 — M6 (A1 loop) build findings (2026-08-05)

### D1. v-scaled optimal-policy references (applied) · gate: M6
- **Problem:** soft-VI reference policies computed on `w` are softer than policies
  computed on `v = β·w` (the τ_pol temperature is not scale-free), so
  `J_{w*}(π*_{w*,τ})` is NOT an upper bound for v-scaled learned policies —
  e_shift came out negative pre-drift.
- **Change:** all optimal-policy references are v-scaled: `π*` solves soft VI on
  `β·w` (teacher decisiveness included). Verified: e_shift ≡ 0 pre-drift, ≥ 0
  post-drift; regret_wt uses `J_{w_t}(π*_{v_t}) − J_{w_t}(π_k)`.

### D2. `mixture_redraw` norms made relative to population scale (applied) · gate: M6
- Absolute `LogNormal(0, 0.1)` rater norms only coincide with the A0 scale
  (‖v‖=1); at A1's ‖v‖≈4.4 they would collapse the population. New raters now draw
  ‖v‖ = (population mean norm)·LogNormal(0, 0.1). A0 behavior statistically
  unchanged; the E3 S3 cells (report-only) predate this fix and will be re-run in
  the next batch for cleanliness.

### D3. M6 DoD outcome: harm is thresholdy/axis-dependent; knobs tested and NOT adopted · gate: M6
- **Result ({S0,S1,S5} × seeds 0-2 × {default, τ_pol=0.02, n_pairs=50}):** S0 gold
  curve reaches ~23.2 within one round and plateaus (uniform baseline −8.5).
  S1 (25° rotation): G_auc ≈ 0 in ALL variants (means −0.14…+0.01) — sub-threshold.
  S5 (family swap): G_auc = 4–6 (≈20% of the achievable J range), R_auc ≈ 4–5.
- **Diagnosis (ceiling analysis, `J_{w*}(π*_{v0}) − J_{w*}(π*_{v_drift})`):** harm
  ceilings are strongly axis-dependent (25° draws range 0→2.2; 45° up to 35.7;
  SC1-3 ≈ 0 — reinforcing existing avoidances moves no policy). The sanctioned
  knobs cannot create harm the reward geometry does not contain; **defaults
  retained**, and the "A1/S1 G_auc > 0" DoD is discharged by S5 + the finding:
  *drift harm is thresholdy and direction-dependent, not proportional to
  magnitude* — precisely the E4 H0-fails / H2-needed story.
- **Altered pre-registration:** none yet (knob defaults unchanged); E4a Y-variance
  will be carried by the S5/S7/S8b/45° cells.

### D4. S8 behavioral-relevance hypothesis appears INVERTED · gate: decision needed at M7 · FLAG
- Ceiling analysis: S8t (top gap-variance eigenvector) ≈ 0 harm, S8m ≈ 0,
  **S8b (lowest in-S_ach variance) = 1.8**. Mechanism: rotating v toward
  high-variance axes re-weights criteria the current behavior already
  discriminates (optimal behavior unchanged — goal-seeking still dominates);
  rotating onto LOW-variance axes (stay/bump-like) rewards behaviorally-novel
  criteria and drags the policy into new behavior. The pre-registered B3 gate
  `G(S8t) > G(S8m) > G(S8b)` would fail in reverse order. Note the flip side:
  low-gap-variance directions are also where the operational stream carries the
  least signal — the dangerous drift is the hard-to-see drift, which strengthens
  the anchor/contrast-panel story. **Lead decision at the M7 gate:** re-register
  the S8 contrast as the inverse ordering (with this mechanism), or keep as-is
  and report the reversal. E4 pilot will measure it with loop harm (not ceilings)
  first.

### D5. E2 executed and ACCEPTED; per-anchor CI corrected for shared baseline noise (applied) · gate: M7/E2
- **Result (20 seeds/cell):** (i) exact-zero cell: S2j D_dir paired against the
  relabeling null = −0.0000, 90% CI [−0.005, +0.005] ⊂ ±0.01 — PASS. (ii) S1p
  D_dir 0.196 vs null 0.069 — PASS. (iii) per-anchor `Δz_j = (0.6−1)·ẑ_j(0)`
  coverage 148/160 = 0.93 ≥ 7/8 — PASS. Cell means: D_norm(S2j) = −0.589
  (prediction −0.4·z̄₀ ≈ −0.617 at panel center), D_norm(S1p) = −0.186.
- **Analysis-code fix (not a prediction change):** the per-anchor CI must include
  the SHARED round-0 baseline error — all post-event rounds compare against the
  same 50-draw p̂₀, whose logit noise (delta method, coefficient f = 0.6) is
  invisible in round-to-round scatter; omitting it makes the CI ~3× too narrow
  (coverage 0.34). With correct propagation: 0.93.

## Wave 5 — E4 full results (M8) (2026-08-05)

### D6. E4 executed (17 schedules × 20 seeds + controls; A0 attenuation 9+3+1 cells × 20)
- **E4a predictor hierarchy (Y = G_auc):** adj R² H0=0.099, H1=0.15, **H2=0.238**,
  H3(oracle)=0.881. **H2−H0 incremental adj-R² 90% CI [+0.092, +0.223] excludes 0
  → PASS** the (lead-approved) qualitative gate: reference-aware beats magnitude.
  Absolute H2≥0.5 and H2≥0.8·H3 both FAIL — expected, because harm is threshold-
  nonlinear (linear net-stats can't reach the oracle); reported as such, not gated.
- **S7 sign-flip (the H0-killer): PASS.** G(S7a)=−1.23 [CI −2.37,−0.32] < 0 <
  G(S7b)=+5.01 [CI 2.86,7.35]. S7a has net_disp 1.9 (real, large drift) yet
  NEGATIVE harm → no magnitude predictor can get the sign. **BUT** the "equal
  magnitude" sub-claim FAILS: net_disp 1.9 (S7a) vs 2.9 (S7b), |Δ| CI excludes 0 —
  because the lead-approved S7a clamp-at-w* stops S7a's rotation at round ~25 while
  S7b rotates the full 40. **Decision flagged:** keep clamp (report "large-but-
  unequal magnitude, opposite sign") vs drop clamp (both rotate 40° → equal
  magnitude, S7a overshoots w* by ~15° — cleaner "equal size, opposite harm").
- **Magnitude grids monotone:** S1 10/25/45° → 0.06/0.29/3.45 (thresholdy: the 45°
  cell carries the harm); S5 cos .9/.7/.5 → 0.28/3.73/6.17; S2 decay ≈ 0 throughout
  (norm-only drift harmless); SC1-3 ≈ 0 (reinforcing existing avoidances/tradeoffs
  moves no policy).
- **E4b √ρ attenuation: PASS cleanly.** recovered_frac tracks realized √ρ cell-by-
  cell (√ρ 1.0/0.72/0.38 → rec 1.0/0.72/0.34; high-ρ near-perfect, low-ρ mildly
  over-attenuated so line slope ~1.2, CI [1.02,1.38]). **NT (ρ=0) cells sit exactly
  at the S0 on-support noise floor 0.168 → off-support drift is invisible**, the
  support-limitation-as-a-number claim demonstrated. Required an unbiased recovered
  statistic (signed projection of Δm onto the true on-support drift direction; raw
  ‖Δm‖ is floor-inflated) and an informed-filter fix (novel_translation inflates
  null-space variance, not isotropic jump) — both applied.

### D4 CLOSED (S8 hypothesis): loop harm does NOT match the ceiling-based inversion worry.
- G(S8t)=0.17, G(S8m)=0.50, G(S8b)=0.014. Top>bottom **as originally registered**
  (S8b−S8t CI [−0.24,−0.07] < 0), but the effect is weak (<0.5) and MID-peaked, not
  monotone. Resolution: **report as-is** — S8 is a weak, mid-peaked effect; the
  ceiling analysis (which predicted S8b worst) overstated because it ignored the
  loop's tracking dynamics. No re-registration; D4 needs no action beyond reporting.

## Wave 6 — E5 result (M9) (2026-08-05)

### D7. E5 matched-budget refresh: RANKING FLIP confirmed (arms x {S1,S2,S5} x 20 seeds, B=100)
- **Pooled harm (lower = less harm):**
  | arm | G_auc (frozen ref) | R_auc (current ref) |
  |---|---|---|
  | accumulate | **1.345** (best) | 1.281 (worst) |
  | window | 3.346 | 0.445 |
  | refresh | 3.226 | **0.417** (best) |
- **best under FROZEN reference = accumulate; best under CURRENT reference = refresh
  -> RANKING FLIP**, exactly the pre-registered corruption-vs-renegotiation
  prediction. Driven by S5 (big swap): accumulate keeps old clean labels (frozen
  3.73 vs 8.9 for window/refresh) but is stale to the live grader (current 3.35 vs
  0.93 refresh). Descriptive, publishes as-is; it is the baseline any correction
  method must beat under BOTH references.
- Implementation: E5 arms live in `rm.RewardModel` (accumulate/window/refresh) +
  the refresh label-budget split (f=0.7 fresh / 0.3 re-label oldest retained under
  the current teacher) in `a1_run`. No pre-registration change.

## Wave 7 — Paper verification + overclaim review (2026-08-05)

Results paper `md_files/GRADER_DRIFT_TIER_A_RESULTS.md` written and verified.
- **Number check:** 6 independent agents re-ran every E1–E5 analysis vs the paper's
  claims — **all 46 quantitative claims match** the recomputed values, zero
  mismatches. E1 A1/S0 sub-claim verified separately (runs/e1_a1_replication.json).
- **Overclaim review (adversarial):** paper judged fundamentally honest; fixes
  applied for one high + four medium + nits:
  - **HIGH:** E4 now discloses the pre-registered S7 equal-magnitude sub-claim FAILED
    (net_disp 1.9 vs 2.9; converge-and-stop); Conclusion 3 reworded to rest on the
    sign-flip alone (does not need equal magnitude).
  - Conclusion 4 "exactly √ρ" → "close to √ρ (slope ≈1.2, low-ρ over-attenuated)".
  - §6 "with calibrated uncertainty" + Conclusion 2 → qualified "when given the drift
    process law"; fixed-Q failure noted.
  - §7 scaling reframed as theory-conjecture, NOT demonstrated result.
  - §1 "provably cannot work" reconciled with note Remark 4.1 (confound is on
    normalized statistics; jointly-fitted-with-observed-gaps temp is identified but
    gaps aren't observed operationally).
  - S8 parenthetical now states the monotonic ordering FAILED (only top>bottom held).
  - Nits: "one seed each" wording; E2 verdict de-generalized.

## Wave 8 — External-review response (2026-08-05)

Reviewer flagged: the E4a "PASS on the qualitative claim" was a laundered pass of a
failed pre-registered gate; E4b slope CI vs 1 unreported; E1 magnitude numbers
internally inconsistent; + suggested free wins (monotone probe, S8 mechanism) and a
matched-magnitude S7. Lead's meta-point: a formal paper should carry NO pass/fail /
pre-registered-gate language at all (not a registered report). Actions:

- **PASS/FAIL language removed from the paper entirely** (all "Verdict:" lines);
  §8 pre-registration reframed as a reproducibility trail, not a scorecard.
  AMENDMENTS.md stays the internal gate record; the paper makes substantive claims.
- **E4a reframed honestly:** pre-registered H2 R²≥0.5 (and ≥0.8·H3) NOT met (0.24 vs
  0.70); reported as "linear anchor predictors cap at 0.24, oracle 0.88 → harm
  nonlinear"; the H2−H0 increment CI [+0.09,+0.22] and S7 sign-flip carry the claim.
- **Monotone-probe (free win) — RAN, NEGATIVE:** isotonic G~(−dalign) leave-1-seed-
  out CV R²=0.07, Spearman 0.38, H2+hinge adds 0.00. Does NOT close the H2→H3 gap.
  Reported as a scope finding: anchors fix harm's SIGN, under-determine its MAGNITUDE
  (needs policy-relative structure anchors don't see). Honest negative, not buried.
- **S8 mechanism (free win) — partial:** loop-harm mid-peak (0.17/0.50/0.01) tracks
  an e_shift mid-peak (0.12/0.61/0.27) — mid achievable-gap axis most misdirects the
  optimal policy. Weak; reported as tentative, no firm mechanism (not the reviewer's
  "tracking-speed" story).
- **Matched-magnitude S7 (new cell S7bm, 20 seeds):** S7a (toward, net_disp 1.9)
  G=−1.23 [−2.4,−0.3] vs S7bm (away, matched ~25°, net_disp 2.3) G=+4.44 [+2.2,+7.1];
  sign flip at matched magnitude (gap 0.43, was 1.0). Closes the equal-magnitude open
  item; E4 §(a) updated to lead with the matched result.
- **E4b slope:** now reports CI [1.016, 1.376] EXCLUDES 1 → mild over-attenuation at
  low ρ (prior shrinkage), a real sub-finding (not "≈1").
- **E1 arithmetic fixed:** std rises 5.4× (0.58→3.11, was mislabeled ~7×); the 7.2×
  is the std/raw per-seed SLOPE ratio, now labeled as such (not a level ratio).
- **E6 NOT run** — held per lead ("hold E6"); unblocked and recommended, awaits go.

## Wave 9 — Second external-review response: cluster-robust stats + H2.5 (2026-08-05)

Reviewer flagged pseudo-replication (seed vs schedule resampling), an unfair
isotonic-vs-H2 comparison, the missing "H2 needs w* / sim-only" label, and proposed
an oracle-free stream-geometry predictor (H2.5). All addressed; new reproducible
module `analysis/e4_predictors.py`.

- **Clustering:** all E4a inference is now cross-schedule. Leave-one-SCHEDULE-out CV
  R²: H0 0.09, H1 0.07, H2.5 0.16, H2 0.18, H3 0.87 (isotonic 0.08). Schedule-mean
  regression (17 pts): H0 0.46, H1 0.47, H2.5 0.58, H2 0.82, H3 0.97. Increment CIs
  from a **schedule-cluster** bootstrap: H2−H0 [+0.08, +0.24] (survives), H2.5−H1
  [+0.01, +0.20] (survives). The seed-bootstrap increment was anti-conservative;
  replaced.
- **Comparability:** isotonic now reported as leave-one-SCHEDULE-out CV (0.08) on the
  same footing as H0-H3 CV; "monotone doesn't help" holds fairly (0.08 ≈ H0 0.09).
- **H2 sim-only:** paper now states H2 needs the objective w* (unavailable in
  deployment) and marks it sim-only; §7 limitation (iv) added.
- **H2.5 (thesis upgrade):** oracle-free predictor = anchor drift (m_K−m_0) in the
  quadratic form of the round-0 operational-gap covariance C_op (recomputed
  deterministically per seed from the reference stream). CV R² 0.16 ≈ 87% of H2's
  0.18 with NO oracle; beats anchor-drift-alone H1 (0.07). Reframes E4 from "anchors
  necessary" to "anchors + operational stream are COMPLEMENTARY — the stream can't
  identify the grader (E1), the anchors can't see the policy geometry, harm
  prediction needs both." E4 title, Conclusion 3, the §6 synthesis, and §7 all
  updated. Residual gap to H3 (0.16 vs 0.87) = genuine threshold-nonlinearity, open.

## Wave 10 — Third external-review response: schedule-rich reallocation + difference-CI (2026-08-05)

Reviewer flagged the fragility of H2.5−H1 (lower bound +0.006, near zero) and the
H1<H0 CV inversion (overfitting at 16 training schedules); recommended reallocating
20 seeds × 17 types → schedule-rich, and reporting the H2−H2.5 *difference* CI
instead of an "~87%" ratio.

- **Expanded to 34 drift types** (17 new: intermediate S1/S5/S2 magnitudes; mixed
  dir+norm SM1-3; multi-event MC1-2; S8 eigendirections r1/r3/r4; criterion combos
  SC4-6). 170 new A1 runs (job 14527487); analysis now 34 types × 20 seeds. `_first_event`
  fixed to use `first_event_round()` (handles multi-event MC*). All in e4_full +
  e4_predictors (reproducible).
- **Difference-CI (point 2):** H2 − H2.5 schedule-cluster CI **[−0.036, +0.052]
  includes 0** → dropping the oracle reference `w*` costs **no detectable
  generalization**. This replaces the "~87%" ratio everywhere (cleaner, defensible).
- **The richer design changed the story honestly:** in-distribution increments hold
  and are cluster-robust — H2−H0 [+0.04,+0.17], H2.5−H0 [+0.02,+0.18] (both exclude
  0); schedule-mean R² H0 0.38 / H2.5 0.49 / H2 0.65 / H3 0.97. BUT leave-one-TYPE-out
  CV collapsed for ALL linear tiers (H0 0.00, H1 −0.09, H2.5 −0.06, H2 −0.04; oracle
  H3 0.91). Honest conclusion: anchor/stream readouts carry real harm signal *within*
  studied drift types and the oracle-free H2.5 matches the w*-requiring H2, but harm
  does NOT linearly extrapolate to an unseen drift *type* (only the true shift/lag
  decomposition does) — threshold-nonlinear, type-specific.
- **Reviewer's specific prediction did NOT materialize** (H2.5−H1 stayed +0.009, not
  "well clear of zero") — reported honestly; the richer design instead revealed the
  extrapolation limit and yielded the clean H2−H2.5-includes-0 deployability result.
  Paper E4(a), Conclusion 3, §7(iv)-(v) rewritten; S7/E4b/E5 unchanged.

## Wave 11 — Review round 4 (model-based + equivalence) + E6 unlocked (2026-08-05)

**Review round 4:**
- **Equivalence margin (applied):** replaced "H2−H2.5 CI includes 0" with a proper
  equivalence statement — CI [−0.036,+0.052], half-width ≈0.045, so w* buys ≤ ~0.05
  adjR² beyond H2.5, < half the ~0.10 reference-aware gain (H2−H0). Rules out w*
  mattering by more than half of what it buys at all.
- **Model-based predictor (RAN, NEGATIVE, sharpens the story):** computed the shift
  structure from anchor estimates m_k via soft VI (oracle-free: m_0 as reference +
  env model), integral form. Leave-one-type-out CV = +0.003 (endpoint proxy was
  +0.048); essentially zero, like the linear readouts. Conclusion: functional form is
  right (oracle H3 = 0.91 computes the same structure from TRUE states), so the
  bottleneck is the ANCHOR INPUTS (√ρ loss propagates through the policy solve), not
  the predictor — "you cannot compute what you cannot observe." Modules
  analysis/e4_modelbased.py (+ e4_predictors H2−H2.5 CI). Paper E4(a) closing
  rewritten around this (better than a pure negative).

**E6 UNLOCKED + RUN (lead go-ahead 2026-08-05; Tier-A, not scaling):**
- Built the gradient-coupled mechanism: policy.feature_expectation (discounted
  behavior direction), drift.gradient_coupled_step / random_rotate_step, a1_run E6
  loop path (per-round μ history → ḡ = Δμ; rotate v̂ 2°/round toward ḡ). Churn
  interpreted as reset-to-v0 (fresh raters from the original pool) — the
  rotation-resets-capture mechanism; smoke-test caught that a random-reset churn was
  wrong (added harm instead of resetting), fixed to reset-to-v0.
- **Both hypotheses HOLD (20 seeds, e6.py, runs/e6):** coupled+persistent G_auc 0.55
  [+0.34,+0.76] largest; coupled>random (persistent) +0.50 [+0.28,+0.72]; churn
  difference-in-differences +0.26 [+0.05,+0.48] (churn resets coupled ~5× more than
  random). Bonus: coupled drift LESS detectable (rec/true 0.36 vs random 0.56) — more
  harmful AND harder to see. Strong signature → Paper-2 centerpiece candidate. Added
  as paper §5 E6 + Conclusion 6. (Detect metric = nan for churn cells: they end at a
  round-40 churn reset so net true drift ≈ 0; reported from persistent cells only.)

## Wave 12 — Review round 5: E6 detectability mechanism check + rate replication (2026-08-06)

- **ρ mechanism check (the important one): the "adversarial drift is less visible"
  finding is DEMOTED, not confirmed.** Computed realized ρ (vs the frozen A1 battery,
  rebuilt deterministically per seed): ρ_coupled = 0.93, ρ_random = 0.97 — ≈equal and
  both on-support (E6 drift is confined to Ŝ_ach by construction, so it CANNOT be
  off-support). ρ ratio 0.96, not the 0.41 the √ρ law would need. Both recovered
  fractions (0.36/0.56) sit far below their √ρ ceiling (~0.96) → the gap is
  tracking-lag + directed-vs-random net-displacement (directed drift accumulates a
  larger net, shrinking the recovered fraction), NOT a support blind spot.
  **Consequence: the reviewer's proposed synthesis "the blind spot is where the
  threat lives" is FALSE here (the threat is on-support) and was NOT added.** The E6
  section, its title, and Conclusion 6 were corrected to drop the visibility claim and
  report ρ≈equal. Good catch — the check prevented a wrong headline.
- **Rate replication (moves E6 to confirmed):** both contrasts hold and scale
  monotonically at η = 1°/2°/4°/round — coupled−random (persistent) +0.34/+0.50/+2.31,
  churn DiD +0.24/+0.26/+2.05, all CIs exclude 0. (Bugfix: the e6.py tag refactor left
  a stale coupled_atten reference that crashed the first rate job; fixed, re-run.)
- **Churn-immune limitation added (§7 iv-churn):** the churn defense works only
  because replacements are drawn toward the baseline grader; if the rater population
  co-adapts, churn draws from an already-drifted pool and gives no protection.

## Wave 13 — Review round 5 follow-up: don't assert an untested mechanism (2026-08-06)

Reviewer: the retraction replaced one unverified claim with another (the "net
displacement + tracking lag" mechanism was itself untested, and the lag half is
backwards — matched instantaneous velocity ⇒ matched absolute deficit ⇒ larger
coupled denominator predicts a HIGHER fraction, not lower).

- **Matched-net-displacement diagnostic (ran it):** coupled true_net 3.16 vs random
  1.02 (denominator IS 3.1× larger), but at MATCHED net displacement the recovered
  fraction is still lower for coupled (0.32 vs 0.56, gap −0.25 persists). So neither
  ρ (equal) nor the denominator explains the gap, and I have no tested mechanism.
  **Paper E6 visibility note rewritten:** report the numbers, note both simple
  explanations fail, assert NO mechanism and NO adversarial-visibility conclusion.
  (Removed the untested "tracking lag + net displacement" sentence.)
- **Scope limitation added (§7 iv-e6scope):** E6 drift is confined to the achievable
  subspace BY CONSTRUCTION (the A6 rank-deficiency fix), so E6 cannot test whether
  adversarial drift goes OFF-support — it forecloses the question. With a learned φ
  the achievable set isn't fixed, so "does adversarial drift escape the battery's
  support?" is a live Tier-A+ question, not settled by E6's on-support null.
- **E4a↔E6 cross-validation (added to E6):** E6's rate scaling (2°→4° more than
  quadruples harm) is the SAME threshold-nonlinearity as E4a's magnitude grids
  (0.06/0.29/3.45) — two designs converging on the harm-function shape → the
  linear-predictor ceiling is about shape, not measurement noise.
- **E6 reframed as demonstration, not keystone** (blind-spot composition is dead);
  √ρ (§E4b) now explicitly stands alone, independent of E6.
- **Bug fixed:** an earlier E6-insert edit had dropped the "## 6. Conclusions"
  header (E6 ran straight into the conclusions list for two turns); restored.

### Implementation-time resolutions (defaults chosen; flag if you disagree)
- **`first_event`** for the `G_auc`/`R_auc` window on schedules without a discrete
  jump: defined as the first round at which the schedule induces *any* drift —
  S2 → 20 (norm_decay start), S4 → 1, S3 → 10 (first redraw), S7 → 1 (continuous
  from round 0). Discrete-event schedules use the event round.
- **S7 toward-rotation endpoint:** 1°/round toward w* from 25°-off over K=40 reaches
  w* at round ~25, then continues past. Default = **clamp at w*** (stop rotating
  once aligned) so S7a is monotone-improving and `G(S7a) < 0` is clean; the "keep
  rotating past" variant would make S7a non-monotone. Flag if you want overshoot.
- **S8 axis guard:** `u = normalize(P_{vhat⊥} e_i)` is ill-defined if the gap
  eigenvector `e_i` is nearly parallel to `vhat`. Default = assert
  `‖P_{vhat⊥} e_i‖ > 0.1` at battery build; if violated, fall to the next
  eigenvector and log the substitution.
- **`n_probe` round-0 baseline:** stays 50 (v0.1 fix #11 NOT adopted per lead) —
  D_dir's equivalence test is slope-based, so a constant baseline-probe noise
  offset does not create a slope; the 500-baseline was unnecessary.
