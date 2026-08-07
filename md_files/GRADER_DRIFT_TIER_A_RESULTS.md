# Measuring Grader Drift in Preference Learning: A Tier-A Identifiability Study

*Results write-up — August 2026. Companion to the theory note
(`grader_drift_identifiability_note_v0_2.md`) and the executable spec
(`tier_a_experiment_spec_v0_2.md`). Every pre-registered claim and every amendment
to it (all made before the confirmatory data) is logged in
`graderdrift/AMENDMENTS.md`. Code: `graderdrift/`.*

---

## 1. Motivation

Modern models are trained against a **grader** — a human annotator, a reward model,
or an LLM judge — that compares two candidate outputs and says which is better. The
policy is optimized to produce whatever that grader rewards. The unstated assumption
is that the grader is a *fixed* standard. It is not. Over weeks and months graders
**drift**: annotators grow stricter or more lenient, rubrics are edited, annotator
pools churn, and LLM judges are silently swapped for new versions. A policy trained
against a moving standard can quietly degrade *by the objective we actually care
about* while continuing to score well against the drifting grader.

The natural way to watch for this — monitor the grader's behavior on the live stream
of comparisons it is already producing — **cannot distinguish drift from rising
difficulty**. As the policy improves, the two candidates in each comparison get
closer in quality, so the grader flips more, agrees with itself less, and looks
noisier *even if it has not changed at all*. Formally: no statistic computed from the
operational stream alone can attribute the change to the grader rather than the pairs
(Prop 3 of the note), and the confound bites hardest on the *normalized* consistency
statistics practitioners actually watch — agreement, flip rate, standardized
temperature (a temperature jointly fitted with the observed margin is identified, but
the feature gaps are not observed operationally). This is the core problem: **you
cannot audit a grader from the traffic it grades.**

## 2. Goals

We set out to answer five questions, in order:

1. **Is the confound real?** Does a perfectly constant grader trigger the naive
   drift detectors purely because the comparisons get harder?
2. **Can a fixed anchor instrument see through it?** If we re-ask the grader a
   *frozen* set of comparisons, does that isolate genuine grader-side change — and
   does it decompose the change the way theory predicts?
3. **Can we estimate the drift with honest uncertainty?** A calibrated tracker of
   the grader's hidden state, with error bars that mean what they say.
4. **Does anchor-measured drift predict harm to the trained policy** — and *what*
   about the drift predicts it: its size, or its direction relative to the
   objective?
5. **What does the dumb baseline already buy?** At a fixed labeling budget, how much
   drift harm does simple data curation (windowing, refreshing) remove — and does
   the answer depend on what "harm" even means?

## 3. Theoretical foundation (the four-symbol model)

A rater's utility is `u_t(x) = w_t · φ(x) + s_t` and labels follow Bradley–Terry,
`P_t(x ≻ y) = σ(β_t (u_t(x) − u_t(y)))`. Five short facts fix the entire design:

- **Severity cancels** inside a genuine (same-rater, same-sitting) comparison — the
  offset `s_t` drops out of the utility difference. Documented rater-severity drift
  is therefore *invisible* to comparison data by construction.
- **The only identifiable rater state is `v_t = β_t · w_t`** — its direction (what
  the grader cares about) and its norm (how decisively). "Drift" means drift of `v_t`.
- **The operational stream cannot attribute change** to the rater vs. the pairs
  (the confound above).
- **An anchor battery is a linear sensor for `v_t`.** Anchor design becomes an
  experiment-design problem with a concrete rule: direction drift is seen by ~50/50
  anchors with spread-out gaps; norm drift by ~80/20 anchors; near-0/1 anchors are
  dead weight.
- **Anchors detect rater-side change robustly, but attributing it to criteria vs. a
  drifting feature map requires assuming `φ` is fixed** — the one assumption doing
  irreducible work.

## 4. Experimental design

Two rungs, both CPU-only; each run is fully determined by one config + integer seed,
and the reported results are 20-seed aggregates.

**A0 — the measurement rung (no RL).** Items are feature vectors (d = 24); "policy
improvement" is simulated by a gap distribution that concentrates over time
(`c_k = 2·0.95^k`). This isolates the instrument and estimator from RL confounds.
The grader's state `v_t` is drifted on purpose in known ways.

**A1 — the loop rung (tabular PbRL).** A 10×10 gridworld with an exact MDP (features
d = 8; goal/lava/water/grass, goal-proximity, wall-bump, stay, constant). A real
closed loop: collect segments → pair → label with the drifting teacher → refit a
logistic reward model → soft value iteration → repeat, 40 rounds. Because the MDP is
tabular we can compute **exact** policy returns, so harm is measured without RL
noise. The segment-gap features have a rank-6 achievable subspace (a constant-per-
step reward shift moves no policy); all random drift is confined to it and
"effective" drift magnitude is logged.

**The instrument.** A frozen two-panel anchor battery built once at round 0 — a
direction panel (~50/50, greedy D-optimal) and a norm panel (~80/20), plus, in A1, a
criterion-contrastive panel aligned with named features. Probed every round; anchors
never enter training.

**The estimator.** An iterated EKF / Laplace filter over `v_t` with a process model.
The calibrated ("informed") mode knows the *drift process law* — types, times, and
typical sizes of events, as a platform knows its own change processes — but not the
realized outcomes.

**Drift schedules** model real causes: sudden rubric reweighting (S1), fatigue/decay
(S2), judge-version swap (S5), rater-pool churn (S3), diffusion (S4), a genuinely new
criterion (NT), drift toward vs. away from the true objective (S7a/S7b), drift along
behaviorally high/low-variance axes (S8t/m/b), named-criterion edits (SC1–3), and a
control that changes nothing (S0) or nothing visible-and-harmful (SN).

**Metrics.** On-support fraction `ρ = ‖P_A δ‖²/‖δ‖²` (how much of a drift the anchors
can see); the drift predictors (net displacement, norm/direction split, and the
reference-aware signed alignment change); and **dual-reference harm** — `G_auc`
against the *frozen* original objective `w*` (drift-as-corruption) and `R_auc`
against the *current* standard `w_t` (drift-as-renegotiation).

## 5. Results

All confirmatory experiments use 20 seeds. Findings are stated substantively; the
pre-registered predictions and every amendment to them live in `AMENDMENTS.md` as a
reproducibility trail (the prose below makes no pass/fail claims against internal
gates).

### E1 — The confound is real (Figure 1)
Under a perfectly constant grader (S0), the naive detectors move and the anchors do
not. Operational self-agreement falls (median slope −0.0052, CI < 0), and the
*normalized* "apparent inconsistency" temperature rises **5.4×** over the run
(0.58→3.11; median slope +0.094, CI > 0). The **raw** (un-normalized) temperature
stays near the grader's true value (level 0.76–1.36); its per-seed slope is **7.2×
smaller** than the normalized one's. Meanwhile both anchor panels stay flat (D_dir
slope ≈ 3e-5, sitting at its 0.08 binomial noise floor; D_norm ≈ 0). The divergence
of the two temperatures *under a constant grader* is the finding: the confound lives
specifically in the *normalized* statistics dashboards actually watch, while the
frozen anchors do not move at all. The A1/S0 replication confirms the anchors stay
flat inside the RL loop too (D_dir at its 0.078 floor, zero slope). **A constant
grader trips the operational detectors; the instrument does not — so the instrument
is necessary, not optional.**

### E2 — The anchor split rule holds exactly
Under a pure norm change (S2j, ×0.6), the direction panel reads **−0.0000**,
statistically indistinguishable from pure sampling noise (paired diff vs. a
re-probing null, CI [−0.005, +0.005]) — the theory's sharpest falsifiable
prediction. Under a pure rotation (S1p) the direction panel moves large (0.196 vs.
0.069 null). The norm panel's per-anchor response matches the exact prediction
`Δz_j = (0.6−1)·z_j(0)` for 93% of anchors. **Each panel responds only to its own
channel — exactly zero, within sampling noise, for the pure-scale → direction-panel
cell — as the split rule derives.**

### E3 — The tracker is calibrated across every drift type
With the process-law-aware filter, per-schedule coverage of nominal-90% intervals
for the on-support state lands in [0.90, 0.93] on every single-teacher schedule (S0
0.92, S1 0.91, S2 0.93, S4 0.91, S5 0.90; pooled 0.913), with NEES/dof ≈ 0.84–0.98
(near the ideal 1.0). The population-churn schedule (S3) reaches 0.90 and is reported
with a Jensen-flattening caveat. A filter given only event *times* but a single fixed
noise magnitude fails structurally (over-covers static schedules at 0.96–0.97,
under-covers continuous-drift schedules at 0.75–0.82) — a reported cautionary
contrast. **The tracker's error bars are honest across jumps, decay, swaps,
diffusion, and churn, and NEES pinpoints exactly which process a mis-specified filter
got wrong.**

### E4 — Predicting policy harm: anchors and the operational stream are complementary
**(a) The predictor hierarchy** (**34 schedule types × 20 seeds** — a schedule-rich
design; harm and drift are largely *schedule*-determined, so the effective N is the
number of drift *types*, and we resample schedules, never seeds, throughout). Two
questions, deliberately separated: does a readout carry harm signal *within* the
studied drift distribution (schedule-mean regression + schedule-cluster bootstrap of
the increment), and does it *extrapolate* to an entirely unseen drift type
(leave-one-schedule-out CV)?

| tier | what it uses | schedule-mean R² | leave-one-type-out CV R² |
|---|---|---|---|
| H0 | anchor drift *magnitude* | 0.38 | 0.00 |
| H1 | + norm/direction split | 0.38 | −0.09 |
| **H2.5** | anchor drift × **operational-gap covariance** (no oracle) | **0.49** | −0.06 |
| H2 | + signed alignment to the objective `w*` (**needs `w*`**) | 0.65 | −0.04 |
| H3 | true shift/lag decomposition (oracle) | 0.97 | 0.91 |

*Within* the distribution, the readouts carry real, cluster-robust signal: the
reference-aware increment **H2−H0 = [+0.04, +0.17]** and the oracle-free
stream-geometry increment **H2.5−H0 = [+0.02, +0.18]** both exclude zero (90%
schedule-cluster CIs). The **headline is the deployability result**, stated as a
proper *equivalence* test rather than an absence of evidence: the cost of *dropping*
the objective `w*` is **H2 − H2.5 ∈ [−0.036, +0.052]** (half-width ≈ 0.045), so at
90% confidence `w*` buys **at most ~0.05** of adjusted R² beyond the oracle-free
predictor — **less than half the ~0.10 total gain that reference-awareness provides
over magnitude** (H2−H0). We can therefore rule out the reference mattering by more
than half of what it buys at all; the oracle-free predictor loses no detectable
generalization. The operational stream cannot *identify* the grader
(E1); the anchors cannot see the *policy geometry*; but the anchor-measured drift
weighted by the stream's own gap covariance — **the two instruments together, no
oracle** — matches what the reference-aware predictor gives. Harm prediction is a
**division of labor**, and the deployable half needs no privileged knowledge of the
objective.

The *extrapolation* column is the honest limit: **every linear anchor-derived
predictor fails out-of-type (CV ≤ 0), while the oracle decomposition generalizes
(0.91)**. The right reading is *feature incompleteness*, not a call for a drift-type
taxonomy: because the oracle (the true shift/lag features) *does* transfer across
types, a **single universal** harm law exists — the anchor/stream readouts are just a
lossy summary of it that drops the mechanism-distinguishing information, so the same
measured drift maps to different harm depending on *what kind* of drift produced it
(the readout→harm relation is **regime-dependent**). (A monotone/isotonic transform
of the directed scalar does not rescue it either — CV ≈ 0, like H0.) We also tested
the natural fix — **computing** the shift structure model-based from the
anchor-estimated grader state (the soft-VI-optimal policy under the estimate, scored
against the round-0 estimate; a world model but no oracle) instead of regressing it.
It *also* fails to extrapolate (leave-one-type-out CV ≈ 0). That pins the bottleneck:
the functional form is right — the oracle, which computes the *same* structure from
the **true** states, generalizes at 0.91 — so the barrier is the **anchor inputs**,
not the predictor. The √ρ information the battery drops propagates through the policy
solve: *you cannot compute what you cannot observe.* Closing the gap needs richer
*observable* grader state (more or better anchors), not a cleverer predictor and not
a drift-type taxonomy. Within calibrated regimes the readouts still give harm's
*direction* and rank drift; they simply do not forecast the magnitude of a mechanism
never seen.

**The decisive dissociation (S7).** At matched drift magnitude — both ≈25° of net
rotation from a shared 25°-off start, net displacement 1.9 (toward) vs. 2.3 (away) —
drift *toward* the objective produced **G_auc = −1.23** (CI [−2.4, −0.3]), *reducing*
harm below the stationary baseline, while drift *away* produced **+4.44**
(CI [+2.2, +7.1]). A 5.7-unit swing in harm across a 0.4 difference in displacement is
a *direction* effect, not a magnitude one: **no magnitude-only predictor can output a
negative harm, so H0 is falsified independently of the R² gate.** (An un-matched away
arm rotating the full 40° gives +5.01, essentially the same.) The magnitude grids
confirm the thresholdy shape (S1 at 10°/25°/45° → 0.06/0.29/3.45; S5 swap →
0.28/3.73/6.17; pure norm decay S2 ≈ 0; reinforcing-existing-criteria edits SC1–3 ≈ 0).

**(b) The attenuation law (A0).** The estimator recovers close to the on-support
fraction of drift: recovered fraction vs. realized `√ρ` has fitted slope ≈ **1.2**
(90% CI [1.02, 1.38], **excluding 1**) — a mild but real over-attenuation at low `ρ`,
consistent with prior shrinkage of small on-support signals. Drift pushed entirely
off-support (a brand-new criterion, NT) is recovered only at the noise floor (0.168)
— i.e. **invisible** — turning "anchors are support-limited" into a measured number.

*(S8, minor: harm mid-peaks across the behavioral-relevance axes (top/mid/bottom =
0.17/0.50/0.01), tracking a mid-peak in the target-displacement term e_shift
(0.12/0.61/0.27) — drift along the middle achievable-gap axis most misdirects the
optimal policy. The effect is weak and we claim no firm mechanism; the pre-registered
monotonic ordering did not hold, and a prior ceiling analysis had guessed the
opposite.)*

### E5 — The dumb baseline, and the corruption-vs-renegotiation flip
Three data-curation arms at a matched labeling budget:

| arm | harm vs. **original** `w*` (G_auc) | harm vs. **current** `w_t` (R_auc) |
|---|---|---|
| accumulate-all | **1.35 (best)** | 1.28 (worst) |
| window (last 10) | 3.35 | 0.45 |
| refresh (re-label old) | 3.23 | **0.42 (best)** |

**The arm ranking flips with the reference.** If drift is corruption (the original
objective is truth), hoarding old clean labels wins and refreshing chases the grader
into the ditch (2.4× worse under a big swap). If drift is renegotiation (today's
grader is truth), refreshing wins and hoarding is hopelessly stale (3× worse). *There
is no universally best curation policy — it depends entirely on which standard you
believe.* This is the quantified baseline any future correction method must beat
under *both* references.

### E6 — Adversarial gradient-coupled drift is more harmful than random drift
A 2×2 (20 seeds): drift {**coupled** — the grader rotates 2°/round toward the
policy's own improvement direction — vs **random** — a magnitude-matched 2°/round
random rotation} × persistence {**persistent** vs **churn**, where every 8 rounds the
rater pool is refreshed toward the baseline grader}. Harm (G_auc vs. an S0 twin):

| | persistent | churn |
|---|---|---|
| **coupled** | **0.55** [+0.34, +0.76] | 0.22 [+0.10, +0.36] |
| **random** | 0.05 [−0.04, +0.14] | −0.02 [−0.08, +0.04] |

Both pre-registered hypotheses hold. **A grader that chases the policy is
significantly more harmful than magnitude-matched random drift** (coupled − random,
persistent: +0.50, CI [+0.28, +0.72]). And **churn resets the capture**: it removes
~5× more coupled harm than random harm (difference-in-differences +0.26,
CI [+0.05, +0.48]) — periodically returning to fresh baseline raters interrupts the
adversarial rotation that random drift never builds. Both contrasts **replicate and
scale monotonically with the coupling rate** — at η = 1°/2°/4° per round, coupled −
random (persistent) = +0.34 / +0.50 / +2.31 and the churn difference-in-differences =
+0.24 / +0.26 / +2.05, with every 90% CI excluding zero — so the adversarial
signature is not an artifact of one rate. Note the *shape*: doubling the rate from 2°
to 4° more than quadruples the harm (0.50 → 2.31). That super-linear scaling is the
same threshold-nonlinearity E4a found independently in its magnitude grids
(0.06/0.29/3.45 across 10°/25°/45°) — two experiments of different design converging
on the same harm-function shape, independent evidence that the linear-predictor
ceiling of §E4a reflects the harm function's *shape*, not measurement noise.

*A note on visibility.* The recovered fraction of the coupled drift (0.36) is lower
than of random drift (0.56), which invites the reading "adversarial drift is harder
to see." We do **not** make that claim: the two obvious explanations both fail, and
we have no tested mechanism for the residual. It is **not** a support/blind-spot
effect — realized ρ is ≈ equal (0.93 coupled vs. 0.97 random), both on-support *by
construction* (all E6 drift is confined to the achievable set the battery spans), so
the √ρ law cannot apply. Nor is it merely coupled's larger *net* displacement
inflating the denominator: at *matched* net displacement (coupled taken at the round
where its net drift equals random's final) the recovered fraction is still lower
(0.32 vs. 0.56), so the gap does not close. We therefore report the numbers and
assert no mechanism and no adversarial-visibility conclusion.

E6's role here is a *demonstration*, not a keystone: it shows the instrument detects
an alignment-relevant, adversarial drift mode, yields an actionable (if
counterintuitive) churn recommendation, and scales cleanly with the coupling rate.
The √ρ scope result (§E4b) stands on its own and does not depend on it.

## 6. Conclusions

1. **The confound is a theorem, and it bites** (E1): a constant grader trips the
   normalized consistency detectors practitioners watch (agreement, flip rate,
   standardized temperature), while the frozen anchors do not move. No statistic
   computed from the operational stream alone can attribute the change to the grader
   — it is not identified.
2. **A frozen two-panel anchor battery is a calibrated instrument** (E2, E3): it
   detects rater-side change and cleanly separates *direction* from *scale* drift;
   and when the tracker is given the drift process's law (the "informed" filter) it
   reports the hidden grader state with honest 90% error bars across jumps, decay,
   swaps, diffusion, and churn. (A filter given only event *times* but not their
   magnitudes is not calibrated — that failure is reported, not hidden.)
3. **Drift harm depends on direction, not magnitude; prediction is a division of
   labor; and it does not extrapolate to unseen drift types** (E4a): at matched drift
   magnitude, drift toward the objective *helped* while drift away *hurt*, so a
   magnitude-only readout cannot even get the *sign* right. Within the studied drift
   types the anchor-measured drift weighted by the operational stream's *own* gap
   covariance — an **oracle-free** predictor — carries cluster-robust harm signal and
   **loses no detectable generalization versus the predictor that needs the objective
   `w*`** (H2 − H2.5 CI includes 0). Neither instrument suffices alone (the stream
   cannot identify the grader, E1; the anchors cannot see the policy geometry). But
   harm does **not** linearly extrapolate to an unseen drift type — every linear
   readout has CV ≤ 0, only the true shift/lag decomposition generalizes. Because the
   oracle *does* transfer, a single universal harm law exists; the readouts are a
   lossy summary of it (feature-incompleteness, not a need to classify drift types).
4. **The instrument's blind spot is a computable number** (E4b): it recovers close
   to the `√ρ` fraction of any drift (fitted slope ≈ 1.2, with low-`ρ` drift mildly
   over-attenuated), and drift into genuinely new criteria is invisible — so every
   anchor-based conclusion carries an explicit scope (`ρ`).
5. **"Harm" is reference-dependent** (E5): the best remedy under the corruption view
   is the worst under the renegotiation view. Any claim about grader drift must state
   which objective it is measuring against.
6. **Adversarial (gradient-coupled) drift is more harmful than random drift, and
   baseline churn resets it** (E6): a grader that rotates toward the policy's own
   improvement direction does significantly more damage than magnitude-matched random
   drift (replicated at 1/2/4°/round), and rater-pool churn that returns to baseline
   removes ≈5× more of the coupled harm than of random harm (the rotation-resets-
   capture mechanism). We checked whether this adversarial drift is also *less
   visible* and found it is not a blind-spot effect — it stays on-support (ρ≈0.93),
   so the √ρ law does not apply; we make no visibility claim.

Together: **no statistic computed from the traffic a grader grades can attribute
change to the grader (least of all the normalized statistics dashboards watch); a
small frozen battery of comparisons can — with uncertainty that is calibrated when
the estimator is given the drift process's law (E3); and predicting harm needs *both*
instruments — the anchor-measured drift and the operational stream's geometry — since
the stream cannot identify the grader and the anchors cannot see the policy geometry,
and the anchors see drift only up to the fraction `√ρ` they span.**

## 7. Scope, scaling, and limitations

**Everything here is tabular / synthetic, by design.** A0 is feature vectors
(d = 24); A1 is an exact tabular gridworld (d = 8) where returns are computed in
closed form. This Tier-A build deliberately excludes continuous control (Metaworld,
DMC), neural reward models, LLM judges, human data, and any drift-*correction*
algorithm (see the spec's non-goals). The purpose was to establish the
*identifiability foundation* — what pairwise data can and cannot see of a drifting
grader — cleanly, without RL or representation-learning confounds.

**Why the *theory* is expected to carry to scale (a conjecture, not a result).** The
theory is representation-agnostic: `φ` is an arbitrary fixed feature map, so the
*structural* results — the confound, the identifiable state `v_t`, the
anchor-as-linear-sensor property, and the split rule — are expected to hold for any
setting whose labels follow a link of a utility difference, including neural
embeddings and LLM-judge preferences. But this is a theoretical expectation, **not a
demonstrated result**: what is tabular here is the implementation (exact policy
evaluation; explicit low-dimensional `φ`), and whether the *empirical* findings
survive at scale — where `φ` is learned and may itself drift — is exactly what the
next rungs must test, not something these experiments establish.

**What genuinely changes at scale (the next rungs, not done here — and not to be
built until this rung is complete):**
- *Continuous control (Tier-B, Metaworld/DMC):* exact evaluation is replaced by RL
  with estimated returns; the reward model becomes neural, so the fixed-`φ`
  assumption is stressed and the note's §6 attribution limit (criterion drift vs.
  representation drift) moves to center stage. The anchor battery and EKF are
  `φ`-dimension-agnostic and carry over; the estimand becomes `P_A v_t` in a learned
  space.
- *LLM judges:* anchors become a frozen eval set re-graded across judge-version
  bumps — a natural experiment. `Φ_A` is unknown, so one works in *anchor
  coordinates* (norm- and angle-type readouts of the trajectory), with distortion
  controlled by `cond(Φ_A)` — one more reason the direction panel spreads its gaps.
- *Human raters:* Bradley–Terry independence and the fixed-link assumption are
  approximations; session-level dependence and context effects become covariates.

**Standing limitations.** (i) The fixed-`φ` assumption is the load-bearing one and
must be named in any deployment. (ii) The instrument's probe budget (frozen anchors
re-graded every round) is a real cost the operational stream does not incur. (iii)
Attribution to a human-level *cause* ("this was a version swap") is not identified —
different causes can produce the same state-change signature; the instrument reports
the signature (direction/scale/which-criteria), not the culprit. (iv-churn) The E6
churn defense works only because replacement raters are drawn toward the *baseline*
grader (un-entrained). If the rater population itself co-adapts — annotators
increasingly exposed to model outputs in general — churn draws from an already-drifted
pool and provides **no** protection; this churn-immune regime is the failure mode any
churn-based recommendation must flag. (iv-e6scope) E6's drift is confined to the
achievable subspace *by construction* (the rank-deficiency amendment that fixes the
A1 feature confound), so E6 **cannot** test whether adversarial drift preferentially
goes *off*-support — it forecloses that question rather than answering it. This
matters because with a *learned* `φ` the achievable set is not fixed: policy
improvement genuinely pushes outputs into regions no round-0 anchor spans, so
"does adversarial drift escape the battery's support?" is a live Tier-A+ question,
not settled by E6's on-support null. (iv) The strongest
harm-*magnitude* predictor that uses the objective `w*` (H2) is sim-only, since `w*`
is unavailable in deployment; the deployable path is the oracle-free H2.5
(anchors + operational geometry), which loses no detectable generalization versus H2
(H2 − H2.5 CI includes 0). But (v) **no** linear anchor/stream readout extrapolates
harm to an *unseen* drift type (leave-one-type-out CV ≤ 0; only the oracle
decomposition generalizes) — the readout→harm relation is regime-dependent, so these
readouts rank drift within calibrated regimes rather than forecast a novel one. This
is feature-incompleteness (a universal law exists from the shift/lag features), not a
requirement to enumerate or classify drift types.
Quantifying and closing that residual is open work, and is the same gap as the
frontier question of whether a drift-invariant reward core exists.

## 8. Reproducibility

- Code: `graderdrift/src/graderdrift/` (teacher, drift, streams, anchors, probe,
  diagnostics, estimator, envs/grid, rm, policy; drivers in `experiments/`, analysis
  in `analysis/`). 30 proposition- and amendment-derived unit tests (`pytest`).
- Every run is one YAML/seed; logs are JSONL; analysis reads only logs.
- Compute: pure CPU; all batches ran via SLURM on `notchpeak-shared-short` (never the
  login node). A full A0 experiment is < 1 min/run, a full A1 run ~10 s.
- Figures: `md_files/figures/fig1_stationarity_null.png`,
  `fig_e3_trajectories.png`; E4/E5 tables in `graderdrift/runs/` and `AMENDMENTS.md`.
- **Pre-registration hygiene:** `graderdrift/AMENDMENTS.md` logs every change to the
  pre-registered predictions/criteria, each made before its confirmatory data, with
  rationale — including the three that materially sharpened the study (the
  standardized-vs-raw temperature pair, the process-law-aware filter, and the
  reference-aware predictor replacing a magnitude-only correlation).
