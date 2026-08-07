# Stage F scale-up — consolidated results

**Arc.** We stress-test the Bayesian-REX safety primitive (a reward-posterior value-at-risk bound
for policy ranking / reward-hacking detection) under grader drift at neural (Metaworld/PEBBLE)
scale, 240 runs. Findings: (1) a clean **drift-blind representation → selection-failure** effect;
(2) it's about **representation coverage, not model class**; (3) it shows up as **reward-hacking
under drift**; (4) the mechanism is a **posterior-mean (point-estimate) mis-ranking**, *not* an
uncertainty/OOD-flag failure — so it is distinct from, and not caught by, B-REX's uncertainty-driven
No-Op defense; (5) harm grows on **realistic multi-run pools**, where even well-specified selection
is not immune at strong drift. The practical landing (§7e) is **auditable per-criterion coverage** (a
graded-coverage law); the honest thesis is §8. *(An earlier "collision hides OOD-ness" mechanism was
tested and refuted — §6.)*

**How to read this.** §1–2 setup & method. §3–5 core result + coverage (not model class) +
reward-hacking. §6–7 does it arise **naturally** (frozen encoders) / on **realistic** pools, plus
the verified mechanism. §7b–7d the same question in **three modalities** (state RL, LLM, vision). §7e
the practical landing — **auditable per-criterion coverage** (drift-independent; drift is the silent
transition onto low-coverage criteria). §7f a **pipeline diagnostic** (the loop can redirect a policy —
*not* a coverage result). §8 thesis · §9 positioning · §10 open questions.

## 1. Setup
**10 Metaworld tasks** (`button-press`, `sweep-into`, `drawer-open`, `door-open`, `window-close`,
`hammer`, `door-close`, `door-unlock`, `drawer-close`, `window-open`) × 8 seeds × 3 arms {**NONE**
(static), **SAT λ0.5**, **SAT λ1.0** (satiation drift)}, 300k steps; **240 runs**. Grader scores
segments by `g_t = w_t·φ(s,a)`, `φ = softplus(P·[s,a])` a fixed random 6-dim map; `w_t` drifts per
feedback round (SAT down-weights features the agent overproduces). Deterministic teacher; shared
`drift_seed=0` (all runs share `w_0`, `φ`). Sanity: NONE `w_dist=0`; SAT `w_dist` monotone in λ
(≈0.15→0.31); grader tracks the agent (`J_current≥J_init`) at moderate λ.

## 2. Decision problem & metrics (post-hoc; no extra runs)
Each run exports per round `t`: grader `w_t`, achieved profile `m_t`, labeled design
`(sphi1,sphi2,lbl)`. The Bayesian-REX question: *can an α=0.05 VaR bound rank/select policies under
the drifted grader?* Pool comparisons up to `t`, **fit a reward model on that fixed data**, take each
policy's α=0.05 VaR lower bound over the eval pool, score vs. true value under the current grader
(`m_i·w_t`). Metrics: **rank correlation** (Spearman LB vs true value) and **selection regret**
(normalized argmax-VaR regret). The RM class is the decision-maker's *post-hoc* choice — it never
re-enters training (which is identical across conditions). Conditions: `full` (linear, full 6-dim =
**well-specified**, since the grader is linear in φ), `kproj` (linear on random k<d = generic
misspec), `adv` (linear on k<d **blind to the drift direction** `u=w_T−w_0` = worst-case), and
nonlinear analogs via a **Bayesian last layer** (`mlp_bll`: one MLP's hidden features through the
same VaR machinery, hidden=6 to match).

## 3. Core result (single-run pool) — pooled over **10 tasks × 8 seeds, n=80** (N / λ0.5 / λ1.0)

| condition | selection regret ↓ | rank corr ↑ |
|---|---|---|
| `full` | 0.000 / 0.004 / 0.021 | 1.00 / 0.91 / 0.73 |
| `kproj` | 0.064 / 0.142 / 0.248 | 0.88 / 0.64 / 0.44 |
| `adv` | 0.207 / **0.496** / 0.338 | 0.68 / **0.18** / 0.30 |
| `mlp_bll` | 0.034 / 0.018 / 0.040 | 0.83 / 0.66 / 0.52 |

Well-specified selection is robust to drift (rank 1.00→0.73); drift-blind misspec collapses at
moderate drift (rank 0.68→0.18); `kproj` degrades monotonically; the matched nonlinear `mlp_bll` is
robust-ish (0.83→0.52) — well above `adv`, confirming coverage (not model class) is what matters.
Neural-scale echo of the tabular Bayesian-REX failure: honest about *"is THIS policy safe?"*,
unreliable about *"WHICH policy?"* The 10-task refresh (n=80) confirms the original 5-task pattern
with tighter CIs — not task-cherry-picked. Figures: `results/fig_sel_regret.png`,
`fig_rank_corr.png` (per-task + pooled, all conditions × drift arms).

**Why moderate drift is worst:** `adv` is blind to the single net direction `u`, so it bites only
when drift is a *coherent* march along `u`. Measured from the logged weight trajectories (n=80/arm):
at λ0.5 the weight path is fairly **straight** (straightness 0.63, dominant feature stays ~1.6
distinct dims → `u` summarizes the move → max damage); at λ1.0 the path **curves through more
feature directions and individual features overshoot then recede** (straightness 0.47, ~2.5 distinct
dominant features; whole-vector reversals are rare, so it *curves* rather than oscillates) → `u`
under-summarizes it and `adv` decays to the `kproj` level. `full`/`kproj` degrade monotonically with
total drift and so are worst at λ1.0.

**Per-task rank correlation** (core conditions, n=8 each, N / λ0.5 / λ1.0). Confirms the pattern
holds task-by-task, with two heterogeneities: **hammer** is the one task where even `full` degrades
under drift (1.00→0.35), and **window-close** has a task-geometry `adv`-at-NONE outlier (0.16).

| task | `full` | `kproj` | `adv` |
|---|---|---|---|
| button-press | 1.00 / 0.99 / 0.76 | 0.88 / 0.74 / 0.41 | 0.76 / 0.31 / 0.18 |
| sweep-into | 1.00 / 0.92 / 0.73 | 0.88 / 0.55 / 0.33 | 0.67 / 0.04 / 0.42 |
| drawer-open | 1.00 / 0.96 / 0.73 | 0.82 / 0.70 / 0.34 | 0.74 / 0.34 / 0.21 |
| door-open | 1.00 / 0.95 / 0.76 | 0.92 / 0.71 / 0.49 | 0.86 / 0.13 / 0.22 |
| window-close | 1.00 / 0.89 / 0.77 | 0.92 / 0.65 / 0.49 | **0.16** / −0.08 / 0.45 |
| hammer | 1.00 / **0.60** / **0.35** | 0.85 / 0.61 / 0.32 | 0.91 / 0.34 / 0.01 |
| door-close | 1.00 / 0.86 / 0.67 | 0.85 / 0.55 / 0.45 | 0.51 / −0.02 / 0.23 |
| door-unlock | 1.00 / 0.96 / 0.79 | 0.81 / 0.60 / 0.44 | 0.48 / 0.02 / 0.20 |
| drawer-close | 1.00 / 0.97 / 0.87 | 0.91 / 0.70 / 0.60 | 0.84 / 0.37 / 0.51 |
| window-open | 1.00 / 0.96 / 0.91 | 0.93 / 0.62 / 0.55 | 0.88 / 0.40 / 0.56 |

## 4. Representation coverage, not model class (2×3 grid)
Cross {linear, nonlinear-BLL} × {full, generic k<d, drift-blind k<d}. Rank corr (N / λ0.5 / λ1.0):

| | full | generic k<d | drift-blind k<d |
|---|---|---|---|
| **linear** | 1.00 / 0.91 / 0.73 | 0.88 / 0.64 / 0.44 | 0.68 / **0.18** / 0.30 |
| **nonlinear** (BLL) | 0.83 / 0.66 / 0.52 | 0.43 / 0.27 / 0.18 | 0.49 / 0.36 / 0.30 |

Full-representation is robust in **both** classes; reduced-representation fails in **both** — so
**"just use an MLP" is refuted; the governing variable is coverage.** *Nuance:* the "blind
*specifically* to `u`" effect (drift-blind ≪ generic) is a linear/well-specified-class result —
because the grader is linear in φ, a random *linear* projection preserves the value signal, leaving
room for targeted `u`-blindness to hurt; a nonlinear model on a random 3-dim input is already
mediocre (drift-blind ≈ generic). The model-agnostic statement is the coarser full-vs-reduced contrast.

## 5. Reward-hacking under drift
Two probes separate two effects:
- **Drift-gap impostor (drift-specific):** inject a policy whose value collapsed most along `u`.
  Under NONE nobody deploys it (0%); under λ0.5 drift-blind `adv` deploys it **36%** while
  `full`/`mlp_bll` never do. Drift breaks detection.
- **Box-corner adversarial-to-the-bound (representation-exploitability):** full-representation bounds
  ≈0 (`full` 0.00–0.06, `mlp_full` ≈0.01); every reduced-representation bound is exploitable
  (box_regret 0.13–0.53), largely **drift-independent** (probes representation gaps, not drift).

## 6. Does the failure arise *naturally*? (frozen encoder, mechanism, geometry)
We trained a reward encoder on **early (pre-drift)** preferences, froze it, tested the late grader:
- **Emergent omission is real but doesn't cause failure.** The frozen encoder retains almost none of
  `u` (`R²(φ·u | z_early)` = 0.07 at bottleneck b=2 → 0.41 at b=6), yet on the natural pool
  `frozen ≈ fresh` (rank) and it is **not** hacked on box corners (box_regret ≈0.01). Only *genuine
  collapse* fails: **ReLU b=1** is hacked (box_regret **0.77**) while tanh (any b) and larger ReLU
  stay ≤0.26 — **bounded activations mitigate.**
- **Mechanism = posterior-MEAN mis-estimation, NOT an uncertainty/OOD-flag failure** (verified).
  Uncertainty (mean−VaR) on the bad `u`-corner: for `full` the ratio is **0.76 (<1)** — `full` does
  *not* flag it; it avoids it because its *mean* estimate (having recovered `w_t`) rates it low. `adv`,
  blind to `u`, mis-estimates the corner's *mean* as high and deploys it. So the failure is a
  **reward-mean bias of a representation that can't track the drift axis** — distinct from B-REX's
  uncertainty-driven No-Op detection, which does **not** catch it. *(This refuted the earlier
  "collision hides OOD-ness" story.)*
- **On the realistic multi-run pool: still no harm** — `frozen ≈ fresh` (λ0.5 b6: 0.082 vs 0.073;
  λ1.0 b6: 0.179 vs 0.223). Dropping one of six directions is compensable (`w_T=w_0+u`; the head
  captures `w_0` in the 5 retained dims). **Three naturalness tests agree** the frozen encoder is robust.
- **"Drift heads for the blind spots" is λ-dependent:** variance ratio of `u` vs comparison
  covariance = 0.67 (blind) at λ1.0 but **1.08 (average) at λ0.5** — the regime where `adv` fails
  most is *imposed* blindness, not natural.

**Scope of the realistic threat (honest):** reproducible failures come only from (i) *imposed* total
`u`-blindness (`adv`/`mlp_adv`), (ii) *severe* collapse (ReLU b=1), or (iii) an RM *fit on
drift-contaminated data* (`full`/`mlp_full` at high λ on diverse pools). An ordinary frozen encoder
that merely *down-weights* `u` is **not** enough. Claim the failure for these conditions, not "any
frozen/misspecified RM."

## 7. The realistic pool: multi-run selection (`results_multirun/`)
Practitioners select across **multiple runs** (best-of-N). Pooling last-half profiles across 8 seeds
gives **2.5×–3.1× more spread along `u`** and exposes more harm. Selection regret, single→multi:

| condition | single-run | multi-run |
|---|---|---|
| `full` | 0.000 / 0.001 / 0.007 | 0.000 / 0.014 / **0.211** |
| `adv` | 0.31 / 0.52 / 0.28 | 0.11 / 0.46 / 0.33 |
| `mlp_full` | 0.007 / 0.005 / 0.017 | **0.137 / 0.079 / 0.257** |
| `mlp_adv` | 0.17 / 0.13 / 0.13 | 0.33 / **0.61** / 0.43 |

(1) the single-run "nonlinear-full robust" verdict is **partly a clustered-pool artifact**; (2) even
`full` is not immune at strong drift (0.21 @ λ1.0); (3) the drift-blind failure is **stronger** here.
**Control:** the reference RM ranks *other* runs' policies → a drift-independent cross-run term
(nonzero at NONE: `mlp_full` 0.137). A **pooled-RM control** only partly lowered it (`mlp_full` stayed
0.118 = intrinsic over-flexibility, not OOD), but the **drift-blind failure survives**: drift-added
regret (SAT−NONE) is **+0.37 (`adv`) / +0.47 (`mlp_adv`)** at λ0.5. The core effect is not an artifact.

## 7b. Real LLM-as-judge under a drifting standard (`llm_judge_drift/`)
A frozen LLM (Qwen2.5-3B, mean-pooled 2048-d hidden) + Bayesian linear head = B-REX's structure on an
LLM; 1500 responses, 6 attributes, standard drifts thoroughness → conciseness. **No drift-specific
blind spot** (drift direction covered R² 0.56 > random 0.44±0.14; regret slope −0.008/stage = flat).
**Narrow head fails, rich representation is robust:** a k=8 head ranks 0.1–0.24, the full (PCA-128)
representation is far better and does not degrade *with drift* (its baseline mismatch is general — the
embedding encodes specificity/formality/length well, R² 0.6–0.8, but hedging/sentiment poorly,
0.01/0.20). Only the narrow-linear-head regime fails — as in the frozen-encoder tests.

## 7c. Vision reward model under a drifting standard (`llm_judge_drift/vision/`)
Frozen **ImageNet ResNet18** and **DINOv2** on 2000 CIFAR images, 6 low-level attributes, standard
drifts vivid/warm → dark/cool. Same result: **no drift-specific blind spot** (drift-axis coverage ≈
random: ResNet 0.73≈0.72, DINOv2 0.39 vs 0.46), **regret falls with drift** (slopes −0.010 / −0.033),
narrow heads worse (ResNet k=2 0.40 → k=64 0.18). Nuance: **self-supervised DINOv2 covers low-level
attributes worse than classification-trained ResNet18** (0.39–0.46 vs 0.73) — a "richer" encoder can
have *lower* linear-readout coverage of specific criteria (relevant to judge reliability, orthogonal
to drift).

### How narrow must the head be? (k-sweep)
On Metaworld (d=6), regret is governed mainly by **capacity k** (random k-head 0.48@k=1 → 0.01@k=5,
even with no drift); the *drift-specific* component is small (≤0.04, peaks at k≈3–4) and
**task-dependent** (most tasks: none; **hammer** fails drift-specifically at every k, +0.07–0.18). On
the **text** embedding the threshold is fuzzy: rank crosses 0.5 at k\*≈16 but regret plateaus ~0.40
for all k — a baseline floor, and a nonlinear probe recovers the drift direction no better than ridge
(0.58 vs 0.56), so the floor is a **genuine absence** of signal, not a linearity limit. **Principle:**
a clean k-threshold requires the frozen representation to *contain* the grader signal; if pretraining
didn't represent the criterion, no head helps.

## 7d. Cross-modality summary — the same result in three modalities

| modality | frozen representation | drift-specific blind spot? | regret trend *with drift* | narrow head fails? |
|---|---|---|---|---|
| **State RL** (Metaworld φ, 6-d) | true φ vs projections | only under *imposed* `adv`, or *hammer* | `full` robust; `adv` inverts (0.64→0.15) | yes (needs k≈d) |
| **Text** (Qwen2.5-3B, 2048-d) | LLM hidden state | **no** (0.56 > random 0.44) | flat (−0.008) | yes (k=8: 0.1–0.24) |
| **Vision** (ResNet18/DINOv2) | frozen features | **no** (≈ random) | falling (−0.010/−0.033) | yes (k=2 0.40→k=64 0.18) |

**Reading:** across all three, a *rich* frozen representation covers the drift axis about as well as
any random direction and is robust to the drift itself; the failure appears only under **imposed**
blindness, **specific task geometry** (hammer), or a **narrow/low-capacity linear head** (B-REX's
structure) — never as a generic property of frozen encoders or judges.

## 7e. The practical landing: auditable per-criterion coverage (`llm_judge_drift/`)
§7d showed drift is **not generically dangerous** — a rich representation covers most criteria/directions
about as well as random. So when *should* you worry? The answer is a property of the **representation,
not of drift**: a frozen judge has **measurable per-criterion blind spots**, and coverage R² predicts
how reliably it can select on a criterion — with or without drift.

- **The law (drift-independent core), verified across 26 criteria** (`coverage_law/`, 6 attributes + 20
  random weightings): coverage R² **predicts** selection rank — **Spearman = +0.75, Pearson = +0.75**
  (rank 0.24 at R²0.09 → 0.81 at R²0.78; shuffled-label null R² ≈ **−0.20** rules out overfitting;
  oracle rank ≈ **0.93**). This is a statement about the judge's per-criterion reliability: a
  practitioner who simply *chooses* to rank by a low-coverage criterion hits the same wall — **no drift
  required**.
- **Some *important* criteria are poorly covered.** For a 3B judge, factual correctness R² = **0.20**
  (nonlinear 0.31), hedging **0.01** — real, concerning reliability limits for LLM-as-judge, not
  artificial adversarial directions.
- **Drift is the silent-transition mechanism** onto those specific criteria (`blindspot/`, `semantic/`):
  as a standard *evolves* specificity→hedging or length→correctness, a previously-fine judge silently
  degrades (rank 0.55→0.24; correctness 0.81→0.35) while the **accuracy monitor stays green** (no
  re-audit is triggered). This is exactly why §7d's "drift is generically safe" does **not** defuse it:
  the danger is not generic drift, it is evolving standards landing on the *specific* low-coverage-but-
  important criteria the audit flags. (The semantic failure is *partial* — the judge exploits a covered
  surface correlate, top-30 still 0.93 correct — so real failures are graded, not catastrophic.)

> **The law:** a frozen RM / LLM-or-vision judge's reliability at selecting on a criterion is predicted
> by its representation's **per-criterion coverage R²** — probeable offline, *independent of drift*.
> Weakly-covered criteria (for a 3B judge, factual correctness) are unreliable to rank on, and **no head
> fix** (capacity, nonlinearity, uncertainty bound) helps. **Drift is what silently moves you onto such a
> criterion without a signal.** Audit per-criterion coverage; treat low-R² criteria as unreliable.

## 7f. Pipeline diagnostic: the closed-loop setup can redirect a converged policy (`morl/`)
This is a **sanity check, not a result.** Question: can the closed-loop pipeline redirect a policy *at
all*? (An earlier drift onto an energy/idle objective did *not* — a weak attractor.) One full-MLP-RM arm
on mo-halfcheetah, grader weight deterministically flipped forward→backward (`w:[1,0]→[0,1]`,
`φ=[relu(x_vel), relu(−x_vel)]`). The full RM faithfully tracked the flipped grader (held-out pref-ACC
≈0.97), SAC maximized it, and the policy ran backward (native forward-reward +4669→−6695; achieved
forward 1.78→0.00, backward →2.53).

This confirms the loop works, but is **not a coverage result and not a reward-hacking or selection-failure
analog.** The learned reward and the *current* objective are **aligned** (no learned-vs-true gap — the
monitor is green because the RM is *correct*, the opposite of the silent selection failure); the drift
axis is a single-obs-dim feature **coverable by every arm** (no representation variable — single arm); and
the −6695 merely measures distance from the *intentionally-retired* forward objective — i.e. optimization
faithfully following a redefined spec ("you rewarded backward, it ran backward"). The genuinely non-trivial
closed-loop experiment — a coverage-**inversion** where a drift-blind RM keeps running forward while a full
RM follows — **could not be constructed** (attractor-vs-coverage tension; §10). So coverage is **verified to
govern *selection*** (§7e); whether it governs *optimization* is **open**.

## 8. Consolidated thesis (honest, scoped)
1. **A VaR bound is honest for *certifying a fixed policy* but can silently corrupt for *choosing
   among policies*** under grader drift, when the RM's representation doesn't cover the drift axis.
   Marginal coverage of the *chosen* policy stays honest while cross-policy ranking degrades — invisible
   to the standard RM-accuracy monitor.
2. **The governing variable is representation coverage of the drift axis** — not linear-vs-nonlinear,
   not the tail level. Full-representation is robust in either class; reduced fails in either.
3. **The mechanism is a reward-*mean* (bias) failure, not an uncertainty/OOD-flag failure** (verified).
   A representation blind to the drift axis mis-estimates value in its *point estimate* and mis-ranks;
   the VaR's uncertainty term does not flag it. Distinct from B-REX's No-Op detection (catches static
   exploits, not drift mis-ranking). Full-input models with lossy representations are *not* automatically
   vulnerable; only genuine collapse (extreme bottleneck + unbounded activation) also fails the mean.
4. **Harm scales with candidate-pool diversity.** Single-run checkpoints cluster and hide it;
   realistic multi-run selection exposes it, and even well-specified selection degrades at strong drift.
5. **The threat is bounded and nameable, not universal** (convergent across 3 modalities). It is sharp
   only under *imposed* blindness, *low-capacity/narrow-linear heads* (incl. a narrow head on a real
   LLM), or *severe* collapse; rich, naturally-trained representations (frozen encoders and a real LLM
   judge) cover the drift axis ≈ any random direction and are robust to drift itself. The dangerous
   case is specifically **B-REX-style narrow-linear-head-over-frozen-features**, not "any frozen/misspec
   RM or judge."
6. **The practical, falsifiable landing — auditable per-criterion coverage (§7e).** The strongest,
   most actionable result is *drift-independent*: a frozen judge's reliability at selecting on a
   criterion is *predicted* by its per-criterion coverage R² (verified: Spearman +0.75 across 26
   criteria; null ≈−0.20; oracle ≈0.93) — a property of the judge, testable offline with **no drift**.
   Some *important* criteria are poorly covered (factual correctness R²0.20 for a 3B judge). **Drift is
   the silent-transition mechanism** onto them: as standards evolve, a previously-fine judge becomes
   unreliable while the accuracy monitor stays green. Actionable: audit per-criterion coverage; treat
   low-R² criteria as unreliable to select on. (Capable LLMs partially encode even important criteria,
   so real failures are graded, not catastrophic.)
7. **Selection vs optimization.** Coverage is *verified* to govern **selection** (§7e). Whether it
   governs closed-loop **optimization** is **not demonstrated**: §7f only confirms — as an engineering
   diagnostic — that the loop *can* redirect a policy at all ("reward it backward, it runs backward" is
   faithful optimization, not a failure). The coverage-**inversion** that would make it a real analog of
   the selection failure could **not** be constructed (attractor-vs-coverage tension, §10).

## 9. Why it matters — positioning
**"Nobody uses linear RMs" doesn't defuse it:** the object of study is coverage, not linearity. B-REX
*is* a Bayesian linear head on frozen features; LLM-judges and reused RLHF reward models are
frozen-feature systems; even fine-tuned RMs are finite-capacity extractors with no guarantee they
track where preferences move. **"But the input has the info":** the VaR is computed from the *learned
representation*, which drops training-irrelevant directions — a frozen pre-drift encoder retains ~none
of `u`, and the realized threat (§8.3) is a posterior-mean mis-estimation along that dropped direction.
**Silent failure** motivates a drift-aware selection monitor (fixed-anchor detector gating cross-policy
ranking). **Audience:** safe imitation/offline RL (B-REX/BROIL/T-REX); RLHF reward-model robustness &
reward hacking; iterated/online RLHF; performative prediction; LLM-as-judge reliability.

## 10. Open questions
- **Closed-loop coverage-inversion.** Coverage governs *selection* (§7e); whether it governs
  *optimization* is **unresolved** — §7f only confirms (diagnostically) that the loop can redirect a
  policy at all. The **coverage-inversion** — a frozen RM protected because it cannot follow the drift —
  could not be constructed: an objective strong enough to redirect the policy is coverable by all arms
  (no gap), while a coverage-gap objective (quadratic/trajectory-level) is too weak an attractor to
  redirect at all. Would need an objective that is *both* a strong attractor *and* differentially coverable.
- **Drift-aware selection monitor** — a fixed-anchor detector gating cross-policy ranking when the
  standard has moved is untested.
- **A catastrophic (not graded) semantic demo** needs a criterion that is important *and* a total blind
  spot *and* surface-uncorrelated — hard to find in capable LLMs (itself a mildly reassuring finding).

## Artifacts
Under `dbrown-group1/zifan/stageF_scaleup/`: `results/` (core + nonlinear + figures), `results_repr/`
(2×3 grid), `results_hack/` (box-corner), `results_repr_verify/` (frozen encoder + geometry),
`results_multirun/` + `results_multirun_pooled/` (pool + control), `llm_judge_drift/` (LLM, vision,
blindspot, semantic, `coverage_law/`). Analyzers in `DriftedGrader/analysis/`; 240 runs orchestrated
by `NeuralDriftedGrader/stageF_scaleup/`.
