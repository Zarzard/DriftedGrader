# Paper A — Outline: *What Can a Frozen Judge Rank? An Offline Coverage Audit for Reward Models and LLM Judges*

*Working title alternatives: "Coverage, Not Accuracy: Auditing What a Frozen Judge Can Rank"; "Representation Coverage Predicts Which Criteria a Reward Model Can Rank On."*

**The pivot (per the venue review):** lead with **representation coverage** as an offline, drift-free predictor of ranking reliability; demote grader drift from thesis to motivation (§2 + §8). Every dramatic drift result becomes a *controlled mechanism illustration*, not the headline.

**One-sentence thesis.** How reliably a frozen judge (reward model / LLM-as-judge) can *rank* behavior by a criterion is predicted by how well its representation can *read* that criterion — a quantity (out-of-sample R²) you can measure offline, with no drift and a small labeled set — and real production-scale judges already fail this test on criteria practitioners care about.

**Legend for evidence status:** **[HAVE]** = already run (this repo). **[RUN]** = proposed, scoped, cheap. **[HAVE*]** = have a controlled/tabular version; strengthen at scale.

---

## Abstract (draft beats)
Reward models and LLM judges are used less to *classify* than to *rank* — best-of-n, RLHF preference optimization, eval leaderboards, safe-policy selection. Their standard health check is accuracy, which is known to poorly predict downstream policy quality. We give a different, **offline** predictor: the **representation coverage** of a criterion — the out-of-sample R² of that criterion from the judge's frozen representation — predicts how reliably the judge can rank on it (Spearman **+0.75** across 26 criteria on a 3B LLM judge). Coverage exposes blind spots that accuracy hides: the same judge reads *factual correctness* at R² **0.20** and *hedging* at **0.01**, so selecting answers by those criteria is unreliable — no drift, no adversary. We add a **sign taxonomy**: whether optimizing a *readable* attribute helps or harms an *unreadable* target is set by the sign of their correlation through the representation, and we find a real **Goodhart** instance (lexical complexity vs. factual correctness) in a production judge. The same representation property governs a reward-posterior safety bound's **off-support certification**, unifying ranking and certification failures under one measurable cause. The practical shift is from "trust the accuracy dashboard" to "**audit per-criterion coverage; treat low-coverage criteria as unreliable to select on.**"

---

## 1. Introduction
- **The use case is selection, not classification.** Enumerate: best-of-n / rejection sampling, RLHF & DPO preference ranking, model-graded evals & leaderboards, Bayesian-REX-style safe-policy selection. All *rank* candidates by a learned judge.
- **The standard check (accuracy) is the wrong instrument.** Reward-model accuracy weakly predicts downstream policy quality [Wang et al. 2025]; a low-*variance* reward is a poor *teacher* even at perfect accuracy [Razin et al. 2025]. Both say: characterize the judge by more than accuracy.
- **Our axis: representation coverage.** A frozen judge ranks by projecting behavior into its representation; if that representation cannot *express* a criterion, no head on top can *rank* by it. We make this quantitative and offline.
- **Contributions:**
  1. **The coverage law** (§3): out-of-sample R² of a criterion from the frozen representation predicts ranking reliability (+0.75, 26 criteria, real 3B judge). **[HAVE]**
  2. **It is a law, not a tautology** (§4): holding coverage fixed while varying criterion nonlinearity/noise/base-rate leaves ranking reliability unchanged. **[RUN]**
  3. **A sign taxonomy** (§5): coverage magnitude says *whether* you can rank; the sign of the readable↔unreadable correlation says whether optimizing a readable proxy *helps or destroys* an unreadable target — with a real Goodhart instance. **[HAVE]**
  4. **Coverage vs. scale** (§6): the first coverage-by-model-size/family/layer sweep; does correctness coverage improve with scale, or plateau? Either answer is a result. **[RUN — headline]**
  5. **Coverage predicts downstream selection quality** on a real preference dataset (§7), and is **estimable from a small gold set** (§8). **[RUN]**
  6. **One property, two failures** (§9): the same coverage gap that breaks *ranking* also bounds which policies a safety bound can *certify* off-support. **[HAVE]**
- **Positioning up front:** we complement the beyond-accuracy line (Razin: reward *variance* → optimization; Wang: accuracy → downstream) with a distinct axis — representation *coverage* → *ranking & certification*. Cite Zhuang & Hadfield-Menell (proxy objectives over attribute subsets), Skalse et al. (STARC / order-preservation under misspecification), Ovadia et al. (UQ under shift), Murphy (calibration/discrimination decomposition) — see §10.

## 2. Setup and definitions (drift enters here only as motivation)
- **Judge / criterion / behavior.** A frozen representation φ(x) ∈ R^m; a criterion c is a target attribute of x; the judge ranks x by a head on φ.
- **Coverage** = out-of-sample R² of c regressed on φ (cross-fitted ridge; grouped by cluster to avoid leakage). Interpretable, offline, label-cheap.
- **Ranking reliability** = rank correlation between the judge's induced score (incl. its VaR safety bound) and the true criterion value over a candidate set.
- **The safety bound as the ranking instrument.** Define the reward-posterior VaR lower bound in two lines; it is the object practitioners rank by in safe selection. (Scale/offset invariance of ranking ⇒ coverage, not calibration, is the relevant property.)
- **Why criteria move (motivation, not thesis).** Evaluators re-weight over time (human satiation, LLM-judge rubric shift, a reward model reused as next-round grader). Drift is simply *the event that moves selection onto a criterion you did not audit*. The audit itself is drift-free.

## 3. The coverage law **[HAVE]**
- **Result:** across 26 criteria (6 interpretable attributes + 20 random combinations) on a frozen Qwen2.5-3B judge, coverage R² predicts ranking reliability at **Spearman +0.75** (Pearson +0.75).
- **Controls:** shuffled-label null R² ≈ −0.20 (no signal — a positive held-out R² would indicate overfitting; the negative value confirms none); oracle on true attributes ranks near-perfectly for most criteria (four interpretable attributes at 0.46–0.65), confirming the criteria are rankable in principle and only coverage limits the judge.
- **The blind spots accuracy hides:** correctness ≈ 0.20 and hedging ≈ 0.10 are poorly covered vs. specificity/formality ≈ 0.6–0.75 (length ≈ 0.53) — important criteria are the poorly-covered ones. *(Coverage values are representation-dependent; these are the standardized-embedding probes used throughout §3/§6/§8.)*
- **Fig 1:** the +0.75 scatter (already produced, `fig4_coverage_law.png`).

## 4. It is a law, not arithmetic **[RUN — A2, the tautology killer]**
- **Objection:** 20 of 26 criteria are combinations of the 6, so effective n ≈ 6; +0.75 could be partly definitional.
- **Design:** hold coverage ≈ constant and independently vary (i) criterion nonlinearity, (ii) label noise, (iii) base rate, (iv) sparsity; show ranking reliability does **not** move with these at fixed coverage, and **does** move with coverage at fixed everything-else. Report a partial-correlation / mixed-effects decomposition.
- **Deliverable:** turns +0.75 from "arithmetic" into "coverage is the operative variable."

## 5. The sign taxonomy: coverage says *whether*, sign says *which way* **[HAVE]**
- **Claim:** for a linear head, selecting on a readable attribute s moves an unreadable target t only through the regression E[t | s]; the dangerous case is low coverage of t **plus** a negative sign.
- **Three regimes** (positively correlated = proxy optimization drags the target along, why rich judges look robust; uncorrelated = pure missed opportunity; **negatively correlated = Goodhart**, optimizing the proxy destroys the target, no uncertainty flag fires).
- **Real instance:** on TruthfulQA the 3B judge is **Goodhart for lexical complexity (avg word length)** vs. factual correctness — top-10% by the judge's reading of "long words" is **0.45** correct vs. **0.50** base (CI excludes base, 5/5 refits), and the judge's *reading* of the attribute is more anti-correct (−0.10) than the attribute itself (−0.02). `length` and `digit_frac` are flat.
- **Honest magnitude caveat:** the effect is real but modest (~5 pts below chance), and it is lexical complexity, not raw verbosity.
- **Actionable:** the audit becomes "measure the **sign**, not only the R²." **Fig:** decile curve of correctness vs. readable-attribute readout + the three-regime table.

## 6. How coverage scales — and where it doesn't **[HAVE — Qwen2.5 {0.5,1.5,3,7,14}B; layer sweep on 3B]**
**Result: coverage scaling is criterion-specific — some criteria are threshold capabilities, others are flat walls, and *where you read the judge* matters as much as its size.**

- **Factual correctness is a late-rising threshold capability, not a flat wall.** Coverage (TruthfulQA, ridge R²): 0.5B **0.22** · 1.5B **0.20** · 3B **0.20** · 7B **0.30** · 14B **0.35** (MLP: 0.27/0.34/0.29/0.41/0.45). Flat/low ≤3B, then rises 7B→14B — small judges are *specifically* bad at reading correctness; scale helps but does not solve it (still 0.35 ≪ 0.9 for surface features).
- **Hedging and sentiment are flat walls at every scale** — hedging ~0.07–0.14, sentiment ~0.22–0.24 from 0.5B to 14B (28×). Scale buys nothing here. So the audit's value is exactly telling you *which* criteria are threshold-vs-wall for a given judge.
- **The coverage law holds at every scale** (Spearman +0.74/+0.77/+0.75/+0.73/**+0.69**) — not a one-model artifact. The drift-onto-correctness ranking failure is near scale-invariant (0.80→0.35 small; 0.86→0.42 at 14B — slightly milder but not fixed).
- **Layer depth matters — the mid-layer reads correctness better than the deployed final layer, at both scales (standardized, §6-consistent).** Correctness coverage peaks at ~0.6–0.75 depth and falls to the final layer: **3B 0.34 (depth 0.75) vs 0.20 final; 14B 0.43 (depth 0.62) vs 0.35 final** — and the final-layer numbers now match §6's semantic probe **exactly** (0.20 / 0.35), confirming consistency (surface attributes are ~0.9 at every layer — correctness is uniquely deep-but-not-final). So **part of the blind spot is a readout choice, not capacity** — the review's prediction, confirmed at both scales but honestly *milder* than the raw-embedding numbers first suggested: (i) *read at ~0.6–0.75 depth, not the last layer* buys **+0.14 R² at 3B (0.20→0.34, ~1.7×)** and **+0.08 at 14B (0.35→0.43, ~1.2×)** — the benefit **shrinks with scale** (a bigger model's final layer already reads correctness better); (ii) *depth roughly rivals scale*: a **3B judge read at 0.75 depth (0.34) ≈ a 14B judge read at its deployed final layer (0.35)** — near-parity, i.e. reading deeper is competitive with a 5× bigger judge (walked back from the raw claim of "beats"). **Fig:** `fig_scaling.png` + `fig_layers_combined.png` (3B, 14B, standardized).
- **Resolved / open follow-ups:** (a) **final-layer convention pinned** — `hidden_states[-1]` == `last_hidden_state` for Qwen (identical), so the 0.28-vs-0.20 gap was NOT pre/post-norm but **raw-vs-standardized embeddings**; the layer sweep now standardizes (matching §3/§6/§7) and the final layer reproduces §6 exactly; (b) the layer sweep's *surface-attribute* numbers are on the TruthfulQA corpus, NOT comparable to the alpaca scaling per-attr table — only the shared `is_correct` is cross-comparable; (c) CV error bars added to the coverage law (mean CV sd 0.029); (d) cross-family (Llama/Mistral/Gemma — gated, need HF token) and extended criteria (calibration, sycophancy, …) are natural expansions.
- Inference-only; `postdex` env, HF cache in place, `dbrown-gpu-grn` (≤7B) / `dbrown-gpu-np` A6000 (14B); the whole sweep was single-GPU-hours, not days.

## 7. Coverage predicts downstream selection **[HAVE — UltraFeedback, real data]**
**Result: the coverage law replicates on real human-rated data at *full strength* (Spearman +0.73), and the deployment best-of-4 metric is a modest, honest +0.27 that carries a sobering finding.**
- **Setup:** UltraFeedback (5000 prompts × 4 human-rated responses; 3B judge). 30 criteria (4 human aspects + 6 surface attrs + 20 signed combos; effective-independent ≈ 6.3). Coverage = grouped held-out R² on the PCA-128 space. Two selection-side metrics, both from a **preference-trained** reward posterior (fit on TRAIN prompts, evaluated on HELD-OUT TEST prompts, averaged over 5 splits): a *pooled rank-correlation* (judge score vs gold over all test responses — full-variance) and *best-of-4 win@1* scored as advantage over the per-criterion random floor (the floor varies 0.25–0.51 by tie structure).
- **The law (full-variance ranking): Spearman(coverage, pooled rank-corr) = +0.73** (full-E coverage: **+0.75**) — statistically indistinguishable from the *synthetic* coverage law (§3, +0.75). So the law is not a synthetic artifact: on real human-rated responses, how well the representation covers a criterion predicts how well a preference-trained judge ranks by it, at the same strength.
- **The deployment metric (best-of-4 top-identification): Spearman(coverage, win-advantage) = +0.27** (regret-reduction +0.26). Every criterion beats random; high-coverage criteria give the largest advantage.
- **Why the two differ — an honest, informative gap:** pooled rank-corr is *full-variance* (aligned with coverage → +0.73); best-of-4 win@1 is *top-identification* on 4 near-tied integer-rated candidates (coarser, noisier → +0.27). A criterion can be linearly legible yet its single top hard to pin among 4 (e.g. *sentiment*: coverage 0.16, pooled-rank +0.41, but a noisy best-of-4). Both point the same way; the gap is a property of the *task granularity*, not of the law.
- **The deployment punchline:** the 3B judge selects **surface style far better than human quality**, and coverage is why — it covers *formality/length/specificity* at R² ≈ 0.6–0.7 but *truthfulness/honesty* at only ≈ 0.23, and both selection metrics track that (truthfulness pooled-rank +0.39, win-advantage +0.15 vs surface pooled-rank ≈ +0.77 / win-advantage ≈ +0.30). A frozen judge is a better selector of *how an answer reads* than of *whether it's true* — predicted in advance by the audit. **Fig:** `fig_selection.png` (two panels: the law, and the deployment metric).

## 8. Estimating coverage in practice — the gold-set recipe **[HAVE — TruthfulQA, 3B]**
**Result: the audit's *decision* (which criteria are low-coverage) is cheap — ~500 labels; pinning each coverage R² precisely is dearer — ~1–2k; and there is a probe-dimension / label-budget tradeoff worth stating.**
- **Setup:** coverage estimated as held-out ridge R² on a PCA-128 frozen embedding (PCA fit unsupervised on the pool, no label leakage); each estimate on random gold sets of size n ∈ {50…4000}, 40 resamples; 7 criteria spanning coverage (surface length/specificity/formality ≈ 0.8 down to *is_correct* 0.14, hedging 0.15).
- **Ranking which criteria are low-coverage — the decision the audit drives — is cheap: ~500 labels** (Spearman of estimated-vs-full coverage order = **+1.00 at n=500**; noisy below, garbage at n≤100). This is the headline: you can tell *which* criteria your judge is blind to with a few hundred labels.
- **Estimating each criterion's R² to ±0.05 is dearer: n\* ≈ 1000–2000** (the lowest-coverage criteria — is_correct 0.14 and hedging 0.15 — need the most, up to ~4000). **Tiny gold sets (n ≤ 100–200) are worse than useless** — with fewer labels than the probe's effective complexity the held-out R² collapses to catastrophic negatives; never trust a coverage number from < ~200 labels.
- **Probe-dimension tradeoff (a real finding, not a caveat):** the probe must be dimension-reduced (a full ~2000-dim ridge over thousands of resamples is intractable *and* label-hungry). Reduction is mildly lossy on *weak-signal* criteria — *is_correct* reads 0.20 on full-E (§6), 0.14 on PCA-128, 0.07 on PCA-64 — so smaller probe = fewer labels but lower fidelity on exactly the hard criteria. PCA-128 keeps most of the correctness signal at a modest label cost; that's the recommended operating point. **Fig:** `fig_goldset.png` (convergence + rank-stability).
- **Remaining (optional):** degradation under *noisy* proxy labels and cross-domain transfer of the coverage estimate are natural extensions; the core "how many labels" recipe is established.

## 9. One property, two failures: coverage also governs off-support certification **[HAVE]**
- The reward-posterior safety bound has two uses — rank policies, and certify one. We show a single representation/data-coverage property governs **both**:
  - **Ranking** fails when the representation cannot cover the criterion (this paper's spine).
  - **Certification** stays honest for policies the comparison data covers (the deployed one), but an adversarial off-support policy is **under-covered even by a correctly-specified model** (coverage → ≈0.5 as the target moves), because on-policy data starves the directions it differs along.
- These are **orthogonal axes** (off-support cert failure appears with a correct representation; ranking failure appears on-support) — so "coverage" (of the criterion for ranking; of policy space for certification) is the unifying lens, not a single covariate-shift effect. **[HAVE — heldout_cert]**
- *Framing:* this is the section that recovers the old certification/ranking result as a **corollary of coverage**, per the review's "demote certification to two paragraphs."

## 10. When the criterion moves: drift as the silent transition (demoted) **[HAVE*]**
- Keep the drift story, tightly, as *why an unaudited low-coverage criterion becomes the one you rank on*: a controlled simulator shows a representation blind to a moving criterion sees its ranking invert (0.78 → −0.29) while its intervals *tighten* (confidently mis-ordered); the neural transfer (240 PEBBLE runs) shows the same gradient; the mechanism is a biased **mean** estimate, not collapsed uncertainty, so an "abstain-if-uncertain" defense misses it.
- **Explicitly scoped:** this needs a *deliberately blind or narrow* model; **rich judges show no drift-*specific* blind spot** — which is exactly why the deployable claim is the **static** coverage audit (§3–§8), and drift is the motivating occasion, not the load-bearing mechanism.
- Two controlled-mechanism appendices (not headline): (a) the endogenous loop drives grader weight into the reward model's null space, and a blind model then fails to select on null-space-loaded policies (two-step; not a graded dose-response); (b) wide eval pools degrade even a correct model's ranking under drift, and this is *not* a collection-breadth artifact — a bounded caveat on the audit.

## 11. Related work
- **Beyond-accuracy reward-model quality:** Razin et al. 2025 (reward *variance* → optimization landscape); Wang et al. 2025 (accuracy → downstream). Our axis (coverage → ranking/certification) is orthogonal and complementary.
- **Proxy objectives / attribute subsets:** Zhuang & Hadfield-Menell 2020 — optimizing a subset of attributes; we quantify *which* subset a representation can even express.
- **Reward misspecification & order preservation:** Skalse et al. (STARC) — make §9's ordering statement a corollary, or state the difference explicitly.
- **UQ under distribution shift:** Ovadia et al. 2019 — we show the failure is a *mean* bias, invisible to uncertainty, distinguishing it from the UQ-under-shift literature.
- **Calibration vs. discrimination:** Murphy's decomposition — our certification/ranking split is the reward-model instance; gives the dissociation a standard vocabulary.
- **Bayesian reward learning / VaR:** Brown et al. (Bayesian-REX, BROIL) — the safety bound we audit; our contribution is a *measurable precondition* (coverage) for its selection use.

## 12. Limitations
- Coverage is verified to govern *selection* directly; the optimization-side link is argued via reward variance (Razin), not separately measured here.
- The Goodhart instance is real but modest in magnitude; we claim existence of the regime in a production judge, not that it dominates.
- Linear-readout coverage is the primary probe; we report an MLP-readout coverage as a nonlinearity check but do not claim linear coverage is the only relevant notion.
- Drift results (§10) require a deliberately blind/narrow model and are positioned as mechanism, not deployment.

## 13. Bottom line
Accuracy tells you whether a judge is usually right; it does not tell you what the judge can **rank**. Representation coverage does, it is measurable offline from a small labeled set, and it already flags real blind spots (factual correctness) and a real Goodhart regime (lexical complexity) in a production-scale judge. **Audit per-criterion coverage before you select on a criterion** — and treat low-coverage criteria as unreliable to rank on, whether or not anything drifts.

---

## Appendix M — Mechanism: why an evaluator lands on a low-coverage criterion (Paper B, folded in)
*Decision: the former standalone "Paper B" is **subsumed here** as a mechanism appendix, not a separate paper — the causal dose-response it would need did not materialize (below), so it does not carry a standalone contribution, but it explains §10's premise and is worth including.*

The audit (§3–§8) is static: it says *which* criteria a judge cannot rank. This appendix explains *why an evolving evaluation standard tends to move onto exactly those criteria*, and pins the limits of that claim with controlled tabular experiments (256 arms, 6 features, linear reward posterior).

- **M.1 The loop drives the grader's weight into the reward model's null space.** When a policy is optimized against a reward model blind to a subspace N, it over-produces the readable features; a satiating grader down-weights what is over-produced; simplex renormalization moves the freed weight into N. Measured on the scale-invariant fraction ‖Π_N w‖/‖w‖ (raw ‖Π_N w‖ is confounded by L2 concentration): the closed-blind loop drives it from 0.73 → 0.86 while every control (full-loop, oracle-tracking, random policy) stays at the ≈0.71 random-direction null; seed-paired contrasts +0.13…+0.17, all 95% CIs exclude 0, at every drift rate. So endogenous drift systematically moves the standard toward what the reward model cannot read. **[HAVE — pushN]**
- **M.2 A model blind to N then fails to select there.** On an N-loaded eval pool, the blind model's ranking is ≈0 (regret ≈0.40) while a full model that reads N ranks it at ≈0.85 (regret ≈0.04). So the two-step chain holds: *the loop reaches N (M.1) and the blind model fails on N (M.2).* **[HAVE — pushN2]**
- **M.3 Honest limit — not a graded dose-response.** Within the blind arm, *more* push does **not** predict *more* damage (Spearman(push, N-rank) = +0.3, the wrong sign) — because a fully-blind model already fails on N regardless of where w sits (the harm is floored by static blindness). We therefore claim the **two-step qualitative** mechanism, **not** "the push causes graded harm." A genuine dose-response would need a *partially*-blind model (headroom); that is future work, and its absence is exactly why Paper B is an appendix, not a paper.
- **M.4 A bounded caveat on the audit (hammer).** Even a correctly-specified model's ranking degrades on a *geometrically wide* eval pool under drift (rank 0.98→0.89 as pool effective-rank goes 1.2→4.2), and this is **not** a collection-breadth artifact — broad vs. on-policy collection makes no difference (rescue −0.02 [−0.06,+0.02]). So the coverage audit's ranking guarantee is weakest for high-dimensional candidate pools; magnitude is task-dependent. **[HAVE — hammer_mechanism]**

---

## Statistical hygiene (A5) — done
- **Task-level interaction test (answers the review's #1 rejection reason: pseudoreplication).** Analysing the 240-run drift results at the *task* level (unit = task, n=10, Wilcoxon signed-rank on per-task drift-drops), the *blindness × drift* interaction is real, not a pooling artifact: at **moderate drift** (the headline), drift-blind vs full interaction = **+0.40 [+0.31,+0.49], p=0.001, 10/10 tasks**; generic-blind = +0.14, p=0.005, 9/10. At **strong drift** it weakens to marginal (drift-blind +0.12, p=0.08) — *consistent with the paper's own "ranking partially recovers at strong drift" mechanism*, and generic-blind stays significant (+0.17, p=0.003). Reported honestly: the effect is strongest and clearly significant at moderate drift, weaker at strong.
- **Coverage law significance:** Spearman +0.75 over 26 criteria has **permutation p < 0.0001** (20k permutations); shuffled-label null R² ≈ 0. Effective-independent criteria ≈ 6–7 (noted).
- **Coverage-estimate CIs:** mean 5-fold CV sd of the coverage R² = **0.029** (tight); the coverage-law figure now carries x-error bars (`fig_coverage_law_cv.png`).
- **Norm convention pinned:** `hidden_states[-1]` == `last_hidden_state` for Qwen (final-layer correctness identical, 0.20/0.35), so the earlier 0.28-vs-0.20 gap was **raw-vs-standardized embeddings**, not pre/post-norm; the layer sweep now standardizes (matching §3/§6/§7) and the standardized mid-layer peak (**0.34 / 0.43** ridge, 3B / 14B) still exceeds the final layer (0.20 / 0.35), so the readout-depth finding survives the correction.

## Build plan (maps to review's calendar)
| Section | Status | Work | Rough cost |
|---|---|---|---|
| §3 coverage law | **HAVE** | polish Fig 1 | — |
| §5 sign taxonomy | **HAVE** | one more figure (decile + regime table) | hours |
| §9 cert corollary | **HAVE** | prose only | — |
| §10 drift (demoted) | **HAVE** | condense to 1 section + 2 appendices | hours |
| §4 tautology control | **HAVE** | done — objection killed (dose-response +0.98; partial\|oracle +0.74) | — |
| §6 scaling + layer sweep | **HAVE** | done — correctness threshold-rising 0.20→0.35; mid-layer 0.34 vs last 0.20 (3B) | — |
| Appendix M (Paper B) | **HAVE** | folded from pushN/pushN2/hammer_mechanism | — |
| §7 downstream selection | **HAVE** | done — law replicates on real data +0.73 (pooled rank); best-of-4 +0.27; surface≫quality | — |
| §8 gold-set curves | **HAVE** | done — rank criteria ~500 labels; estimate R² ~1–2k; probe-dim tradeoff | — |
| align final-layer extraction | **RUN** | pre/post-norm reconcile in layer_sweep | hours |
| 14B layer sweep | **RUN** | lock the "3B-mid ≳ 14B-last" claim | ~30 min GPU |
| CV error bars on R² | **RUN** | multi-seed coverage CIs across the sweep | ~1 day |
| §5 A5 hygiene | throughout | mixed-effects, CIs, interaction tests, compute budget | threaded |

**Critical path to a submission:** the two long poles — §6 (scaling+layer) and §4 (tautology) — are **done**, and stronger than the plan assumed (criterion-specific scaling + a readout-depth finding, not a single flat-wall result). Remaining new work is §7 (downstream selection) and §8 (gold-set), ≤2 days each, plus a few short hardening runs. Well inside the September window.
