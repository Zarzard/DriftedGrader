# Paper A — Submission Outline (handoff)

*Clean section-by-section skeleton to draft from. Each section states **What / Why / How / Result-Verdict**. Citations are listed per section as `[Cite: …]`; a companion "Citation ledger" at the end marks each ✓confirmed / ✎corrected / ✗not-found from a web check. Numbers are source-verified (consistency sweep 2026-08-06). Working data + figures live in `dbrown-group1/zifan/llm_judge_drift/*` and `md_files/figures/*`.*

**Working title:** *Coverage: An Offline Audit of What a Frozen Judge Can Rank*
**Alt titles:** "Coverage, Not Accuracy: Auditing What a Reward Model Can Rank On"; "You Can't Rank What You Can't Read."
**One-line thesis:** How reliably a frozen judge ranks behavior by a criterion is predicted by how well its representation can *read* that criterion — a quantity measurable offline from a small labeled set — and real production-scale judges already fail this on criteria that matter.

---

## Abstract (write last; ~150–180 words)
- **What:** reward models / LLM-judges are used to *rank & select* (best-of-N, RLHF, leaderboards, safe-policy selection), but are checked with *accuracy*, which poorly predicts ranking quality.
- **We propose** representation **coverage** — the out-of-sample R² of a criterion from the judge's frozen representation — as an offline predictor of ranking reliability.
- **Results (lead with these):** coverage predicts ranking reliability at **Spearman +0.75** across 26 criteria on a 3B judge (perm p<1e-4); it is a *law, not arithmetic* (within-criterion dose-response +0.98; partial +0.74 controlling for confounds); it grounds out on **real** human-rated selection (+0.73 pooled-rank on UltraFeedback); it exposes blind spots accuracy hides (a 3B judge covers *factual correctness* at only ≈0.20 vs *surface style* ≈0.6–0.8) and predicts that judges **select style far better than substance**; the audit needs ≈500 labels to flag which criteria are unreliable.
- **Takeaway:** *audit per-criterion coverage before you select on a criterion.*

---

## §1 Introduction
- **What (framing):** judges are used to *select*, not classify; the field's health check is accuracy, and accuracy is known to be insufficient. `[Cite: Wen-reward-accuracy, Razin-good-teacher]`
- **Why (the gap):** a judge ranks by projecting behavior into a frozen representation and scoring on top; whatever the representation discards, no head can recover. So a judge's *rankable* criteria are bounded by what it can *read* — an axis accuracy doesn't measure.
- **How (this paper):** define coverage; establish coverage→ranking as a law (and prove it's not a tautology); scale it, ground it on real selection, turn it into a recipe, and recover the safety-bound certification/ranking split as a corollary.
- **Contributions (numbered):** (1) the coverage law; (2) predictive-validity (dose-response + partial); (3) a sign taxonomy (Goodhart); (4) scaling + readout-depth; (5) real-selection grounding; (6) the gold-set recipe; (7) coverage unifies ranking- and certification-failure.
- **Positioning up front:** the surface≫substance *observation* is known (length/style bias); our contribution is a *continuous, offline, per-criterion predictor* that also predicts ranking on criteria the observation says nothing about. `[Cite: Singhal-length, Zheng-mtbench]`
- **Cite:** use-cases — best-of-N/summarization `[Cite: Stiennon-summarize]`, RLHF `[Cite: Ouyang-instructgpt or Christiano-deep-rlhf]`, DPO `[Cite: Rafailov-dpo]`, LLM-as-judge & leaderboards `[Cite: Zheng-mtbench, Lambert-rewardbench]`, safe selection `[Cite: Brown-bayesianrex]`.

## §2 Setup and definitions
- **What:** define judge, criterion, behavior; **coverage = held-out ridge R² of the criterion from the frozen (standardized) representation**; **ranking reliability = rank-corr of the judge's score vs the true criterion**; the reward-posterior **safety bound** (the object used in safe selection). Drift appears *only* as motivation for why the criterion one ranks on can shift.
- **Why:** coverage is a linear probe *on purpose* — a linear readout is what the judge's head can exploit; and R² (variance-explained) is the natural "how much of the criterion is in there" scale. `[Cite: Alain-bengio-probes]`
- **How:** ridge R² with an inner λ-selection, out-of-sample (in-sample R²→1 trivially with high-dim features); a shuffled-label null (≈ negative, no overfitting); an MLP-probe reported as a nonlinearity check.
- **Result/verdict:** definitions are operational and offline; everything downstream is measurable before deployment.
- **Cite:** Bayesian-REX safety bound `[Cite: Brown-bayesianrex, Brown-broil]`; performative/endogenous drift as motivation `[Cite: Perdomo-performative, Hardt-performative-power]`.

## §3 The coverage law
- **What:** across many criteria, coverage predicts ranking reliability.
- **Why:** the core claim — reading a criterion is the precondition for ranking on it.
- **How:** 26 criteria (6 interpretable attributes + 20 random combinations) on a frozen Qwen2.5-3B judge; coverage R² vs the judge's VaR-bound rank. `[Cite: Qwen25]`
- **Result/verdict:** **Spearman +0.75** (Pearson +0.75), **permutation p < 1e-4**; shuffled-null R² ≈ −0.20 (no overfitting); oracle-with-true-attributes ranks near-perfectly (rankable in principle); **CV sd 0.029** (tight). Blind spots: correctness ≈0.20, hedging ≈0.10 vs specificity/formality ≈0.6–0.75. **Fig:** `fig_coverage_law_cv.png`.

## §4 It is a law, not arithmetic (predictive validity)
- **What:** rule out that coverage merely re-encodes a known axis (e.g., surface-vs-substance) — i.e., that +0.75 is a bookkeeping artifact of 26 criteria built from 6 attributes.
- **Why:** the decisive credibility gate; a reviewer will raise "any high-surface/low-substance metric would correlate."
- **How:** (a) **dose-response** — continuously lower one criterion's coverage while its identity is fixed and an oracle still ranks it; (b) **confound bank** — 140 constructed criteria (linear/nonlinear/thresholded), partial correlation controlling for intrinsic rankability, nonlinearity, base rate.
- **Result/verdict:** dose-response Spearman(coverage, ranking) = **+0.98** (a static label is *constant* along the curve, so cannot predict the falling ranking); bank **+0.78**, **partial +0.74** (|oracle) / **+0.76** (|oracle+nonlinearity+base-rate); confounds alone don't predict (nonlin −0.52, base +0.29). So coverage is the *sufficient statistic*. **Fig:** `fig_tautology.png`. **Cite:** reward hacking / misspecification `[Cite: Skalse-reward-hacking]`.

## §5 The sign taxonomy: coverage says *whether*, sign says *which way*
- **What:** for a linear head, optimizing a readable attribute moves an unreadable target only through their correlation; the dangerous case is low coverage **plus a negative sign** (Goodhart).
- **Why:** turns the audit from "can you rank X?" into "will optimizing readable Y help or hurt unreadable X?"
- **How:** three regimes (positive = drag-along; zero = missed opportunity; negative = Goodhart); measured on TruthfulQA with cross-fitted readouts and cluster-bootstrap CIs.
- **Result/verdict:** a **real Goodhart instance** — lexical complexity (avg word length) vs factual correctness (top-decile correctness 0.45 < base 0.50, CI excludes base); honest caveat: **modest magnitude** (~5 pts). `[Cite: Goodhart, Razin-good-teacher]` (coverage↔reward-variance link). **Fig:** `fig_sign_taxonomy.png` (judge's top-10% correctness by selected attribute; word-length below chance = Goodhart).

## §6 How coverage scales — and where it doesn't
- **What:** does coverage of a criterion improve with model size and with *where* you read the model?
- **Why:** distinguishes a *capacity* limit from a *readout* choice; and tells you the size threshold per criterion.
- **How:** Qwen2.5 {0.5,1.5,3,7,14}B frozen judges (inference-only); + a layer-depth sweep on 3B & 14B (standardized, matching §3). `[Cite: Qwen25]`
- **Result/verdict — criterion-specific:**
  - **Correctness is a late-rising threshold capability:** 0.22/0.20/0.20/0.30/**0.35** (0.5→14B) — flat ≤3B, rises 7B→14B; still ≪ surface (~0.8).
  - **Hedging & sentiment are flat walls:** ~0.10 / ~0.22 across 28× — scale buys nothing.
  - **The law holds at every scale:** ρ +0.74/+0.77/+0.75/+0.73/+0.69.
  - **Readout depth matters (survives standardization, milder):** mid-layer > deployed final at both scales — 3B **0.34 @0.75 depth vs 0.20 final** (+0.14, ~1.7×); 14B **0.43 @0.62 vs 0.35** (+0.08, ~1.2×); benefit shrinks with scale; a 3B judge read at 0.75 depth (0.34) is *on par* with a 14B judge at its deployed final layer (0.35). `[Cite: Tenney-bert-pipeline, Alain-bengio-probes]` (mid-layers carry more decodable semantic content). **Figs:** `fig_scaling.png` (correctness vs size), `fig_scaling_criterion.png` (criterion-specific: correctness rises while hedging/sentiment stay flat walls), `fig_layers_combined.png` (readout depth). TruthfulQA `[Cite: Lin-truthfulqa]`.

## §7 Coverage predicts real downstream selection
- **What:** does coverage predict *real* best-of-N selection quality on human-rated data (not just synthetic rank-corr)?
- **Why:** deployment grounding — closes "is coverage practically predictive?"
- **How:** UltraFeedback (5000 prompts × 4 human-rated responses; helpfulness/honesty/instruction-following/truthfulness). 30 criteria (4 aspects + 6 surface + 20 signed combos; effective-independent ≈6.3). Coverage on PCA-128; a *preference-trained* posterior selects best-of-4 on held-out prompts (two metrics: full-variance pooled rank-corr; deployment best-of-4 above the per-criterion random floor). `[Cite: Cui-ultrafeedback]`
- **Result/verdict:** **the law replicates on real data: Spearman(coverage, pooled rank-corr) = +0.73** (full-E +0.75) — indistinguishable from synthetic §3; deployment best-of-4 advantage = **+0.27** (coarser: top-of-4 identification vs full-variance coverage). Within-*mix* (no surface/quality label) = **+0.84**. **Deployment punchline:** the judge selects **surface style ≫ human quality** (surface pooled-rank ≈+0.77 vs truthfulness +0.39), and coverage predicts it in advance. **Fig:** `fig_selection.png`, `fig_coverage_selection_categories.png`. `[Cite: Singhal-length, Dubois-lc-alpacaeval]`.

## §8 Estimating coverage in practice — the gold-set recipe
- **What:** how many labeled examples to audit a criterion's coverage?
- **Why:** turns coverage from a lab result into a recipe.
- **How:** coverage estimate on random gold sets n∈{50…4000}, 40 resamples, PCA-128 probe, 7 criteria spanning the range (TruthfulQA 3B).
- **Result/verdict:** **ranking which criteria are low-coverage (the audit's decision) is cheap — ~500 labels** (order-Spearman +1.0); **estimating each R² to ±0.05 is dearer — ~1–2k**; **n≤200 is unusable** (estimate collapses to catastrophic negatives). **Probe-dimension tradeoff** (a finding): smaller probe = fewer labels but lower fidelity on weak criteria (is_correct 0.20 full-E / 0.14 PCA-128 / 0.07 PCA-64) — PCA-128 recommended. **Fig:** `fig_goldset.png`.

## §9 One property, two failures: coverage also governs off-support certification
- **What:** the reward-posterior safety bound has two uses — rank policies, certify one — and a single coverage property governs both.
- **Why:** recovers the (older) certification-vs-ranking dissociation as a *corollary* of coverage, and demotes it from thesis to two paragraphs.
- **How:** held-out certification in the simulator (certify never-deployed policies off the comparison support), across correct/blind representations and drift levels.
- **Result/verdict:** **ranking** fails under representation blindness (criterion-coverage); **certification** stays honest for on-support policies (incl. the deployed one) but an adversarial off-support policy is under-covered **even by the correct model** (coverage → ≈0.5 in the well-specified block as the target moves). **Orthogonal axes:** off-support cert-failure appears with a correct representation; ranking-failure appears on-support — so it's a two-axis (coverage-in-criterion-space vs coverage-in-policy-space) phenomenon, not one covariate-shift effect. **Frame precisely:** the "≈0.5" is the well-specified (k=d) block. `[Cite: Brown-bayesianrex, Skalse-starc, Ovadia-uncertainty-shift, Murphy-calibration]` (calibration/discrimination decomposition as the vocabulary for the split).

## §10 Related work
- **Beyond-accuracy reward-model quality:** reward *variance* → optimization landscape `[Cite: Razin-good-teacher]`; accuracy weakly predicts downstream `[Cite: Wen-reward-accuracy]`. Our axis (coverage → ranking/certification) is orthogonal and complementary.
- **Reward over-optimization / Goodhart:** `[Cite: Gao-overopt-scaling, Skalse-reward-hacking]`.
- **Proxy objectives over attribute subsets:** `[Cite: Zhuang-consequences]`.
- **Reward-function distance / order preservation:** `[Cite: Skalse-starc]` (state whether §9 is a corollary).
- **Uncertainty under shift:** `[Cite: Ovadia-uncertainty-shift]` — we show the failure is a *mean* bias, invisible to UQ.
- **Calibration vs discrimination:** `[Cite: Murphy-calibration]` — the certification/ranking split is the reward-model instance.
- **Bayesian reward learning / VaR:** `[Cite: Brown-bayesianrex, Brown-broil, Brown-trex]` — the bound we audit.
- **Preference-based RL systems:** `[Cite: Lee-pebble, Lee-bpref]`; passive-vs-active `[Cite: Ostrovski-passive]`.
- **LLM-judge biases / eval:** `[Cite: Zheng-mtbench, Singhal-length, Dubois-lc-alpacaeval, Lambert-rewardbench]`.
- **Non-stationary / performative:** `[Cite: Perdomo-performative, Hardt-performative-power]` (drift motivation).
- **Probing / linear representations:** `[Cite: Alain-bengio-probes, Tenney-bert-pipeline]`.

## §11 Limitations
- Coverage is verified to govern *selection*; the optimization-side link is argued (Appendix M / via reward variance), not separately measured.
- The Goodhart instance is real but modest; we claim regime-existence, not dominance.
- Linear-readout coverage is primary; an MLP-probe is a nonlinearity check, not a claim that linear coverage is the only notion.
- §7 best-of-N is capped at 4 (dataset); pooled rank-corr complements it. UltraFeedback aspects are mutually correlated (effective ≈2) — mitigated by adding surface criteria.
- Within-Qwen scaling; cross-family (Llama/Mistral/Gemma) deferred (gated). Drift results (Appendix M) require a deliberately blind/narrow model — mechanism, not deployment.

## §12 Conclusion
- Accuracy tells you whether a judge is usually right; it does not tell you what it can **rank**. Coverage does, offline, from a small labeled set — and it already flags real blind spots (correctness) and a real Goodhart regime (lexical complexity) in a production-scale judge. **Audit per-criterion coverage before selecting on a criterion.**

## Appendix M — Mechanism: why an evaluator lands on a low-coverage criterion (Paper B, folded in)
- **What/why:** the audit is static; this explains *why an evolving standard moves onto low-coverage criteria*, with controlled tabular experiments. (Subsumed here; not a standalone paper — the causal dose-response didn't materialize.)
- **M.1** the closed loop drives grader weight into the reward model's null space (‖Π_N w‖/‖w‖ 0.73→0.86 vs controls at the 0.707 null; seed-paired CIs exclude 0).
- **M.2** a model blind to N then fails to select there (blind rank_N ≈0 / regret ≈0.40 vs full 0.90 / 0.04).
- **M.3** honest limit — *not* a graded dose-response (push doesn't predict extra damage; harm floored by static blindness).
- **M.4** wide-eval-pool degradation under drift is real even for the correct model, and *not* a collection-breadth artifact.
- **Statistical hygiene (belongs in the main neural results or here):** task-level interaction test (unit=task, n=10, Wilcoxon) — drift-blind×drift **+0.40, p=0.001, 10/10 tasks at moderate drift**; generic-blind significant at both; drift-blind marginal at strong (consistent with recovery). `[Cite: Perdomo-performative]` (endogenous drift).

---

## Citation ledger (web-verified 2026-08-06)
*Every reference below was checked against the web. **30/30 exist; 2 were mis-cited and are corrected here** — do not use the wrong forms. Author lists abbreviated with "et al." where long.*

**⚠ Corrections (I had these wrong — use the corrected form):**
- `Wen-reward-accuracy` — **NOT "Wang."** Xueru **Wen**, Jie Lou, Yaojie Lu, et al. *"Rethinking Reward Model Evaluation: Are We Barking up the Wrong Tree?"* **ICLR 2025 (Spotlight)**, arXiv:2410.05584. (First author is Wen; the finding — accuracy weakly predicts downstream policy performance — matches.)
- `Brown-bayesianrex` — **authors are Brown, Coleman, Srinivasan, Niekum — NOT "Goo."** Daniel S. Brown, Russell **Coleman**, Ravi **Srinivasan**, Scott Niekum. *"Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences"* (Bayesian REX). **ICML 2020**, arXiv:2002.09089. (Goo is a co-author on T-REX 2019, not this one.)

**Confirmed references:**
- `Razin-good-teacher` — Razin, Wang, Strauss, Wei, Lee, Arora. *"What Makes a Reward Model a Good Teacher? An Optimization Perspective."* NeurIPS 2025, arXiv:2503.15477.
- `Gao-overopt-scaling` — Gao, Schulman, Hilton. *"Scaling Laws for Reward Model Overoptimization."* ICML 2023, arXiv:2210.10760.
- `Skalse-reward-hacking` — Skalse, Howe, Krasheninnikov, Krueger. *"Defining and Characterizing Reward Hacking."* NeurIPS 2022, arXiv:2209.13085.
- `Skalse-starc` — Skalse, Farnik, Motwani, Jenner, Gleave, Abate. *"STARC: A General Framework for Quantifying Differences Between Reward Functions."* ICLR 2024, arXiv:2309.15257. *(distinct from arXiv:2410.05584, which is Wen above.)*
- `Zhuang-consequences` — Zhuang, Hadfield-Menell. *"Consequences of Misaligned AI."* NeurIPS 2020, arXiv:2102.03896.
- `Brown-broil` — Brown, Niekum, Petrik. *"Bayesian Robust Optimization for Imitation Learning"* (BROIL). NeurIPS 2020, arXiv:2007.12315.
- `Brown-trex` — Brown, Goo, Nagarajan, Niekum. *"Extrapolating Beyond Suboptimal Demonstrations via IRL from Observations"* (T-REX). ICML 2019, arXiv:1904.06387.
- `Lee-pebble` — Lee, Smith, Abbeel. *"PEBBLE: Feedback-Efficient Interactive RL via Relabeling Experience and Unsupervised Pre-training."* ICML 2021, arXiv:2106.05091.
- `Lee-bpref` — Lee, Smith, Dragan, Abbeel. *"B-Pref: Benchmarking Preference-Based Reinforcement Learning."* NeurIPS 2021 Datasets & Benchmarks, arXiv:2111.03026.
- `Ostrovski-passive` — Ostrovski, Castro, Dabney. *"The Difficulty of Passive Learning in Deep Reinforcement Learning."* NeurIPS 2021.
- `Zheng-mtbench` — Zheng, Chiang, Sheng, Zhuang, Wu, et al. *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."* NeurIPS 2023 Datasets & Benchmarks.
- `Singhal-length` — Singhal, Goyal, Xu, Durrett. *"A Long Way to Go: Investigating Length Correlations in RLHF."* arXiv:2310.03716 (later COLM 2024).
- `Dubois-lc-alpacaeval` — Dubois, Galambosi, Liang, Hashimoto. *"Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators."* arXiv:2404.04475.
- `Cui-ultrafeedback` — Cui, Yuan, Ding, Yao, He, et al. *"UltraFeedback: Boosting Language Models with Scaled AI Feedback."* ICML 2024.
- `Lin-truthfulqa` — Lin, Hilton, Evans. *"TruthfulQA: Measuring How Models Mimic Human Falsehoods."* ACL 2022.
- `Lambert-rewardbench` — Lambert, Pyatkin, Morrison, Miranda, Lin, et al. *"RewardBench: Evaluating Reward Models for Language Modeling."* 2024.
- `Alain-bengio-probes` — Alain, Bengio. *"Understanding Intermediate Layers Using Linear Classifier Probes."* ICLR 2017 Workshop, arXiv:1610.01644.
- `Tenney-bert-pipeline` — Tenney, Das, Pavlick. *"BERT Rediscovers the Classical NLP Pipeline."* ACL 2019.
- `Ovadia-uncertainty-shift` — Ovadia, Fertig, Ren, Nado, Sculley, Nowozin, Dillon, Lakshminarayanan, Snoek. *"Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift."* NeurIPS 2019.
- `Perdomo-performative` — Perdomo, Zrnic, Mendler-Dünner, Hardt. *"Performative Prediction."* ICML 2020.
- `Hardt-performative-power` — Hardt, Jagadeesan, Mendler-Dünner. *"Performative Power."* NeurIPS 2022, arXiv:2203.17232.
- `Rafailov-dpo` — Rafailov, Sharma, Mitchell, Ermon, Manning, Finn. *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model."* NeurIPS 2023.
- `Murphy-calibration` — Murphy. *"A New Vector Partition of the Probability Score."* J. Applied Meteorology 12(4):595–600, 1973. (companion: DeGroot & Fienberg 1983, *"The Comparison and Evaluation of Forecasters"*.)
- `Qwen25` — Qwen Team. *"Qwen2.5 Technical Report."* arXiv:2412.15115 (Dec 2024).
- `Christiano-deep-rlhf` — Christiano, Leike, Brown, Martic, Legg, Amodei. *"Deep Reinforcement Learning from Human Preferences."* NeurIPS 2017, arXiv:1706.03741.
- `Ouyang-instructgpt` — Ouyang, Wu, Jiang, et al. *"Training Language Models to Follow Instructions with Human Feedback"* (InstructGPT). NeurIPS 2022.
- `Stiennon-summarize` — Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, Christiano. *"Learning to Summarize from Human Feedback."* NeurIPS 2020.
- `Goodhart` (→ use `Goodhart-manheim`) — Manheim, Garrabrant. *"Categorizing Variants of Goodhart's Law."* arXiv:1803.04585 (2018).

*Notes:* arXiv IDs are included where the web check surfaced them; a few venue-only entries omit the ID (confirmed via proceedings, not arXiv). `Saito-verbosity` was proposed but **removed** (not web-verified and redundant with Singhal/Dubois/Zheng). Consider adding, if desired (verify first): RLHF-poisoning (Rando & Tramèr), and a best-of-N/rejection-sampling reference beyond Stiennon.
