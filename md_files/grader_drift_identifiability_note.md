<!-- Transcribed 2026-08-05 from user-provided attachment; encoding (mojibake) fixed, content otherwise verbatim. -->

# What pairwise preference data can and cannot see of a drifting grader

*Working note, v0.1 — August 2026*

**Summary.** A rater is modeled as criterion weights + a severity offset + a discriminability level, drifting over time. Five short facts follow: (1) severity cancels inside genuine comparisons but survives in score-derived preference pairs; (2) the only identifiable rater state is the product v_t = beta_t.w_t, so "drift" must mean drift of v_t; (3) no statistic computed from the operational preference stream alone can attribute change to the rater versus to the pairs — attribution requires fixed anchor pairs; (4) an anchor battery is a linear sensor for v_t, which makes anchor design an experiment-design problem and yields a concrete rule: direction drift is best detected by ~50/50 anchors with spread-out gap directions, norm drift by ~80/20 anchors, and near-0/100 anchors are useless; (5) anchors robustly *detect* rater-side change even if the rater's internal representation drifts, but *attributing* drift to criterion weights requires assuming a fixed feature map — the one assumption in this note that does irreducible work.

Nothing below is mathematically deep; every proof is a few lines of algebra. The point is that these small facts fix the experimental design — which anchors to build, at what difficulty, what the estimand is, what the Bayesian estimator's coverage target should be — by derivation rather than by taste. They also settle three objections from the internal review: the severity-drift evidence-transfer problem (§2), the "drift statistics just measure rising pair difficulty" confound (§4), and the "anchors are support-limited" objection, which becomes a computable number instead of rhetoric (§5a).

---

## 1. Model and scope

A rater's utility for item x at time t:

> u_t(x) = w_t . phi(x) + s_t

- **phi : X -> R^d** — feature map, unknown, fixed over time. The substantive content is not linearity (phi is arbitrary) but that rater change is confined to a fixed d-dimensional space of criteria: whatever the rater's standards do over time, they move within span(phi) plus a constant.
- **w_t in R^d** — criterion weights (how much helpfulness vs. style vs. safety, etc.).
- **s_t in R** — severity offset: overall strictness. This is the parameter the many-facet Rasch / rater-severity literature documents drifting.
- **beta_t >= 0** — discriminability: choice consistency.

Labels follow Bradley-Terry:

> P_t(x > y) = sigma( beta_t . [u_t(x) - u_t(y)] ),  sigma(z) = 1/(1+e^{-z}).

Write **dphi = phi(x) - phi(y)** and sigma'(z) = sigma(z)(1-sigma(z)).

**Scope of a "comparison":** both items are judged by the *same rater state* — same rater, same sitting. The population-of-raters extension is Remark 6.1. The logistic link is inessential: §2-§3 need only that the label law depends on the utility difference through some fixed strictly increasing link; §4-§5 carry over with sigma replaced by that link (specific numbers like "p ~ 0.82" are link-specific; the probit analogue is p ~ 0.84).

---

## 2. Severity cancels — but only inside genuine comparisons

**Proposition 1.** The label law of any within-rater, same-sitting comparison does not depend on s_t. Consequently the likelihood of such a dataset is invariant to *arbitrary* drift of s_t.

*Proof.* u_t(x) - u_t(y) = w_t.phi(x) + s_t - w_t.phi(y) - s_t = w_t.dphi. The offset is gone before the link is applied. QED

**Corollary 1 (contamination).** If a preference pair is *synthesized* from absolute scores collected at different times t != t' (or from different raters A != B), the relevant difference is w_t.phi(x) - w_{t'}.phi(y) + (s_t - s_{t'}), and the severity difference survives. Such pairs are severity-contaminated.

This sorts preference datasets into two columns:

| Severity-immune (simultaneous comparison) | Severity-contaminated (score-derived / cross-context) |
|---|---|
| Standard RLHF side-by-side annotation (one annotator, both responses on screen) | Pairs built from Reddit/forum karma differences (different voters, different times — e.g. SHP-style constructions) |
| Arena-style battles (same user, same session) | Stack-Exchange-style vote-differential pairs |
| Simulated teachers (B-Pref etc.), trivially | Pairs built from star/Likert ratings collected at different times |
| — | Cross-annotator pairs (rater A's score vs. rater B's score) |

*(The specific dataset citations for the right-hand column come from the literature sprint and go into Paper 1's version of this table.)*

**Verdict on the evidence-transfer objection.** The review objected that documented rater-severity drift (Congdon & McQueen; MFRM work) might not transfer to comparison designs. Proposition 1 says: it does not transfer — severity drift is invisible to genuine comparisons *by construction*, and this is plausibly the deep reason comparison designs exist. Corollary 1 says: it transfers fully to score-derived preference datasets. Both directions of the repair are now derived, not argued. What genuine comparisons remain exposed to is drift in **w_t** (criterion reweighting) and **beta_t** (consistency) — which is exactly what the rest of the note is about.

---

## 3. What drift is identifiable: v_t = beta_t.w_t

**Proposition 2.** The label law depends on (beta_t, w_t, s_t) only through **v_t := beta_t.w_t**:

> P_t(x > y) = sigma(v_t . dphi).

So the identifiable rater state is v_t, and "grader drift," operationally, means drift of v_t. It splits into

- **direction** what_t = v_t/||v_t|| — the criterion mix (rotation on the sphere), and
- **norm** ||v_t|| — effective discriminability.

*Proof.* beta_t(w_t.dphi) = (beta_t w_t).dphi. Nothing else enters. QED

**Remark 3.1 (scale split is impossible).** beta_t vs. ||w_t|| ("rater got noisier" vs. "criteria weakened uniformly") is never separable from choice data: any factorization of v_t gives the same likelihood. Separating them needs a side channel — response times, or occasional absolute ratings (which carry s_t and their own noise). Do not claim it from comparisons.

**Remark 3.2 (gauge).** (w, phi) -> (A^{-T}w, A.phi) for invertible A changes nothing observable. Harmless as long as A is fixed in time; a time-varying representation is the subject of §6.

**Consequence for the Bayesian plan.** Coverage claims should target v_t (or its anchor projection, §5) — coverage statements about beta, w, s separately are vacuous by construction.

---

## 4. The operational stream cannot attribute change to the rater

Operational pairs at time t are drawn from the current policy's output distribution; write g = dphi(x, y) for the (unobserved) feature gap of a presented pair.

**Proposition 3.** Given any presented pair, the label law depends only on the scalar **z = v_t . g**. Since phi is unknown, the analyst observes prompts, responses, and labels — never g. Therefore: (i) v_t enters the likelihood of the operational data only through the scalars {v_t.g}; (ii) the two worlds "rater scale c.v_t, gaps g" and "rater scale v_t, gaps c.g" induce identical label laws on every presented pair. Rater scale and pair-gap magnitude are confounded, exactly.

*Proof.* (i) is Proposition 2 applied per pair. (ii): (c.v).g = v.(c.g). QED

**Consequence (the stationarity null, derived).** As the policy improves, comparison pairs concentrate near each other in quality, so typical |z| shrinks. Then inter-rater agreement falls, flip rates on repeated presentation rise (P(flip) = E[2 sigma(z)(1-sigma(z))]), and any fitted Bradley-Terry temperature rises — **under an exactly constant rater**. No consistency statistic computed from the operational stream alone (this includes the kappa-hat-style diagnostics) can attribute such change to the rater rather than to the pairs. This was a pre-registered null hypothesis in the experimental plan; it is now a theorem, and the corresponding experiment (§7, first run) is a *validation of the instrument's necessity*, not a test that could fail.

The escape from Proposition 3 is external control of g: pairs whose gap is *fixed by construction* across time. That is all an anchor is.

---

## 5. Anchors are linear sensors for v_t

Fix anchor pairs j = 1, ..., m with constant content. Then each anchor's log-odds is exactly linear in the rater state:

> z_j(t) = v_t . dphi_j,  p_j(t) = sigma(z_j(t)),

and the battery is a linear measurement through a fixed (unknown) design matrix Phi_A = [dphi_1; ...; dphi_m]:

> z(t) = Phi_A . v_t.

### (a) Detection condition, and the support-limitation objection as a number

A drift step delta = v_{t'} - v_t is visible to the battery iff Phi_A.delta != 0, i.e. iff delta has a component in span{dphi_j}. With m anchors the battery sees at most a rank-m projection of drift. Define the **on-support fraction**

> rho(delta) = ||P_A delta||^2 / ||delta||^2,  P_A = projection onto span{dphi_j}.

In simulation delta is known, so rho is computable — report it for every run. The reviewer's "anchors are support-limited" objection stops being rhetoric and becomes a per-experiment scope statement: anchor-based conclusions are conclusions about P_A.v_t, with rho saying how much of the truth that was.

### (b) Anchor selection is experiment design

Per label on anchor j, the Fisher information for v_t is sigma'(z_j).dphi_j dphi_j^T. Choosing which anchors to include, and how many labels n_j to spend on each, is a D-optimal design problem: maximize det Sum_j n_j sigma'(z_j) dphi_j dphi_j^T. At Tier-A scale a greedy solver is fine.

### (c) The split rule

**Norm sensitivity.** Perturb the norm: v -> (1+eps)v. Then dz_j = eps.z_j and

> dp_j = eps . z_j sigma'(z_j).

|z.sigma'(z)| is maximized at |z*| ~ 1.543 (the root of z(2 sigma(z)-1) = 1), i.e. **p ~ 0.82**, and is *zero* at p = 0.5: a 50/50 anchor cannot see scale change at all, since scaling does nothing at z = 0.

**Direction sensitivity.** Perturb the direction: v -> v + dv with dv perp v. Then

> dp_j = sigma'(z_j) . (dphi_j . dv).

The prefactor sigma'(z_j) is maximized at z_j = 0, i.e. **p = 0.5** — and one further needs the gap directions dphi_j to be spread out, so that no rotation direction hides in the battery's null space (this is also what keeps cond(Phi_A) small).

**Saturated anchors** (p near 0 or 1) have sigma' ~ 0: near-zero information for everything. Dead weight.

**Interpretation.** A 50/50 anchor is one the rater is currently indifferent about — dphi_j perp v_t — hence maximally sensitive to any change in what the rater cares about and blind to how sharply they care. An 80/20 anchor is the reverse. So the battery has two panels: a **direction panel** (~50/50, directions spread) and a **norm panel** (~80/20, both polarities, aligned with current criteria). The folk rule "anchors should be hard pairs" is exactly half right.

### (d) Real data, unknown Phi_A

With human raters dphi_j is unknown, but z(t) = Phi_A v_t is still observed (through sigma). Norm-type and angle-type summaries of the trajectory *in anchor coordinates* remain meaningful readouts of v_t's norm and direction changes, with distortion controlled by the condition number of Phi_A — one more argument for spreading directions. This is what gets computed on the natural-experiment data (frozen eval set re-graded across LLM-judge version bumps).

---

## 6. What anchors certify, and what they cannot

**Proposition 5.** If anchor *content* is fixed, then any change in an anchor's label law implies something rater-side changed — some combination of w, beta, the link, or phi restricted to those items. Detection is therefore robust even to drift of the rater's internal representation. The converse fails: an unchanged battery does not imply no drift (any delta in the null space of Phi_A is invisible, §5a). And *attribution* is not robust: (w_t, phi fixed) and (w fixed, phi_t) can produce identical anchor trajectories, so separating criterion-weight drift from representation drift requires the fixed-phi assumption.

*Proof.* The anchor's inputs are constant; the only remaining arguments of the label law are rater-side. The attribution claim: for any battery trajectory z(t), pick phi_t along span{dphi_j} to reproduce it with w constant. QED

Fixed phi is the one assumption in this note doing irreducible work. Name it in the paper's limitations; a referee will otherwise.

**Remark 6.1 (population of raters).** With a rater pool, severity still cancels *per rater* (each comparison lives inside one rater-sitting), and the anchor curve becomes a population mean E_r[sigma(v_{r,t}.dphi_j)]. Churn is re-draw of the mixture. Detection logic is intact; exact linearity becomes approximate.

**Remark 6.2 (gaming).** Repeated exposure invites anchor recognition and strategic answering. Rotate presentation (order, framing, interface position) — never content — and log per-rater exposure counts. This is a required covariate in the panel study.

---

## 7. What this fixes in the Tier-A plan

- **Estimand & coverage target:** drift of v_t; in reporting, P_A v_t with rho alongside.
- **Generator, in identified coordinates:** rubric edit = direction jump of what; fatigue/attention = norm decay; churn = mixture re-draw over v_r; judge swap = teacher-family change; the year-two gradient-coupled arm = rotation of what toward the policy's improvement direction. This also gives the e_shift component of the error decomposition a precise generator-side meaning.
- **Instrument:** build both panels (direction ~50/50 spread; norm ~80/20) via the §5b objective *before any drift schedule runs*; freeze; report rho for every run.
- **Estimator, now determined rather than chosen:** a logistic state-space model — latent state v_t; anchor labels as measurements through sigma; event-triggered jump variance plus diffusion variance growing with label staleness (the Glicko-style uncertainty inflation, placed exactly where it belongs). Fit by Laplace/EKF or particle filter; validate coverage of the filtered posterior *for v_t* in simulation.
- **Headline regression, unchanged but sharper:** anchor statistics vs. gold gap, on- and off-support separately, with the §4 stationary-rater arm as the mandatory null.

---

## 8. Assumptions, stated plainly

1. **Fixed phi over time** — equivalently: rater variation is confined to a fixed finite-dimensional criterion space. The real assumption (§6). d is unknown; misspecification shows up as anchor residuals no v-trajectory explains.
2. **Same-rater, same-sitting scope** for Proposition 1. Known real-world violations — context effects between adjacent comparisons within a session — are treated as covariates in the panel study, not modeled here.
3. **Independence of labels given v_t** (standard BT). Session-level dependence is future work.
4. **Link known up to monotone transform.** All structural results survive any strictly increasing link; the numeric anchor difficulties are link-specific (logistic: 0.82; probit: ~ 0.84).
5. Population statements (Remark 6.1) are first-order approximations around per-rater linearity.

Everything above is elementary. Its value is that the instrument design, the estimand, the estimator, and two of the pre-registered nulls are now consequences of a model written in four symbols — and each objection from the review round has become either a derived limitation with a number attached (rho), a mandatory control (§4), or a scoping table (§2).
