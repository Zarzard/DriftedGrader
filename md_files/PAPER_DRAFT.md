# When the Grader Moves: Reward Over-Optimization, Safety Bounds, and Ranking Validity under Endogenous Preference Drift

*Consolidated writeup. Companion to `EXPERIMENT_LOG.md` (full chronology + numbers)
and the reusable simulator in `drifted_grader/`. All results are from a controlled
CPU simulation study; the scale caveat is stated explicitly in §7.*

---

## Abstract

In preference-based reinforcement learning (PbRL) and RLHF, a learned reward model is
fit to the judgments of a *grader* (a human, a reward model, or an LLM judge). A
standing worry is that the grader is not static: what it rewards can **drift**, and it
can drift *endogenously* — as a function of the very behavior the agent produces —
creating a closed loop in which the agent shapes the objective it is then optimized
against. We ask whether this endogenous drift produces the failure modes it is feared
to: (i) a self-reinforcing collapse of true reward that standard monitors miss, and
(ii) a breakdown of Bayesian *safety bounds* (value-at-risk lower bounds derived from a
reward posterior). We build an instrumented simulator with an explicitly moving,
endogenously-coupled grader and run a pre-registered battery of falsification tests.

**Every dramatic version of the fear fails.** Endogenous co-adaptation adds nothing
beyond ordinary non-stationarity (null across 8 paired tests); reward
over-optimization occurs only under engineered on-policy coverage gaps and is absent
in broad-coverage control PbRL; and the reward-posterior VaR lower bound's **marginal
coverage is robust** to drift — it over-covers and tightens, never entering the feared
"confident-but-wrong" quadrant. Higher model capacity does not amplify the failure.
What *does* survive is one sharp, reproducible positive result: **under representational
misspecification, drift silently destroys the bound's *ranking* validity while its
marginal coverage stays perfect.** The bound remains honest about "is *this* policy
safe?" while becoming actively dishonest about "*which* policy is safest?" We give the
mechanism (drift concentrates in the directions the on-policy comparison data least
constrains) and discuss why the marginal/ranking distinction is the practically
important one.

---

## 1. Introduction

A PbRL/RLHF pipeline has three moving parts: a **grader** that expresses preferences
over candidate behaviors, a **reward model** fit to those preferences, and a **policy**
optimized against the reward model. Almost all analysis assumes the grader is a fixed
oracle. In practice it is not. An LLM judge's rubric shifts as the distribution of
things it scores shifts; a human labeler satiates, anchors, and re-calibrates; a reward
model used as a grader for a downstream policy is itself a moving target. Crucially,
this drift can be **endogenous**: the agent's own behavior changes what the grader
attends to next, closing a feedback loop between the policy and its objective.

The intuition this invites is alarming. If the agent can *move the goalposts*, perhaps
it can enter a self-reinforcing spiral — over-produce whatever the grader currently
likes, cause the grader to re-weight toward directions the agent has exploited, and so
on — with the standard health check (held-out preference accuracy) staying green
throughout because it is measured on the same co-moving distribution. And if we try to
protect against this with a **Bayesian safety bound** — a value-at-risk lower bound on
the deployed policy's true value, computed from the reward posterior — perhaps a moving
grader makes that bound *confidently wrong*: tight (so it looks trustworthy) yet no
longer covering the truth.

This paper tests those two fears directly. We contribute:

1. **A reusable, instrumented simulator** (`drifted_grader/`) for PbRL under a moving,
   endogenously-coupled grader: five drift models, three policy classes, linear / MLP /
   ensemble Bradley–Terry reward models, a tabular-MDP fitted-Q agent, a reward-posterior
   VaR module, and metrics that isolate *endogeneity*, *coverage*, and *ranking* as
   separate axes. A 20-check sanity suite (including an exact oracle-regret floor) passes.

2. **A battery of falsification results (§4).** The endogeneity fear is a robust null;
   over-optimization is coverage-gated; the standard monitor is not merely blind but
   *anti-informative*; and the safety bound's marginal coverage is robust to drift.

3. **One sharp positive result (§4.5): ranking-validity collapse.** Under
   representational misspecification, drift drives the bound's ranking of alternative
   policies from strongly correct to *inverted* (Spearman 0.78 → −0.29) while marginal
   coverage stays at 1.0 and the bound *tightens*. This is the paper's real finding, and
   the marginal/ranking split is, we argue, the one that matters for deployment.

The overall posture is deflationary-but-precise: the field's dramatic worry does not
reproduce, its literature's over-optimization claims are reconciled with the empirical
null via a coverage argument, and a single previously-unarticulated failure (ranking
under misspecification+drift) is isolated and characterized.

---

## 2. Preliminaries and notation

We keep the model deliberately minimal so that every effect can be attributed. This
section is self-contained; readers familiar with Bradley–Terry PbRL can skim it.

**Candidates and features.** There is a finite set of $N$ candidate behaviors
$\{a_1,\dots,a_N\}$ (the "arms" of a contextual-free bandit; in RLHF, think of them as
responses or trajectories). Each candidate has a fixed **feature vector**
$\phi(a)\in\mathbb{R}^d$; stack them as $\Phi\in\mathbb{R}^{N\times d}$. Features are the
axes along which the grader can care about behavior (e.g. "helpfulness", "length",
"formality"). We use $N=256$, $d=6$.

**The grader (Bradley–Terry).** The grader has a **weight vector** $w$ on the
$(d{-}1)$-simplex ($w_i\ge 0$, $\sum_i w_i = 1$): $w_i$ is how much it currently cares
about feature $i$. Given two candidates it prefers $a$ over $b$ with probability

$$
P(a \succ b) \;=\; \sigma\!\big(\beta_T \cdot w^\top(\phi(a)-\phi(b))\big),
\qquad \sigma(x)=\tfrac{1}{1+e^{-x}},
$$

where $\beta_T$ is the grader's **inverse temperature / rationality** (how sharply it
acts on utility differences; $\beta_T=10$). The grader's **utility** for a candidate is
$g(a)=w^\top\phi(a)$; we call this the *true reward under the current grader*.

**Drift models.** The grader weight $w_t$ evolves over feedback rounds $t=0,1,\dots$.
The primary, endogenous model is **DriftSat** (*satiation*): let $\tilde m_{t,i}$ be a
running, range-normalized measure of how much feature $i$ the agent has recently
produced. Then

$$
w_{t+1,i} \;\propto\; w_{t,i}\,\exp(-\lambda\,\tilde m_{t,i}),
$$

renormalized to the simplex. The grader **satiates** on whatever the agent
over-produces: features the policy exploits lose weight, so the objective moves *away*
from the agent's current behavior. This is (a) **endogenous** (driven by the agent's own
$\tilde m$) and (b) generically **ordering-changing** (it can reorder candidates, not
just rescale). $\lambda\ge 0$ is the **drift rate** (0 = static; 0.1 mild, 0.4
substantial, 0.8 extreme). SAT converges to a fixed point (settles by round ~40). We
also use four controls: **DriftScale** (order-*preserving* rescaling), **DriftExo**
(exogenous random walk, no coupling to the agent), **DriftReplay** (a pre-recorded SAT
weight trajectory played back *open-loop*, i.e. same marginal drift with the feedback
loop cut), and **DriftNone** ($\lambda=0$, static). Replay and Exo are the crucial
counterfactuals for isolating *endogeneity*: they hold the drift's marginal statistics
fixed while removing the closed loop.

**Policy.** A policy $\pi$ is a distribution over the $N$ candidates. The deployed policy
is a **Gibbs (softmax) policy** on the reward model's estimate $\hat r$:
$\pi(a)\propto\exp(\hat r(a)/\tau)$, with **deployment inverse-temperature** $\beta$ (we
write scores range-normalized so $\beta$ is a pure "optimization pressure" knob;
$\beta=0.05$). Its expected feature vector is $f_\pi=\sum_a \pi(a)\phi(a)=\Phi^\top\pi$.

**Reward models.** Fit to a growing buffer $D=\{(\phi(a_k),\phi(b_k),y_k)\}$ of grader
comparisons ($y=1$ if $a\succ b$) by Bradley–Terry maximum likelihood (or MAP):

- **LinearBT**: $\hat r(a)=\theta^\top\phi(a)$. Convex; warm-started to the buffer MLE.
- **MLPBT**: $\hat r(a)=w_2^\top \tanh(W_1\phi(a)+b_1)$. Non-linear; **warm-started**
  (weights + Adam state persist across rounds, matching PEBBLE — not refit from scratch).
- **EnsembleBT**: $K$ members, either *bootstrap* diversity (resampled buffers) or
  *initialization* diversity (shared data, different seeds — what neural RLHF actually
  uses). Ensemble **disagreement** $\mathrm{std}_k \hat r_k$ is the usual epistemic signal.

**Regret (the true-reward yardstick).** We measure the deployed policy against the
*same-temperature oracle that knows the current grader*. Let
$\pi^\star_t=\mathrm{Gibbs}(\Phi w_t,\beta)$ be the Gibbs policy the agent *would* deploy
if handed the true current weights $w_t$. Then

$$
R_t \;=\; \big(w_t^\top f_{\pi^\star_t} - w_t^\top f_{\pi_t}\big)\big/\mathrm{range}_t,
$$

range-normalized so $\beta$ is comparable across rounds. $R_t=0$ exactly when the agent
is handed $w_t$ (the **oracle-tracking floor**, verified as an exact sanity check). $R_t$
rising means the deployed policy is losing true value *relative to what is achievable at
the same pressure against the current grader* — i.e. genuine over-optimization, not just
the goalposts moving.

**The reward posterior and the VaR safety bound (Plan 02).** Instead of a point-estimate
reward, put a posterior over grader weights:

$$
P(w\mid D)\;\propto\;\mathcal N(0,\,\sigma_0^2 I)\;\prod_{(a,b,y)\in D}
\sigma\!\big(\beta_T\,w^\top\Delta\phi\big)^{y}\,\sigma\!\big(-\beta_T\,w^\top\Delta\phi\big)^{1-y},
\quad \Delta\phi=\phi(a)-\phi(b).
$$

We draw samples $\{w_i\}$ via a Laplace proposal (MAP + Hessian) with importance
reweighting (validated against Metropolis–Hastings). For a policy $\pi$ with feature
expectation $f_\pi$, the induced **value distribution** is $\{w_i^\top f_\pi\}$, and the
**value-at-risk lower bound** at level $\alpha$ is its $\alpha$-quantile:

$$
\mathrm{LB}_\alpha(\pi)\;=\;\mathrm{Quantile}_\alpha\{\,w_i^\top f_\pi\,\}.
$$

The **safety claim** is a one-sided guarantee: the true value
$V_t(\pi)=w_t^\top f_\pi$ should fall *above* the bound with probability $\ge 1-\alpha$.
We measure two very different things:

- **Marginal coverage**: $\Pr[\,V_t(\pi_t)\ge \mathrm{LB}_\alpha(\pi_t)\,]$ over
  seeds/rounds — is the bound honest about the *deployed* policy? Target $\ge 1-\alpha$.
- **Ranking validity**: $\mathrm{Spearman}\big(\mathrm{LB}_\alpha(\pi^j),\,V_t(\pi^j)\big)$
  over a fixed set of *evaluation* policies $\{\pi^j\}$ — does the bound rank
  *alternatives* correctly? This is the Bayesian-REX use case (pick the policy with the
  best worst-case value). It is affine-invariant and orthogonal to marginal coverage.

$\beta_T$ **must be known** to the posterior: it sets the utility scale, and a $\pm50\%$
misspecification of $\beta_T$ flips coverage between 0 and 1 (Gate PG3). We treat
$\beta_T$ as given (as in Bayesian-REX-style setups).

**The mechanism metric — variance ratio (VR).** To explain *where* drift hurts, let
$u_t$ be the (unit) drift direction $\propto w_{t+1}-w_t$ and let
$C_t=\sum \Delta\phi\,\Delta\phi^\top$ be the **comparison design matrix** (the Fisher
information geometry of the queries actually asked). Define

$$
\mathrm{VR}_t \;=\; \frac{u_t^\top C_t\,u_t}{\mathrm{tr}(C_t)/d}.
$$

$\mathrm{VR}<1$ means the grader is drifting into a direction the comparison data
constrains *less* than average — a blind spot of the queries. Endogenous SAT drift has
$\mathrm{VR}<1$, amplified relative to exogenous drift.

**Notation summary.**

| Symbol | Meaning |
|---|---|
| $\phi(a),\ \Phi$ | feature vector of candidate $a$; feature matrix ($N\times d$) |
| $w_t$ | grader weights at round $t$ (on the simplex); $g_t(a)=w_t^\top\phi(a)$ |
| $\beta_T$ | grader inverse temperature (rationality), $=10$ |
| $\beta$ | deployment inverse temperature (optimization pressure), $=0.05$ |
| $\lambda$ | **drift rate** (0 = static). *Distinct from* $\lambda_{\text{cons}}$ below. |
| $\lambda_{\text{cons}}$ | conservative-penalty coefficient in $\hat r_{\text{mean}}-\lambda_{\text{cons}}\,\mathrm{std}$ |
| $k$ | reward-model feature rank; $k<d$ = **misspecified** (reward can't see all axes) |
| $\pi_t,\ f_\pi$ | deployed policy; its feature expectation $\Phi^\top\pi$ |
| $R_t$ | regret vs the same-$\beta$ oracle that knows $w_t$ (true-reward yardstick) |
| $\mathrm{LB}_\alpha,\ V_t$ | VaR lower bound at level $\alpha$; true value $w_t^\top f_\pi$ |
| $\mathrm{VR}_t$ | variance ratio: is drift into a query blind spot? ($<1$ = yes) |

---

## 3. Simulator and experimental method

Each run is a closed loop over $T=100$ feedback rounds: (1) deploy $\pi_t$ (Gibbs on the
current reward estimate / posterior mean); (2) sample $Q=64$ comparison queries — either
**on-policy** (pairs drawn from $\pi_t$, as in PEBBLE) or **broad** (uniform over
candidates); (3) label them with the current grader $w_t$; (4) append to the buffer and
refit the reward model / posterior (warm-started); (5) advance the grader
$w_t\!\to\!w_{t+1}$ using the agent's recent feature usage. Metrics ($R_t$, coverage,
rank-corr, VR, disagreement, bound width) are logged per round. We report final-round
statistics (mean over rounds $T{-}5\dots T$) across 8–40 seeds with bootstrap CIs. To
isolate endogeneity we additionally run **replay** (step 5 replaced by a pre-recorded
weight trajectory) and **tandem** (a passive learner trained on a separate active
learner's data, following Ostrovski et al., 2021) counterfactuals.

A **representational-misspecification** knob is central to Plan 02: the reward model sees
only a rank-$k$ projection $\zeta=P\phi$ of the features ($P\in\mathbb{R}^{k\times d}$,
$k<d$). At $k=d$ the reward can represent any grader; at $k<d$ there are grader
directions it is structurally blind to. Real reward models are always misspecified in
this sense.

---

## 4. Results

We organize results as falsification tests. Plan-01 tests (§4.1–4.3) concern the
endogeneity/over-optimization/monitoring fears; Plan-02 tests (§4.4–4.7) concern the
safety bound. Hypothesis labels (HG*) are the pre-registered names from the plans.

### 4.1 Endogenous co-adaptation adds nothing beyond non-stationarity (robust null)

**Test.** Does closing the loop between policy and grader worsen outcomes relative to the
*same drift experienced open-loop*? We compare, paired by seed: (a) closed-loop SAT vs.
**replay** of its own weight trajectory; (b) **active vs. passive** tandem learners
(Ostrovski et al., 2021), with and without TD bootstrapping in a tabular MDP; (c)
endogenous SAT vs. exogenous Exo matched on drift magnitude.

**Result.** **Null across 8 tests.** Regret trajectories of closed-loop and replayed
drift are statistically indistinguishable; active and passive tandem learners match. The
extra structure in endogenous drift — that the agent *caused* the goalpost move — does
not translate into extra regret beyond what the marginal non-stationarity already
produces. The feedback loop is real but inert as a *harm amplifier*.

*Interpretation.* The dramatic "the agent games its own objective into a spiral" story
requires the endogeneity to do work beyond non-stationarity. It does not.

### 4.2 Reward over-optimization is coverage-gated

**Test.** Does the deployed policy's true reward fall while the proxy rises
(over-optimization)? We sweep reward-model class and query breadth.

**Result.** With a **correctly-specified linear** reward model, over-optimization is
provably and empirically negligible (regret floor $\le 0.02$). With a **non-linear MLP**
reward, over-optimization appears (regret $\approx 0.20$) **only** when on-policy
collection starves the off-policy regions — i.e. only under **engineered coverage gaps**.
The effect is flat across a $7.5\times$ per-round training budget (40→300 Adam steps:
NONE $0.176\!\to\!0.176$, SAT $0.288\!\to\!0.279$), so it is a genuine coverage artifact,
not under-fitting. In **broad-coverage** control PbRL it vanishes — consistent with the
collaborator's DMC/Metaworld PEBBLE experiments (no over-optimization even with a
*frozen* reward model) and with REBEL (which reports "over-optimization" but shows no
true-reward decline).

*Interpretation.* Over-optimization in PbRL is gated on **non-linear FA $\times$
on-policy coverage gaps**, not on drift per se. This reconciles the literature's alarm
with the empirical null: broad coverage (the norm in control PbRL) closes the gate.

### 4.3 The standard monitor is anti-informative; fixed anchors fix it

**Test.** Does held-out preference accuracy — the standard health check — track true
regret under drift?

**Result.** It moves the **wrong way**: held-out accuracy *rises* (71%→76%) while true
regret rises, because the held-out set is re-elicited *on-policy* and therefore co-moves
with the drift, and the reward model is refit to track it. Drift is only **weakly
identifiable** from co-moving comparison data (static-vs-oracle-drift held-out NLL margin
0.002–0.02 nats, CI $>0$; smaller under on-policy/high-pressure querying). But **any
fixed anchor set**, re-elicited over time, detects the drift with power $\approx 1.0$.
The decisive variable is *fixed vs. co-moving queries*, not query breadth.

*Interpretation.* The monitor everyone uses is not merely blind under drift — it is
actively reassuring while things get worse. The fix is cheap: hold out a *fixed* probe
set and watch its labels move.

### 4.4 Safety bounds under drift: marginal coverage is robust (HG1–HG3 null)

**Precondition (HG0).** The Plan-02 story requires that, under drift, ensemble
disagreement *falls* as regret *rises* (so a disagreement-based penalty would shrink
exactly when the reward gets more wrong). Recomputed as the updated plan's paired-by-seed
difference of $\mathrm{Spearman}(\text{disagreement}, R_t)$ (40 seeds, $\lambda=0.1$):

$$
\rho_{\text{SAT}}=-0.318\,[-0.436,-0.139],\quad
\rho_{\text{NONE}}=-0.109\,[-0.213,+0.132],\quad
\Delta=\rho_{\text{SAT}}-\rho_{\text{NONE}}=-0.213\,[-0.396,-0.034].
$$

The paired difference is significantly **negative** (CI excludes 0): SAT is more
anti-correlated than the static control, confirming the mechanism. *(Note: the plan's
HG0 table states the support condition as "$\Delta>0$", which is sign-inverted relative
to its own §1.4 mechanism; it should read "$\Delta<0$". The data satisfy the corrected
condition.)*

**Main test (HG1–HG3).** Does drift push the VaR bound into the *dangerous quadrant* —
tight (small width $W_t$) yet no longer covering $V_t(\pi_t)$? Linear posterior,
on-policy queries, $\alpha=0.05$, 20 seeds:

| $k$ | $\lambda$ | coverage | rank-corr | width $W_t$ | $R_t$ |
|----:|----:|----:|----:|----:|----:|
| 6 (well-spec) | 0.0 | 0.925 | 1.000 | 0.142 | 0.001 |
| 6 | 0.4 | **1.000** | 0.977 | 0.120 | 0.037 |
| 6 | 0.8 | **1.000** | 0.980 | 0.110 | 0.062 |
| 3 (misspec)   | 0.0 | 0.950 | 0.777 | 0.083 | 0.093 |
| 3 | 0.4 | **1.000** | 0.081 | 0.040 | 0.368 |
| 3 | 0.8 | **1.000** | **−0.288** | 0.034 | 0.377 |

**Marginal coverage is robust to drift and, if anything, *over*-covers**: at every drift
rate, in both the well-specified and misspecified reward models, coverage is $\ge 0.95$
and rises to 1.0 as $\lambda$ grows. Drift *tightens* the settled posterior rather than
un-covering it. The dangerous quadrant (tight + uncovered) **does not appear**. HG1–HG3
are falsified.

*Why coverage survives.* The deployed policy's value is bounded by a posterior that has
seen comparisons drawn around that same policy; drift shrinks the buffer's effective
disagreement as it settles, tightening the bound *around a value that stays covered*.
Marginal one-sided coverage is a weak property and drift does not break it.

### 4.5 The real failure — ranking-validity collapse (Finding 1)

Look at the **rank-corr** column of the table above, not the coverage column. At $k=6$
(well-specified) ranking is robust ($\ge 0.98$). At $k=3$ (**misspecified**), as drift
increases, ranking validity **collapses and inverts**: Spearman $0.78 \to 0.08 \to
-0.29$ — while marginal coverage stays at $1.000$ **and the bound gets tighter**
($W_t: 0.083\to0.034$). The bound becomes **confidently mis-ranked**: it is honest about
the deployed policy yet, asked "which of these alternative policies has the best
worst-case value?", it returns an increasingly *reversed* ordering, with *narrower*
(more confident-looking) intervals.

*Mechanism.* Misspecification ($k<d$) means the posterior lives in a $k$-dimensional
projection of weight space; the grader's drift has a component in the null space of that
projection ($\mathrm{VR}<1$: drift concentrates where the on-policy queries least inform
the model). Marginal coverage of $\pi_t$ survives because $\pi_t$ itself is what the
queries surround. But the *ranking* of off-policy alternatives depends on exactly the
null-space component the projected posterior cannot represent — so as drift loads that
component, the ranking degrades and then inverts. Coverage and ranking come apart
precisely because one is on-policy and the other is off-policy.

**This is the paper's positive result.** For a safety bound used to *marginally certify*
a fixed policy, drift is not the threat it is feared to be. For a safety bound used to
*choose among* policies under any representational misspecification (i.e. always), drift
silently destroys the thing you are using it for, and the bound's shrinking width makes
the failure look like increasing confidence.

### 4.6 The conservative penalty does not invert for neural ensembles (HG4 null, Finding 2)

**Test.** The classic conservative objective is $\hat r_{\text{mean}} -
\lambda_{\text{cons}}\,\mathrm{std}_k$. HG4 fears the penalty *inverts* — shrinks as the
reward gets more wrong — making the "conservative" policy anti-conservative. MLP
ensemble, $K=5$, full features, 10 seeds:

| drift | breadth | rank-corr | pen–$R$ corr |
|---|---|---:|---:|
| NONE | on-policy | 1.000 | +0.515 |
| SAT | on-policy | 0.888 | +0.283 |
| SAT | broad | 0.928 | +0.122 |

**No inversion.** For **initialization-diversity** ensembles (what neural RLHF actually
uses), the penalty–regret correlation is **positive** everywhere: the penalty *grows* as
regret grows, and ranking stays robust ($\ge 0.89$). The inversion is specific to
*bootstrap*-diversity ensembles, whose disagreement collapses as data accumulate. This
contradicts the plan's §1.4 fear and is reassuring for practice: the ensemble tracks the
drift instead of going quietly confident.

### 4.7 Capacity does not amplify the failure (HG7 null)

**Test.** Does higher reward-model **capacity** fit the pooled grader *mixture* better
(the union of all rounds' graders) while covering the *current* grader worse — a
capacity-driven coverage collapse? Last-layer Laplace over an MLP trunk refit each round;
drift-attributable coverage drop $=C(\lambda{=}0)-C(\lambda{=}0.4)$; pooled held-out
log-loss as the second axis; 8 seeds:

| capacity | $C(\lambda{=}0)$ | $C(\lambda{=}0.4)$ | drop | pooled LL | posterior width $K_t$ |
|---|---:|---:|---:|---:|---:|
| linear   | 0.792 | 1.000 | **−0.208** | 0.587 | 0.002 |
| mlp-32   | 0.979 | 1.000 | −0.021 | 0.437 | 25.1 |
| mlp-128  | 1.000 | 1.000 | **+0.000** | 0.444 | 318.6 |

**Null.** The first axis of the mechanism *does* hold — pooled log-loss falls with
capacity ($0.59\to0.44$: higher capacity fits the mixture better). But the coverage drop
does not grow with capacity; it **vanishes** ($-0.208\to0.000$; drift never lowers
coverage). And the high-capacity "perfect coverage" is **vacuous**: posterior width
$K_t$ blows up $\sim10^5\times$ ($0.002\to319$), so the last-layer Laplace bound goes
slack and cannot miss. Neither the feared collapse **nor** a useful tight bound survives
at high capacity. HG7 is falsified on its own stated criterion.

---

## 5. Summary of hypotheses

| Label | Fear | Verdict |
|---|---|---|
| — (endogeneity) | closed-loop drift amplifies harm beyond non-stationarity | **Null** (8 paired tests) |
| — (over-opt) | drift causes true-reward collapse | **Coverage-gated**; absent in broad-coverage PbRL |
| — (monitor) | held-out accuracy misses it | **Worse: anti-informative**; fixed anchors detect (power ~1.0) |
| HG0 | disagreement falls as regret rises (precondition) | **Holds** ($\Delta=-0.213$, sig.) |
| HG1–HG3 | drift → tight-but-uncovered VaR bound | **Null**; marginal coverage robust, over-covers |
| **Finding 1** | — | **Ranking validity collapses/inverts** under misspec+drift (0.78 → −0.29) while coverage = 1.0 |
| HG4 / **Finding 2** | conservative penalty inverts | **Null for init-diversity ensembles** (the practical case) |
| HG7 | capacity amplifies coverage failure | **Null**; capacity fits mixture better but coverage robust/vacuous |

---

## 6. Related work

**Reward over-optimization / Goodhart.** Gao et al. (2023) characterize
over-optimization scaling for a *static* gold reward; we add that in PbRL the effect is
gated on non-linear FA $\times$ on-policy coverage, explaining its absence in
broad-coverage control settings. **Tandem learning** (Ostrovski, Castro & Dabney,
NeurIPS 2021) isolates the "passive" deep-RL failure to non-linear FA + bootstrapping; we
reuse their active/passive design and find the *supervised* (reward-learning) tandem
effect largely vanishes, consistent with their supervised control. **Bayesian reward
learning / VaR** (Brown et al., Bayesian-REX) computes exactly the value-posterior lower
bound we stress-test; our contribution is to show its marginal robustness *and* its
ranking fragility once the grader moves and the model is misspecified. **PbRL systems**
(PEBBLE; B-Pref) supply the on-policy, warm-started reward-learning protocol we match.
**Non-stationary/strategic rewards** (performative prediction; corrupted/ hackable
rewards) motivate endogenous drift; our null says the *endogeneity per se* is not the
harm amplifier there.

*Novelty.* The specific, reproducible contribution is the **marginal-vs-ranking split**
of a reward-posterior safety bound under grader drift: to our knowledge the observation
that drift can leave one-sided marginal coverage perfect (even over-covering, with
*tightening* intervals) while inverting the bound's policy *ranking* under
misspecification has not been articulated.

---

## 7. Discussion, limitations, and threats to validity

- **Scale.** Everything here is a CPU simulation with $d=6$, $N=256$, linear/small-MLP
  rewards. Whether the ranking-collapse finding (or any null) transfers to DMC/Metaworld
  scale is **untested**; the PEBBLE integration (Stage F) is the intended scale check and
  its results are not yet in. The nulls are *predictions* the collaborator's DMC
  experiments already partially corroborate (no over-optimization), but the ranking
  finding specifically needs a scale test.
- **$\beta_T$ known.** The VaR bound requires the grader's rationality scale $\beta_T$;
  $\pm50\%$ misspecification flips coverage (PG3). Real judges have unknown, possibly
  drifting $\beta_T$ — a real external limitation, not modeled here.
- **Finite-sample coverage.** At small $N$ the linear posterior slightly *under*-covers
  by the Bernstein–von Mises rate (nominal only $\approx0.96$ at $N=30$k); the $k=6,
  \lambda=0$ cell's $0.925$ is this finite-sample reference, not a drift effect. All
  drift claims are *increments over each instrument's own $\lambda=0$ reference row*,
  paired by seed (the plan's §1.6 baseline principle).
- **Affine invariance.** MLP rewards are identified only up to an additive constant, so
  absolute coverage is ill-posed for MLP ensembles; we use the affine-invariant rank
  correlation there.
- **Drift model.** SAT satiation is one plausible endogenous mechanism; results may not
  generalize to adversarial or reward-hacking-style grader dynamics.

**Bottom line.** The engineering takeaway is asymmetric and, we think, actionable: a
reward-posterior VaR bound is *more* robust to grader drift than feared when used to
**certify a fixed deployed policy**, and *less* robust than it looks when used to
**select among policies** under representational misspecification — because there its
ranking silently inverts while its intervals tighten. Pair any such bound with a
**fixed-anchor drift monitor** (§4.3), and do not trust its cross-policy ranking under
suspected drift without an out-of-model probe.

---

*Artifacts.* Simulator and all analysis drivers in `drifted_grader/` and `analysis/`;
per-stage numbers, CIs, and figures in `EXPERIMENT_LOG.md`; 20-check sanity suite
(including the exact oracle-regret floor) passes. Commits: gates+P0 `8e58818`, P1
`f37848c`, P2 `8896544`, HG7+HG0-paired `9443a45`.
