# EXPERIMENT_LOG

Process discipline X3: every attempted variant logged, including failures and
abandoned parameterisations. Nothing silently dropped.

Convention: dates absolute. Dev seeds used for all design decisions.

---

## 2026-07-29 — Build: foundational modules

Implemented `arms.py`, `graders.py`, `policies.py`, `reward_models.py`,
`runner.py`. Interpretation calls made (see plan Sections cited):

- **Arms (§3.1):** w0 ~ Dirichlet(1) on d-simplex; Phi ~ U[0,1]^{n×d};
  per-column standardise → shift non-negative → single GLOBAL rescale 1/range
  so max−min of w0·φ = 1 (preserves feature geometry; fixes achievable range).
  w0 generated inside `make_arms` (couples with the range-fix and guarantees
  conditions sharing a seed share BOTH Phi and w0 — §8 pairing).
- **Loop order (§7.2):** pi → sample/label/fit → update EMA m → LOG ROW →
  grader.step(m). Verified row t carries w_t.
- **Two RNG streams:** problem stream (Phi/w0/instruments, shared across
  conditions at a seed) and agent stream (policy noise, sampling, RM bootstrap;
  selectable via `agent_stream` so DRIFT-REPLAY feeds a decoupled fresh agent).
- **Score standardisation:** scores standardised to unit std over arms before
  the Gibbs policy, so β is a pure pressure knob with identical meaning across
  Stage A (ŵ·φ) and Stage B (r̂). Monotone → preserves argmax (P2 safe).

Smoke test (seed 0): SAT λ=0.4 β=0.05 → **Δ=0.093** (argmax of w moved dim 0→4:
ordering-changing confirmed, simplex sum=1.0). NONE → **Δ=0.000** exactly.
Runtime 0.11 s / 100-round run. Well under the 60 s budget.

---

## 2026-07-29 — P3 (vanilla over-optimisation): DOES NOT reproduce in strong form

**Gate status: BLOCKED pending user decision.** Plan §6/P3 says do not proceed
until vanilla over-opt reproduces (interior true-reward peak with visible
decline, peak shifting right with more comparisons).

### What I ran (all on ≥12–25 seeds, linear RM, linear-or-nonlinear true reward)
1. Static pressure sweep (Gibbs β ∈ [2, 0.002]), budgets q ∈ {8..1024},
   d ∈ {6,10,20}, βT ∈ {1,3,10}, dense w0.
2. Same over sparse w0 (k ∈ {1,2,3} of d nonzero) and nonlinear true reward.
3. Best-of-n axis (n_bo = 1..256), the RM-overopt scaling-law literature's knob.
4. Arm-set size n ∈ {256, 1024, 4096}; nonlinearity strength nl ∈ {0.7,1.5,3,5}.

### Result
- **Proxy reward:** monotonically increases with pressure everywhere (clean).
- **Proxy–true GAP:** grows sharply with pressure (clean Goodhart *decoupling*).
- **True-reward DECLINE:** ≤ **0.017** anywhere with Gibbs (nonlinear, tiny
  budget); **exactly 0.000** for best-of-n across ALL settings including nl=5.
  Below the 0.05 threshold and below the seed-noise floor everywhere.

### Why (structural, not a tuning miss)
Finite fixed arm set (n=256) + well-covered (uniform) comparisons + non-adversarial
feature geometry ⇒ the RM's argmax arm is genuinely high-true-reward. Best-of-n
is a pure ranking op: selecting top-1 of more samples can only climb the r̂
ordering, which correlates with the true ordering ⇒ monotone up, no decline. A
visible classic peak needs the RM's top arm to be true-BAD, which requires
ENGINEERED ANTI-CORRELATED spurious features — independent spurious dims give a
plateau-above-baseline (sparse-w0 tests: decline ≈ 0), not a decline. The Gao et
al. scaling law was derived for neural RMs over huge/continuous output spaces; it
does not transfer to the tabular finite-arm linear bandit.

### Interpretation (why this is arguably GOOD for the paper)
- Stage A has **no reward model at all**, yet Δ ≈ 0.09. The drift decline is a
  *moving-target* effect (agent faithfully optimises the drifted w_t, scored
  against w0), NOT reward hacking — so it cannot be an over-opt artifact.
- Quantified contrast: vanilla RM over-opt decline ≤ 0.017; drift decline ≈ 0.09.
  This makes the plan's wish — "the linear RM cannot be at fault" — a *measured*
  statement, not a hope.

### Options put to user (see conversation)
A. Reinterpret P3 pass = {proxy↑ with pressure; proxy–true gap↑ with pressure;
   vanilla decline < 0.02 ⇒ RM-not-at-fault} + keep a machinery-works demo. Proceed.
B. Add engineered anti-correlated-spurious-feature P3 variant that reproduces the
   literal classic peak, as an extra check. Proceed.
C. Halt per pre-registered kill and reassess.

---

## 2026-07-29 — Stage-A "existence" is largely mechanical / reference-dependent (user prompt)

User asked: in Stage A the agent directly optimises the drifted w_t, so isn't a
decline vs w0 bound to happen? Investigated on 10 dev seeds (λ=0.4, β=0.05).

### (1) Reference sensitivity — the decline is mostly a w0-only phenomenon
SAT: **Δ_init=+0.109, Δ_avg=+0.011, Δ_final=+0.002**. Trajectory (seed 0):
J_init falls 1.278→1.186 while J_final RISES 1.157→1.187. The agent tracks the
moving target: worse vs the ORIGINAL grader, better vs the FINAL grader. Under
J_avg the effect (0.011) is below the 0.05 threshold. Per plan §3.6 this must be
"reported as a finding": the SAT decline is NOT robust across the 3 references.

### (2) Endogeneity contributes ~nothing in Stage A (H3 looks dead)
EXO (fixed schedule, agent-independent) gives Δ_init=+0.097 — nearly equal to
SAT's +0.109. Paired SAT−EXO median ≈ 0.00–0.016 with IQRs spanning 0 across all
L,σ. Triggers the H3 kill condition: "the effect is mere non-stationarity,
reframe the paper accordingly (weaker claim)." At σ=0 the deterministic agent
makes REPLAY≡closed-loop exactly, so endogeneity can only bite through noise.

### Verdict
The bare TP1/H1 "over-optimisation exists" result is shallow: real vs w0 but
reference-dependent and not endogenously driven. This does NOT sink the project —
it relocates the contribution to what is untouched by the critique:
- **TP2/H4 detection blindness** (Stage B): a learned RM tracks the CURRENT
  grader, so held-out preference accuracy stays high even as J_init falls. This
  is a statement about *observability*, independent of whether the decline is
  "deep" or reference-dependent.
- **TP4 identifiability** (Stage D): is drifted comparison data even
  distinguishable from a static grader? If not → theorem: no preference-only
  monitor can work. This EMBRACES the reference-arbitrariness rather than
  fighting it. Plan §9 calls this "the strongest outcome."
The honest paper is about detection/identifiability under drift, not about a
surprising decline mechanism. Consistent with plan: "H2 is the most important
single run; H6 is the least important."

---

## 2026-07-29 — Plan rewrite received: R_t (regret vs current grader) + misspecified
## RM (ζ=Pφ, k<d) replaces Δ and full-rank linear RM. Validated before adopting.

User supplied a revised plan directly incorporating the two issues above: primary
metric Δ (decline vs `J_init`) → `R_t` (regret vs the SAME-β oracle `Gibbs(Φw_t,β)`
that KNOWS `w_t`), demoted `J_init/J_final/J_avg` to descriptive-only; RM basis
`r̂=ψ·φ` → `r̂=ψ·ζ`, `ζ=Pφ`, `P` a fixed random row-orthonormal `k×d` projection
(structural misspecification, `k<d`) replacing the old (measured-unsatisfiable)
full-rank P3. New mechanism metric `A_t` (drift direction · least-informed
comparison-covariance eigendirection) and gating test **N7** (oracle-tracking
regret floor, `R_t<1e-6` when agent is handed `w_t` exactly, under every drift
model). Ran two numeric validations on dev seeds before committing to the rewrite:

### (1) A_t's stated null `sqrt(2/(πd))` is asymptotic, not exact — CONFIRMED, fixed
Exact closed form for `E|u·v|`, `u,v` iid uniform on `S^{d-1}`:
`E|X| = (2/√π) · Γ(d/2) / ((d−1)·Γ((d−1)/2))`. Verified against 500k-sample Monte
Carlo to 4 decimals at `d∈{2,3,6,10,20}`. At the plan's default `d=6`: exact
**0.3395** vs. asymptotic 0.3257 (**4.1% low**); at `d=3`: 7.9% low; `d=2`: 11.4%
low. Since Stage A's kill condition is "`A_t` at the random-direction null → dead,"
a 4%+ miscalibration of the null threshold could flip a borderline call.
**Adopted: implement the exact Γ-function formula (`scipy.special.gammaln`), use
Monte Carlo only as the validation cross-check the plan already asks for (task
list item 2), not as the null itself.**

### (2) Corrected-P3 mechanism (ζ=Pφ, k<d) — WORKS, but exposed an R_t scale bug
First pass, literal reading of the plan's formula (raw, unnormalized scores on
both `π_t=Gibbs(r̂,β)` and `π*_t=Gibbs(Φw_t,β)`): **R_t went NEGATIVE**
(e.g. k=6/d=6, β=0.5: R_t=−0.337). This is a mathematical impossibility — `π*_t`
maximizes `E_π[true]−β·KL(π‖unif)`, which bounds `E_{π*}[true]−E_π[true]` only
*after* subtracting the KL term, not on raw `E_π[true]` alone. `r̂` comes from an
independently-scaled BT fit on the *projected* basis `ζ`; its magnitude has no
tie to `Φw_t`'s calibrated scale, so at a literal shared `β`, the agent's
effective greediness and the oracle's effective greediness are NOT the same
regularization strength — a misspecified-but-accidentally-sharper `r̂` can look
like negative regret. Root cause confirmed: this is a scale-calibration gap the
plan's text doesn't address (it never arises in Stage A, where `ŵ_t` shares
`w_t`'s exact units).

### 2026-07-29 — Rewrite build + Stage-A characterisation (dev seeds)

Built `metrics.py`, projection into `reward_models.py`, rewrote `runner.py`
(R_t/A_t/proj_angle, parquet 2-schema, errstate over/invalid/divide=raise,
under=ignore since softmax tails underflow benignly). Timings: Stage A 74 ms/run,
Stage B ~0.9 s/run. Parquet full 55 KB/run, reduced 31 KB/run.

**N7 PASS exactly:** oracle-tracking regret = `0.00e+00` for all
drift∈{NONE,SAT,SCALE,EXO,REPLAY} × λ∈{0,0.4,0.8} × β∈{0.5,0.1,0.02}. Gates
Stage B. Range-normalising both agent+oracle scores is what makes this exact.

**A_null:** using exact Γ formula (0.3395 at d=6), MC-verified.

**Drift dynamics (Stage A question 1 — the most important one):** SAT drift
CONVERGES to a fixed point. Clean perfect-tracking driver, avg 5 seeds:
- λ≤0.1: still moving at T=100 (`term_step`~2e-3, `late_slope`~+2e-3) — slow
  SUSTAINED drift over the horizon. argmax changes 0–60%.
- λ≥0.2: SETTLED (`term_step`<1e-3, `late_slope`~1e-4); travels ‖w_T−w_0‖~0.37–0.41
  then stops; argmax changes 100%. Never collapses to 1 dim (support stays ~6).
- Sustained *oscillation* (argmax cycling 5→2→5→2) only at extreme λ=3, α fast,
  and even then ‖Δw‖ is tiny (near-tie flicker), not large drift.
Consequence: the generator gives a **large persistent displacement**, not a
runaway spiral. This is a genuine property of satiation (stable fixed point), and
it is exactly what Stage A is meant to surface. It does NOT sink the paper —
TP2/TP4 need a persistent blind-regime, not a spiral — but it does constrain H1.

**H1 compounding (new positive-2nd-half-slope form) IS achievable, at small λ.**
Stage B (k=3 ensemble, 8 seeds), 2nd-half OLS slope of R_t on t:
- NONE (null): R̄=0.094 FLAT — this is the **misspecification floor** (k=3<6);
  onset δ must be re-pinned above it.
- λ=0.05: R̄=0.23, slope **+1.5e-3**  (rises 0.24→0.31)
- λ=0.10: R̄=0.34, slope **+3.9e-3**  (rises 0.37→0.50) ← cleanest H1 cell
- λ=0.20: R̄=0.42, slope −3.5e-3 (drift settled, RM catching up → falls)
- λ=0.40: R̄=0.38, slope ~0
So the **still-drifting small-λ regime (0.05–0.10) is where H1's compounding
lives**; large λ settles and regret plateaus/falls. Headline H1 cell → λ≈0.1.

**N6 confirmed with the proper regret metric + coverage (40× separation):**
correct-spec k=d regret VANISHES with coverage (R̄: 0.018@q64 → 0.0003@q8192);
misspec k=3 regret is STRUCTURAL (R̄~0.11–0.13, flat in q, unmoved at q=8192).
So N6 pass = {k=d, q≥256 → R̄≤0.02}; the effect is created by misspecification,
not the RM class per se. **P1:** cos(ψ̂,w)=1.000 at k=d (fitting correct);
cos(ψ̂,Pw)=0.98 (min 0.95) at k=3 (pseudo-projection, since the true model is
outside the k<d class) — so P1 tests >0.99 at k=d, >0.94 at k<d.

**Fix validated:** range-normalize both scores before Gibbs —
`score(x) = x / (max_a x(a) − min_a x(a))` — the SAME statistic (`range_t`)
already used to normalize `R_t`'s numerator (§3.6), applied consistently upstream
too. Result: **R_t ≥ 0 everywhere** (checked over the full β sweep, all k), and
an N7-style direct check (agent uses true `w_t` with no fit at all) gives
`R_t = 0.00e+00` exactly at every β. Mechanism is real and behaves as hoped:
monotone deepening as `k` shrinks (k=1: rise-after-elbow +0.31; k=6/unmisspecified:
+0.04 — an 8x gap), and the elbow/plateau shifts left with more comparisons
(q=16 plateaus ~0.22; q=1024 plateaus ~0.18, consistent with the misspecification
being a *structural* floor that more data cannot fully overcome, only partially
mitigate via `ψ`'s within-ζ component).

**Open point, not yet resolved — brought to user:** under range-normalization the
curve shape is monotone-rise-then-plateau, not the strictly interior-dip-then-rise
described by the plan's literal P3-corrected acceptance text. Translating the
OLD P3's "raw true reward has an interior PEAK then declines" into regret-space
(`R_t` = gap from the ceiling) actually predicts monotone-rise-then-plateau too
(regret is lowest where raw true reward peaks; nothing requires regret to ALSO be
elevated at the low-pressure boundary), so this may be a wording-precision issue
rather than a real discrepancy — but it touches a documented acceptance
criterion, so resolving the exact convention (and whether "interior minimum" is
meant strictly) is for the user, not a silent call.
