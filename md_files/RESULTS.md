# Drifted-Grader: Consolidated Results

Honest summary of the investigation. Full chronology + numbers in `EXPERIMENT_LOG.md`.
All code committed; 20 sanity checks pass (incl. exact N7 oracle floor).

## One-line summary

We built a preference-based-RL-under-drift simulator (bandit + tabular MDP) and
stress-tested the intuition that *endogenous grader drift causes a self-reinforcing
over-optimization the standard monitor misses*. **Every dramatic version of that
claim failed under testing.** What survives is a set of honest characterizations and
a reusable simulator.

## What we found

| Question | Result | Strength |
|----------|--------|----------|
| Does endogenous/closed-loop co-adaptation worsen outcomes beyond plain non-stationarity? | **No.** Paired closed-loop vs. replayed-drift, and active vs. passive (tandem), null across **8 tests** (bandit + MDP, with and without TD bootstrapping). | Robust null |
| Does reward over-optimization (proxy↑/true↓) occur? | Only under **engineered on-policy coverage gaps**. Correctly-specified linear RM: provably clean (≤0.02). Non-linear (MLP): over-optimizes (~0.20) **only** when on-policy collection starves off-policy regions. | Coverage-gated |
| …in real control PbRL? | **No** — matches collaborator's DMC/Metaworld PEBBLE experiments (no over-opt even with a frozen RM) and REBEL (claims "over-optimization" but shows no true-reward decline). Coverage is sufficient. | Confirmed externally |
| Is the standard monitor (held-out preference accuracy) blind? | **Yes, and worse — misleading.** It *rises* (71%→76%) while true regret rises, because it re-samples on-policy comparisons that co-move with the drift and the RM is refit to track them. | Clean |
| Is preference drift identifiable from comparison data? | **Weakly identifiable**, not impossible. Static-vs-oracle-drift held-out NLL margin is small (0.002–0.02 nats) but CI > 0. Smaller under on-policy / high-pressure querying. | Weak positive |
| Can a simple monitor detect drift? | **Yes** — any *fixed* anchor set re-elicited over time detects it (power ~1.0), regardless of breadth. The distinction is fixed-vs-co-moving queries, not breadth. | Clean but modest |

## Mechanism (the one clean thread)

Ordering-changing drift moves grader weight preferentially along the directions the
**on-policy comparison data least constrains** (variance-ratio VR < 1, amplified vs.
exogenous drift). Consequences: (i) the RM is least informed exactly where the grader
newly cares; (ii) an on-policy monitor never probes the drift direction, so held-out
accuracy stays high / rises — blind; (iii) a *fixed* anchor set does probe it, so
it detects the drift.

## Honest assessment

- **Not** the originally-envisioned result (endogenous over-optimization collapse).
  The endogeneity is null; the over-optimization is a coverage artifact absent in real
  control PbRL.
- The defensible contributions are **deflationary / characterizing**: (a) co-adaptation
  adds nothing beyond non-stationarity; (b) over-optimization in PbRL is coverage-gated,
  reconciling the literature's claims with the empirical null; (c) the standard held-out
  monitor is actively misleading under drift, and fixed anchors fix it.
- Scale caveat: everything is CPU-tabular. Whether *any* of these effects matter at
  DMC/Metaworld scale (Stage F / PEBBLE) is untested and is the only way to know if
  there is a real-scale phenomenon.

## Robustness of the RM protocol (checked)

The reward model is **inherited (warm-started) across feedback rounds**, not refit
from scratch — matching PEBBLE (MLP weights + Adam state persist; linear is convex
so it re-converges to the full-buffer MLE). The closed-loop regret is **flat across
a 7.5× per-round training budget** (40→300 Adam steps: NONE 0.176→0.176, SAT
0.288→0.279), so it is genuine, not under-fitting. The persistent no-drift floor
(~0.18) is on-policy-coverage over-optimization: the MLP's off-policy values are
unconstrained regardless of training budget because on-policy collection never
samples those regions — the coverage axis, consistent with PEBBLE's broad-coverage
null. Stage F (PEBBLE integration, `NoisyPbRL/STAGE_F_DRIFT.md`) is the scale test.

## Deliverable that stands regardless

`drifted_grader/`: reusable PbRL-under-drift simulator — arms, five drift models
(SAT/SCALE/EXO/REPLAY/NONE), Gibbs/best-of-n/softmax policies, linear + MLP BT reward
models, tabular-MDP fitted-Q value agent, runner with parquet logging, tandem/replay
and identifiability/detectability harnesses, regret (R_t) + mechanism (VR) metrics,
and a 20-check sanity suite. Analysis drivers for every stage.
