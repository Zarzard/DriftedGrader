# Drifted-Grader

A controlled simulator and experiment suite for **preference-based RL / RLHF when the
grader drifts** — i.e. when *what the grader rewards* changes over time, possibly
*endogenously* (as a function of the agent's own behaviour). We stress-test two fears:
(1) that endogenous drift causes a self-reinforcing collapse of true reward that
standard monitors miss, and (2) that a Bayesian reward-posterior **value-at-risk (VaR)
safety bound** (the Bayesian-REX primitive) breaks under drift.

**Headline finding.** The dramatic fears mostly *don't* reproduce: endogenous
co-adaptation adds nothing beyond ordinary non-stationarity, reward over-optimization is
coverage-gated (absent in broad-coverage control PbRL), and a VaR bound's **marginal
coverage of a fixed policy stays honest** under drift. What *does* survive is one sharp
result — under representational **misspecification + drift**, the bound's **policy
ranking / selection** corrupts while its per-policy coverage stays fine: it is
trustworthy for *"is this policy safe?"* and misleading for *"which policy should I
deploy?"*. This is dramatic in the tabular limit (Bayesian-REX picks the wrong policy
91% of the time under misspec+drift) and **muted but measurable at neural (Metaworld
PEBBLE) scale**.

Full narrative in [`md_files/RESULTS.md`](md_files/RESULTS.md) and
[`md_files/PAPER_DRAFT.md`](md_files/PAPER_DRAFT.md); complete chronology (including every
null and abandoned variant) in [`md_files/EXPERIMENT_LOG.md`](md_files/EXPERIMENT_LOG.md).

---

## Repository structure

```
drifted_grader/          # the simulator package (CPU, numpy/scipy only)
  arms.py                #   candidate set: features Phi (N x d), grader weights w0 on the d-simplex, range-normalisation
  graders.py             #   Bradley-Terry grader + drift models: NONE, SAT (satiation, ordering-changing), SCALE, EXO, REPLAY
  policies.py            #   Gibbs/softmax policies, expected features, comparison-pair sampling
  reward_models.py       #   LinearBT / MLPBT / EnsembleBT reward models; make_projection (zeta = P phi, k<d misspecification)
  posterior.py           #   RewardPosterior: reward posterior over grader weights (Laplace + importance reweighting, MH-validated); VaR lower_bound
  metrics.py             #   regret R_t (vs the same-beta oracle that knows w_t), rangenorm, rank_correlation, coverage, bound_width, variance_ratio (mechanism)
  runner.py              #   Config + run(): the closed loop  deploy -> query -> label -> fit -> drift
  mdp.py                 #   tabular MDP + fitted-Q agent + MDP grader (Stage E: RL with TD bootstrapping)
  tandem.py              #   active/passive tandem test (Ostrovski et al. 2021) for endogeneity
  sweeps.py              #   sweep helpers

analysis/                # experiment drivers (each writes runs/*.parquet and/or figs/*.png)
  run_stage_a.py         #   Plan 01 Stage A: drift-effect existence (regret under drift)
  run_stage_b.py         #   Plan 01 Stage B: monitor blindness (held-out accuracy rises while regret rises)
  run_detectability.py   #   detector taxonomy: fixed-anchor vs co-moving queries
  run_p0.py              #   Plan 02 P0/HG0 precondition (ensemble disagreement falls as regret rises)
  run_p1.py              #   Plan 02 P1: posterior-VaR coverage + ranking under drift  (k x lambda x breadth x alpha)
  run_p2_mlp.py          #   Plan 02 P2: MLP-ensemble conservative bound (HG4 penalty inversion)
  run_hg7.py             #   Plan 02 HG7: does model capacity amplify the coverage failure?
  brex_demo.py           #   Bayesian-REX decision demo: selection regret + violation under misspec+drift  (the headline)
  stageF_ranking.py      #   neural-scale ranking analysis: loads a PEBBLE run's ranking_export.pkl (see Stage F)
  p1_analyze.py, figures.py, stats.py   # analysis / plotting / stats helpers

tests/                   # pytest suite (gates downstream analysis)
  test_sanity.py         #   20 checks incl. the exact oracle-tracking regret floor (N7)
  test_posterior.py      #   posterior-correctness gates (PG1/PG4/PG5)

md_files/                # documents
  RESULTS.md             #   consolidated, honest verdicts
  PAPER_DRAFT.md         #   self-contained paper draft (notation, methods, results, related work)
  EXPERIMENT_LOG.md      #   full dated chronology (every variant, including failures)
  STAGE_F_HANDOVER.md    #   scale-up plan for the neural (Metaworld/PEBBLE) experiments

runs/  figs/             # generated outputs (git-ignored)
requirements.txt
```

The **neural-scale (Stage F)** experiments run in a *separate* repo (`NeuralDriftedGrader`, a fork
of B-Pref/PEBBLE) using this package for analysis — see
[`md_files/STAGE_F_HANDOVER.md`](md_files/STAGE_F_HANDOVER.md).

---

## Installation

CPU-only; Python 3.8+.

```bash
pip install -r requirements.txt      # numpy, scipy, pandas, matplotlib, pytest
```

`torch` is **not** needed here (only for the Stage F PEBBLE integration in `NeuralDriftedGrader`).

---

## Quick start

```bash
# 1. Sanity gates (must pass before trusting any analysis)
pytest tests/ -q

# 2. Plan 02 core: posterior-VaR coverage + ranking under drift  -> runs/p1_rows.parquet
python -m analysis.run_p1 --seeds 20 --workers 8

# 3. The headline: Bayesian-REX decision demo (selection regret + violation)
python -m analysis.brex_demo --seeds 20 --workers 8            # -> runs/brex_demo.parquet
```

Each driver prints a summary table and writes a parquet to `runs/`. All are seeded and
parallelised (`--seeds`, `--workers`); dev seeds are fixed for reproducibility.

### Reproducing the main results

| Result | Command |
|---|---|
| Endogeneity is null / drift-effect existence (Plan 01 A) | `python -m analysis.run_stage_a --seeds 20` |
| Monitor blindness (held-out accuracy misleads) | `python -m analysis.run_stage_b --seeds 20` |
| Fixed-anchor drift detector works | `python -m analysis.run_detectability` |
| VaR coverage robust, ranking collapses under misspec+drift | `python -m analysis.run_p1 --seeds 20 --ks 6 3 --lams 0.0 0.8` |
| Conservative-penalty (HG4) does not invert for neural ensembles | `python -m analysis.run_p2_mlp --seeds 10` |
| Capacity does not amplify the failure (HG7) | `python -m analysis.run_hg7 --seeds 8` |
| **Bayesian-REX picks the wrong policy under misspec+drift** | `python -m analysis.brex_demo --seeds 20` |

Interpretation of every table is in `md_files/EXPERIMENT_LOG.md` (search the driver name).

---

## Key concepts (notation)

- **Grader**: Bradley-Terry preferences, `P(a > b) = sigma(betaT * w . (phi(a) - phi(b)))`,
  weights `w` on the `d`-simplex, rationality `betaT`.
- **Drift (SAT)**: `w_{t+1,i} ∝ w_{t,i} * exp(-lambda * m_i)` — the grader *satiates* on
  whatever features the agent over-produces (`m` = achieved feature profile). Endogenous,
  ordering-changing. `lambda` = drift rate (0 = static).
- **Regret `R_t`**: deployed policy's regret vs the same-temperature oracle that *knows*
  `w_t` — the true-reward yardstick (0 when handed `w_t`).
- **VaR bound**: `LB_alpha(pi)` = `alpha`-quantile of `{ w_i . f_pi }` over posterior
  samples `w_i`. **Marginal coverage** = is `w_t . f_pi >= LB`? **Ranking validity** =
  does `LB` order *alternative* policies like their true value does? (The two come apart.)
- **Misspecification**: reward model sees only `zeta = P phi` (`k < d`); the drift's
  component in `null(P)` is invisible to it.

Full definitions in `md_files/PAPER_DRAFT.md` §2.

---

## Neural-scale (Stage F)

The scale experiments (drifting grader integrated into real PEBBLE on Metaworld) live in
the `NeuralDriftedGrader` repo and export `ranking_export.pkl`, which `analysis/stageF_ranking.py`
analyses with this package's posterior/projection/ranking machinery. To reproduce or
scale up (multiple Metaworld tasks × seeds on a cluster), follow
[`md_files/STAGE_F_HANDOVER.md`](md_files/STAGE_F_HANDOVER.md) — it is self-contained
(repos, conda env, reproduction gate, commands, grid, known issues).

---

## Status & caveats

- Findings are **deflationary but precise**: the field's dramatic worry does not
  reproduce; the durable result is the coverage-vs-ranking split under misspecification +
  drift.
- The scale result is currently **single-task (button-press-v2), 2 seeds** — the main
  open item, addressed by the Stage F scale-up.
- The premise (endogenous grader drift) has weak *direct* real-world evidence; the paper
  scopes this honestly (nearest support: recommender feedback loops, performative
  prediction, non-stationary DPO). See `md_files/PAPER_DRAFT.md` §7.
