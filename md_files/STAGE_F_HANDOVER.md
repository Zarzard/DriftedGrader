# Stage F Scale-Up — CHPC Handover

**Purpose.** Everything a fresh agent needs to scale up the "drifted-grader" Stage F
experiments on a cluster, without re-deriving anything. Read top to bottom once, then
use §5 (commands) and §6 (plan) to execute.

**TL;DR of the science.** A reward-posterior value-at-risk (VaR) bound — the
Bayesian-REX safe-RL primitive — stays *honest about certifying a fixed policy* under
grader drift, but its *policy selection / ranking* corrupts under representational
misspecification + drift. Dramatic in the tabular/closed-form limit; **muted but
measurable** at neural scale (one Metaworld task so far). The scale-up job is to
confirm the neural-scale effect across **multiple validated Metaworld tasks × more
seeds**, and (optionally) sharpen the nonlinear-misspec axis.

---

## 1. Repos, branch, environment

| Thing | Value |
|---|---|
| PEBBLE/B-Pref repo (drift runs) | GitHub `git@github.com:Zarzard/NeuralDriftedGrader.git` (fork of `rll-research/BPref`); on this box `/home/zifan/NoisyPbRL` |
| Branch | `main` (Stage F drift code lives here) |
| Research hub (analysis, tabular sim) | GitHub `git@github.com:Zarzard/DriftedGrader.git`; on this box `/home/zifan/DriftedGrader` |
| Conda env (has mujoco/dmc2gym/metaworld + torch-cuda) | `bpref` → `/home/zifan/miniconda3/envs/bpref/bin/python` |
| Headless MuJoCo | `MUJOCO_GL=egl` (required) |
| Thread pinning per run | `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` (avoid BLAS oversubscription; ~12–16 single-runs fit 24 cores) |
| Throughput (state-based, single run) | ~100–160 env-steps/s → a 300k run ≈ 30–60 min; 500k ≈ 1–2 h. GPU is NOT the bottleneck (tiny MLPs); CPU/env-stepping is. |

**Clone both repos on the cluster.** The PEBBLE repo produces the runs; this analysis
repo analyses them. Both are pushed and current (branch `main`):
```bash
git clone git@github.com:Zarzard/NeuralDriftedGrader.git   # PEBBLE + Stage F drift code
git clone git@github.com:Zarzard/DriftedGrader.git         # tabular sim + analysis (stageF_ranking.py)
```
The `/home/zifan/...` paths in §5 commands are this-box examples — substitute your clone
locations on the cluster (the PEBBLE clone dir is `NeuralDriftedGrader/`, not `NoisyPbRL/`).

---

## 2. The reproduction gate (do this FIRST for every new task)

**Only `metaworld_button-press-v2` is validated.** Vanilla PEBBLE on it reproduces
(100% success, true return ~4500). Do NOT trust drift results on a task until vanilla
PEBBLE reaches its expected success on that task.

- **DMC is OUT.** PEBBLE-on-DMC (walker_walk) does NOT learn in this repo (SAC-with-true-reward
  reaches ~950, but PEBBLE's proxy return never climbs — the failure is in the
  preference→policy path, not our mods: `r_hat`/`relabel_with_predictor` are byte-identical
  to upstream). Stay on Metaworld.
- **Reproduction command (canonical PEBBLE, oracle teacher):** the repo's own script is
  `scripts/<task>/1000/noisy/run_PEBBLE.sh`. Inline for one seed:
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MUJOCO_GL=egl \
/home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE.py \
  env=metaworld_<TASK> seed=12345 gpu=0 \
  agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 \
  gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 \
  agent.params.batch_size=512 \
  double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
  diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
  reward_update=10 num_interact=5000 max_feedback=10000 reward_batch=50 feed_type=1 \
  teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 \
  hydra.run.dir=/abs/out/dir
```
  Check `eval.csv` `success_rate` reaches the task's known PEBBLE level. Candidate tasks
  to validate: `sweep-into-v2`, `drawer-open-v2`, `door-open-v2`, `window-close-v2`,
  `hammer-v2` (the collaborator's earlier `perf_diag` runs touched these).

Notes: this old hydra does NOT resolve `${oc.env:...}` in the output-dir template — always
pass `hydra.run.dir=<abs path>` explicitly. Config is `strict=True` → only pass keys that
exist. `use_wandb` defaults false; `eval.csv`/`train.csv` are written regardless.

---

## 3. Current results (what's established)

**Tabular (sharp) — `DriftedGrader/analysis/`:**
- Ranking (`run_p1.py`): well-spec (k=d) rank robust across drift; misspec (k<d) rank
  0.78 (no drift) → **−0.29** (strong drift) — inversion. Marginal coverage robust (over-covers).
- **Bayesian-REX decision demo (`brex_demo.py`) — the money shot.** Select argmax-VaR
  policy from a diverse pool; 4 cells × 20 seeds:

  | spec | drift | regret_brex | regret_naive | wrong-pick | violation |
  |---|---|---|---|---|---|
  | well-spec | NONE | 0.000 | 0.000 | 0.000 | 0.082 |
  | well-spec | SAT | 0.039 | 0.035 | 0.132 | 0.000 |
  | misspec | NONE | 0.229 | 0.218 | 0.314 | 0.009 |
  | **misspec** | **SAT** | **0.640** | 0.654 | **0.909** | **0.000** |

  Under misspec+drift B-REX picks the **wrong policy 91%** of the time (near-worst-case
  regret), yet `violation≈0` (coverage of the *chosen* policy holds) and `regret_brex ≈
  regret_naive` (VaR conservatism gives **no protection**). Thesis: honest about "is THIS
  policy safe?", broken about "WHICH policy?".

**Scale (muted, single task `button-press-v2`, 2 seeds, 300k):**
- Dynamics: NONE `w_dist=0`, `J_init==J_current`. SAT `w_dist` 0.52–0.65 (argmax flips),
  `J_current` collapses 0.94→0.57 below `J_init` 0.82 — ordering-changing drift +
  grader-tracking failure confirmed at scale.
- Ranking (`stageF_ranking.py`): full 1.00→0.87; kproj(k=3) 0.88→0.55; adv 0.86→**0.21**;
  small (nonlinear MLP) −0.48/−0.45 = **inconclusive** (see §7).
- Decision (selection regret): full 0.00→0.00; kproj 0.08→0.21; adv 0.04→**0.26**.
  Well-spec selection robust to drift; misspec selection regret 2.5–6× under drift. No
  inversion (regret 0.26 not 0.64) — **muted vs tabular, fits the deflationary arc.**

---

## 4. Code map

**`NeuralDriftedGrader` (a.k.a. local `NoisyPbRL`, branch `main`) — the drift integration:**
- `drifting_reward_model.py` — `DriftingRewardModel(RewardModel)`. Overrides `get_label`
  ONLY: replaces per-step env reward with `g_t = w_t·φ(s,a)`, then calls parent
  byte-identically. `drift_step()` advances `w_t` (SAT/EXO/REPLAY/NONE) once per feedback
  round. Exports per-round comparison design (`sphi1`, `sphi2` = per-segment summed
  features — needed for a *nonlinear* reward; linear uses `dphi=sphi1−sphi2`) + `w_t` + `m_t`.
- `drift_features.py` — `make_random_projection_features` (turnkey, task-agnostic, d=6),
  `make_selected_features` (hand dims).
- `train_PEBBLE.py` — drift wired in behind `cfg.drift_enabled` (default false; vanilla path
  untouched). `_advance_drift()` logs `drift_log.jsonl` (per round: `w`, `w_dist_from_w0`,
  `J_init`, `J_current`, `w_argmax`) and dumps `ranking_export.pkl` (list of rounds, each
  `{step,t,w,m,sphi1,sphi2,lbl}`), then advances drift.
- `config/train_PEBBLE.yaml` — `drift_enabled`, `drift_model` (NONE|SAT|EXO|REPLAY),
  `drift_lambda`, `drift_alpha`, `drift_seed`, `drift_feature` (random|selected),
  `drift_feature_dim`. All additive; defaults keep vanilla behavior.

**`DriftedGrader/analysis/` — tabular sim + scale analysis:**
- `run_p1.py` — tabular posterior-VaR coverage + rank under drift (k×λ×breadth).
- `brex_demo.py` — tabular Bayesian-REX decision demo (selection regret + violation).
- `stageF_ranking.py` — **the scale analyzer.** Loads `ranking_export.pkl` from PEBBLE runs,
  reuses `drifted_grader.{posterior,reward_models,metrics}`. Conditions: `full` (well-spec
  linear), `kproj` (random k<d), `adv` (k<d blind to the drift direction u=w_final−w_initial),
  `small` (tiny nonlinear MLP ensemble — see §7). `--exp_glob`, `--k`.
- `drifted_grader/` package — arms, graders, `RewardPosterior`, `make_projection`,
  `rank_correlation`, `policy_from_scores`, `expected_features`, `sample_pairs`, `rangenorm`.

**Run launcher (template):** `run_stageF_batch.sh <LAM> <STEPS>` (currently in the working
scratchpad; reproduced in §5). Launches NONE+SAT × 2 seeds, 2/GPU, waits.

---

## 5. Commands (copy-paste)

**One drift run (produces `drift_log.jsonl` + `ranking_export.pkl`):**
```bash
cd /home/zifan/NoisyPbRL
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl \
/home/zifan/miniconda3/envs/bpref/bin/python train_PEBBLE.py \
  env=metaworld_<TASK> seed=<SEED> gpu=0 \
  drift_enabled=true drift_model=<NONE|SAT> drift_feature=random drift_feature_dim=6 \
  drift_lambda=1.0 drift_seed=0 \
  agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh \
  num_seed_steps=1000 num_unsup_steps=9000 num_train_steps=300000 agent.params.batch_size=512 \
  double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
  diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
  reward_update=10 num_interact=5000 max_feedback=10000 reward_batch=50 feed_type=1 \
  teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 \
  hydra.run.dir=/home/zifan/NoisyPbRL/exp/stageFrank/<TASK>_<NONE|SAT>_seed<SEED>
```
`drift_model=NONE` = static synthetic-feature grader (the control: same teacher, no drift).
`drift_model=SAT` = the drifting grader. Match seeds across NONE/SAT for pairing.

**Analysis (rank + selection regret) after a batch:**
```bash
cd /home/zifan/DriftedGrader
python -m analysis.stageF_ranking --exp_glob '/home/zifan/NoisyPbRL/exp/stageFrank/*' --k 3
# selection-regret metric: see the inline snippet in EXPERIMENT_LOG / ask; it reuses
# _load,_pool,_adv_projection,_fit_linear from stageF_ranking and reports argmax-VaR regret.
python -m analysis.brex_demo --seeds 20 --workers 8      # tabular decision demo (reference)
```

---

## 6. Scale-up plan (the actual CHPC job)

**Goal:** turn the single-task scale result into a multi-task, multi-seed confirmation that
(a) well-specified selection is robust to drift, (b) misspecified selection regret rises
under drift, (c) marginal coverage/violation stays honest. Expected outcome: muted-but-
consistent with §3 (deflationary).

**Grid:**
- **Tasks:** reproduction-validate each (§2) first, then keep the passers. Target 3–5:
  `button-press-v2` (validated) + `sweep-into-v2`, `drawer-open-v2`, `door-open-v2`,
  `window-close-v2` (validate before use).
- **Drift arm:** `{NONE, SAT}` (both `drift_enabled=true`). Optionally sweep
  `drift_lambda ∈ {0.5, 1.0}` for SAT to show monotonicity.
- **Seeds:** 5–10 per cell (currently only 2 — this is the main statistical gap).
- **Misspecification is POST-HOC** (applied in `stageF_ranking.py`: full / kproj / adv,
  and `--k ∈ {1,2,3}`), so you do NOT need separate runs per misspec — only NONE/SAT×λ×seed×task.
- **Steps:** 300k (≈60 feedback rounds, drift settles by ~40) is enough; 500k for margin.
- **teacher_beta=-1** (deterministic) — clean for rank + decision metrics (both scale/affine-
  invariant). Absolute coverage is NOT well-posed with deterministic labels (§7); if you want
  coverage, use `teacher_beta=10` + `bt_normalize_by_gap_std=true` and pin the posterior βT.

**Metrics to report per task, aggregated over seeds (NONE vs SAT × full/kproj/adv):**
1. `rank_corr` (Spearman LB vs true value over eval policies) — `stageF_ranking.py`.
2. `selection_regret` (argmax-VaR pick's normalized regret) — the decision metric (§5 snippet).
3. Drift sanity: `w_dist_from_w0`, `J_init` vs `J_current` from `drift_log.jsonl`.

**Definition of done:** across ≥3 tasks and ≥5 seeds, a clean table showing well-spec
selection regret ≈ 0 under drift while misspec (adv) selection regret rises 2–6× under SAT,
with per-task error bars. That is the neural-scale confirmation of the tabular B-REX result.

**Compute budget:** ~ (tasks × 2 drift × λ × seeds) runs × ~0.5–1 h each. E.g. 4 tasks × 2
drift × 5 seeds = 40 runs ≈ 20–40 GPU-h → trivial on a cluster; ~1 day on the 2-GPU box.

---

## 7. Known issues & caveats (read before trusting numbers)

1. **`small` (nonlinear MLP misspec) axis is unreliable as implemented.** The hand-rolled
   tiny ensemble (`_fit_small`, hidden=8) anti-ranks even under NONE (−0.48) — an under-fit /
   conservative-bound artifact, NOT a drift result. To do this axis properly: use a larger,
   better-trained ensemble (e.g. reuse `drifted_grader.reward_models.EnsembleBT(base='mlp',
   hidden≥32, K=5)` fed with per-segment features `sphi1/sphi2`), and sanity-check that NONE
   ranks positively before reading SAT. Until then, report only the linear conditions
   (full/kproj/adv), which are clean.
2. **Absolute coverage at scale is ill-posed** with deterministic (or saturated-BT) labels —
   the reward is identified only up to scale, so `V=w_t·m` vs a scale-unpinned posterior is
   meaningless (the affine-invariance issue). Use scale/affine-invariant metrics: `rank_corr`
   and `selection_regret`. The tabular `violation` metric works because βT is known there.
3. **Task competence is NOT testable** with random-projection features — the synthetic grader
   `w_t·φ(random)` is decoupled from the task reward, so `success_rate=0` is expected, not a
   harm signal. If you want "does chasing drift hurt the real task", use
   `drift_feature=selected` with `drift_obs_indices` tied to task-relevant obs dims so the
   grader correlates with task progress (requires per-task obs-layout knowledge).
4. **Single-task, 2-seed** current scale result — the muted effect could be genuine
   (deflationary) or under-powered; multi-task multi-seed (§6) resolves this.
5. **The premise (endogenous grader drift) has weak direct real-world evidence.** Nearest
   support: recommender feedback loops, performative prediction, the DPO-preference-drift paper
   (arXiv:2407.18676). The paper should scope this honestly or add a real-grader mini-demo.
6. **`J_current < J_init` under SAT is partly mechanical** (satiation down-weights exactly what
   the agent produces). Treat it as a validation/sanity signal, not a headline finding.

---

## 8. What the paper needs next (context for prioritization)
- The scale-up above (multi-task selection-regret) = the core neural-scale evidence.
- A mitigation/detector (e.g. fixed-anchor drift monitor gating cross-policy ranking) — a
  Plan-01 positive result to fold in.
- Honest scoping of the drift premise, or a small real-grader (LLM-judge) demo.
- Positioning vs Bayesian-REX (arXiv:2002.09089) — our result is a documented failure mode of
  its policy-selection/reward-hacking-detection use case under drift — and the 2024–26
  reward-uncertainty / preference-drift literature.
