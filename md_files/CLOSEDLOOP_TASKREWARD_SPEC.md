# Closed-loop task-reward experiment — spec

**Status:** running (SLURM array `14466238`, `0-89%10`, dbrown-gpu-np). Analysis:
`analysis/stageF_closedloop.py`; results → `closedloop/results/`; write-up target §7f of
`STAGE_F_SCALEUP_RESULTS.md` (pending final numbers).

## 1. What it adds

The open-loop results are **selection-side**: fit an RM post-hoc, use its VaR to *rank* policies
under grader drift (the graded-coverage law, Spearman +0.75). This experiment is **closed-loop**:
*optimize* a policy against the RM and measure whether it still does the real task — the difference
between "a bad ranking metric" and "reward hacking." `train_PEBBLE.py` already trains SAC against the
learned ensemble `r_ψ` (relabels the buffer via `r_hat`) and logs real Metaworld reward, so the
experiment = (a) a task-composed grader + (b) the RM representation class as the variable.

## 2. Grader — multi-feature, task-tied, standardized

`drift_feature=task` → `g_t = w_t · φ_task(s,a)`, `φ_task = [reach, place, grip, lift, effic]`
(`drift_features.make_task_features`): reach `exp(-5‖hand−obj‖)`, place `exp(-5‖obj−goal‖)`, grip
`obs[3]`, lift `relu(obj_z)`, effic `1/(1+‖a‖²)` — the ingredients of the Metaworld dense reward,
all non-negative; `w_t` drifts (SAT).

**Standardization (required).** Raw features have very different realised scales, so a feature's
influence on preferences (∝ `w_i·std(φ_i)`) is dominated by high-variance features regardless of
`w`. We divide each feature by a fixed per-step scale (≈ its std) — non-negativity preserved — so
influence ≈ `w_i` and `w0` genuinely sets the criterion mix. With w0=[reach .30, place .45, grip
.01, lift .00, effic .24], place (task completion) then drives ~45% of preferences, so the grader is
actually task-aligned and drift away from w0 moves the standard off-task.

## 3. Design

**Independent variable = RM representation class** (`rm_mode`):
- `full` — trainable MLP ensemble (3×256). Reference.
- `linear` — affine map over `[obs,act]` (`rm_num_layers=0`). Intermediate.
- `frozen_narrow` — **B-REX**: frozen random encoder `[obs,act]→8` (LeakyReLU) + trainable linear
  head (each ensemble member a distinct frozen encoder). The vulnerable case.

The three arms form a coverage gradient (full ≥ linear ≥ frozen), so the closed-loop result can be
graded like the selection law.

**Grid (90):** 5 tasks (button-press, sweep-into, drawer-open, door-open, window-close) × 3 seeds ×
{NONE, SAT λ1.0} × {full, linear, frozen_narrow}, 300k steps.

## 4. Metrics & claim

Primary axis: `eval/true_episode_reward` (dense real env return; success_rate is floor-bound/noisy
on some tasks). Also: `eval/episode_reward` (learned-RM return, for the hacking panel), RM held-out
preference accuracy (the monitor), `drift_log.jsonl` (`w_dist_from_w0`).

**Two effects, kept separate:**
- *Main effect (not the claim):* a frozen/narrow RM is a worse training signal, so it trails `full`
  even at NONE — a capacity gap.
- *Interaction (THE claim):* the **drift-induced** drop, per-arm retention `R = trueR(SAT)/trueR(NONE)`.
  Prediction: `R_full ≥ R_linear ≥ R_frozen` — optimizing the frozen RM hacks the learned reward
  (its `episode_reward` holds while real reward falls) while the RM's preference accuracy stays green.

**Analysis requirements:** compute `R` only for cells at the *same converged step* (both ≥280k) —
never pool different training stages; and flag arms whose NONE `trueR` is near a per-task floor
(the ratio is unstable there). `analysis/stageF_closedloop.py` enforces both and prints an
honest verdict (no support unless the pooled interaction exceeds its own spread and is sign-consistent).

## 5. Code (all in `NeuralDriftedGrader`, additive; `rm_mode=full` + raw features reproduce nothing
new so defaults are backward-compatible)

| file | change |
|---|---|
| `drift_features.py` | `make_task_features` (5 standardized MW features) |
| `reward_model.py` | `construct_ensemble` builds the frozen-encoder+linear-head member; `__init__` `rm_mode/rm_frozen_dim/rm_frozen_seed` |
| `train_PEBBLE.py` | threads the rm knobs; `drift_feature=task` branch |
| `config/train_PEBBLE.yaml` | defaults |

## 6. Run scripts (`stageF_scaleup/`)

`gen_cells_cl.py` → `closedloop/manifests/cl_cells.json`; `closedloop_array.slurm` (skip-if-done);
`submit_cl.sh [dep_jobid]` (submits `--array=0-89%10`, optional `afterany` dependency); `smoke_cl.slurm`.
Cost ≈ 90 × ~3 h ≈ 270 GPU-h (~overnight at ~8 concurrent).
