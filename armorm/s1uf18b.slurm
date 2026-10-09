#!/bin/bash
#SBATCH --job-name=s1uf18b
#SBATCH --partition=dbrown-gpu-np
#SBATCH --account=dbrown-gpu-np
#SBATCH --requeue
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=28g
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
#SBATCH --output=%x_%A_%a.out
# ENDOGENOUS INTENSITY SWEEP. Three intensities per mechanism, so the four endogenous arms can be
# compared to each other and to the rotations on a common scale.
#
# REPLICATION AT A NEW SEED, 18 ROUNDS. Two gaps a reviewer would hold against the 12-round
# series. First, every arm there ran seed 0, and the only variance estimate comes from six
# identical-config replicates at 0% and 46-49% flips plus one homoscedasticity check at 70%. The
# arms carrying the headline run at 92-100% flips, where variance has never been measured at all.
# Second, 12 rounds truncates a non-monotone trajectory: momentum peaks at round 7-9 and then
# partly recovers, and nobody knows whether the recovery continues or reverses.
#
# WHAT --seed VARIES, which is deliberately MORE than the existing replicates varied. It selects
# the training prompts (stage1.py:770) and, for matched_random only, the rotation direction
# (seed+17). The s1uf12g replicates held both fixed and re-ran the identical config, which measures
# training nondeterminism alone -- correct for a null, too narrow for a robustness claim. Here the
# seed moves, so rot105's replicates share its MAGNITUDE but not its direction. That is a stronger
# test, since the selection study found direction matters enormously at fixed angle, and it must be
# described as such rather than as a repeat of the same drift.
#
# Rounds 1-12 remain directly comparable to the seed-0 runs round by round; 13-18 are new.
#
# THIS ARRAY IS THE BULK: every other distinct configuration, two seeds each. Lower value per run
# than s1uf18a, since most of these sit in the noise band where six replicates already exist, so it
# is queued behind it and is the one to cancel if the node is needed.
#
# Why this is needed. The exogenous arms have always had --drift-deg and were calibrated with it:
# 75 degrees was chosen to match a target flip rate and 89 to sit near the ceiling for a rotation of
# at most 90 degrees. The endogenous arms had no such flag. ETA, LAM and ALPHA were module constants,
# so all four ran at whatever those happened to give, and the measured per-round angles came out at
# 19.0 (satiation), 19.4 (momentum), 13.7 (assimilation) and 0.9-2.1 (acclimation). Reporting
# "acclimation has no effect" beside "satiation has a small effect" was therefore comparing a 2-degree
# drift with a 19-degree one. The three new flags fix that; their defaults are the old constants, so
# every arm already run is reproduced exactly.
#
# Calibration. Done offline on the round-0 LLaMA3-SFT pool (pools/uf_llama3sft.npz), where QRM v1's
# linear model reproduces 1.0000 of its own picks, iterating each mechanism's own recursion on the
# frozen pool. Anchors there: 45 deg -> flip 0.140, 75 -> 0.447, 89 -> 0.670.
#
#   satiation     eta 0.25/0.32/0.40 -> flip 0.147/0.275/0.569 at 19/28/41 deg. This is the only
#                 ladder that crosses the 0.45 flip rate where the selection study puts the harm
#                 threshold, so it is the arm that tests whether the loop reproduces that threshold.
#                 eta 0.45 is deliberately skipped: it hits 0.878 at round 2 then self-corrects to
#                 0.10, so it measures the oscillation rather than the intensity.
#   momentum      eta 0.25/0.50/0.90 -> 19/45/84 deg. Flip CANNOT be predicted offline: its direction
#                 is a profile DIFFERENCE, which is identically zero on a frozen pool because the
#                 profile is a fixed point when the weights do not move. It is not dead online -- the
#                 running arm logs 0.135 flips and -0.010 gold at round 3. Note it reaches 0.135 at
#                 19 deg where a random rotation needs 45, so its direction is the most churn-
#                 productive per degree of any arm here: it points along what the policy is actively
#                 changing, which is where the pool discriminates. Flips get read off the runs.
#   acclimation   lam 1/3/9 -> terminal flip 0.003/0.019/0.294. Each setting is a RAMP, not a level,
#                 because r accumulates: lam 9 goes 0.017 at round 2 to 0.294 by round 11. lam 27 is
#                 skipped, it oscillates between 0.28 and 0.73 across rounds.
#   assimilation  eta 0.25/0.50/0.99 -> 0.046/0.088/0.155. This ladder documents a CEILING rather
#                 than testing harm, and the ceiling is structural: assimilation rotates w0 toward
#                 the profile direction and cannot pass it, and that direction is 58.4 deg away
#                 (cos 0.524). At eta 1.0 the weights sit exactly on it, 59.8 deg from the start, and
#                 still move only 15.5% of picks, where a random 59-deg rotation moves about 30%.
#                 Chasing your own strengths is a large weight change that changes few decisions.
#                 Offline predicts 0.045 at eta 0.25 and the loop logs 0.046, so this transfers.
#
# Index 8, sat_cum25, is a SENSITIVITY arm, not another intensity. The endogenous arms re-derive
# their blend from the current undrifted w0 each round, so their displacement never accumulates:
# satiation sits 19.0 deg off-true at round 2 and still 19.0 at round 7. Acclimation is the one
# exception, accumulating through its r EWMA. Nothing in the literature settles which is right --
# Helson gives a pooling of present stimuli rather than a recursion, Lant fits free coefficients
# rather than an EWMA, Panickssery and Zheng measure a static bias with no loop, and satiation and
# momentum have no source at all -- so the temporal form is our choice for all four. The choice is
# not free: at the SAME eta=0.25, accumulating reaches 79.8 deg by round 5 against 19.3 flat, which
# is roughly 0.8 flips against 0.15. This arm is round-matched against the running satiation arm and
# measures how much of the endogenous no-harm result rests on that unsupported choice.
#
# Everything else matches s1uf12b/c/d exactly: same task, policy, grader pair, gold, drift round 2,
# 12 rounds, so these arms are directly round-matched against the completed stationary baseline.
set -o pipefail
Z=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan
PY=$Z/miniconda3/envs/bpref/bin/python
export LD_LIBRARY_PATH="$Z/miniconda3/envs/bpref/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$Z/graderdrift_runs/pylibs:/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/armorm"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$Z/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
"$PY" -c "import sentencepiece" || { echo "NO SENTENCEPIECE (check PYTHONPATH)"; exit 2; }
cd /uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/armorm
D=$Z/graderdrift_runs/stage1
ARMS=(stationary stationary matched_random matched_random matched_random matched_random matched_random matched_random matched_random matched_random release_swap_real release_swap_real satiation satiation satiation satiation satiation satiation momentum momentum momentum momentum acclimation acclimation acclimation acclimation acclimation acclimation acclimation acclimation self_preference self_preference self_preference self_preference self_preference self_preference self_preference self_preference)
KNOB=("" "" "--drift-deg 45" "--drift-deg 45" "--drift-deg 75" "--drift-deg 75" "--drift-deg 89" "--drift-deg 89" "--drift-deg 89 --drift-schedule ramp" "--drift-deg 89 --drift-schedule ramp" "" "" "" "" "--drift-eta 0.32" "--drift-eta 0.32" "--drift-eta 0.40" "--drift-eta 0.40" "" "" "--drift-eta 0.50" "--drift-eta 0.50" "" "" "--drift-lam 3" "--drift-lam 3" "--drift-lam 9" "--drift-lam 9" "--drift-lam 9 --drift-alpha 1.0" "--drift-lam 9 --drift-alpha 1.0" "" "" "--drift-eta 0.50" "--drift-eta 0.50" "--drift-eta 0.99" "--drift-eta 0.99" "--drift-eta 0.25 --drift-cumulative" "--drift-eta 0.25 --drift-cumulative")
TAGS=(stationary_s1 stationary_s2 rot45_s1 rot45_s2 matched75_s1 matched75_s2 max89_s1 max89_s2 ramp89_s1 ramp89_s2 release_swap_real_s1 release_swap_real_s2 satiation_s1 satiation_s2 sat_eta32_s1 sat_eta32_s2 sat_eta40_s1 sat_eta40_s2 momentum_s1 momentum_s2 mom_eta50_s1 mom_eta50_s2 acclimation_s1 acclimation_s2 acc_lam3_s1 acc_lam3_s2 acc_lam9_s1 acc_lam9_s2 acc_lam9_a1_s1 acc_lam9_a1_s2 assimilation_s1 assimilation_s2 asm_eta50_s1 asm_eta50_s2 asm_eta99_s1 asm_eta99_s2 asm_cum25_s1 asm_cum25_s2)
SEED=(1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2)
i=$SLURM_ARRAY_TASK_ID
A=${ARMS[$i]}
# No --resume anywhere: every tag is a fresh output tree, and the endogenous arms accumulate a
# criterion profile across rounds that no saved adapter can reconstruct, so a resume would silently
# restart the mechanism from a blank profile.
echo "=== host $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) arm=$A tag=${TAGS[$i]} knob=${KNOB[$i]} rounds=12 $(date) ==="
COMMON="--task uf --policy RLHFlow/LLaMA3-SFT --policy-step dpo --reward-model none \
        --grader qrm_v1 --grader-drifted qrm_v2 --gold-model armorm \
        --drift-round 2 --gen-backend ${BACKEND:-vllm}"
"$PY" stage1.py --arm "$A" $COMMON --seed ${SEED[$i]} --rounds 18 --n-train 1500 --n-eval 500 \
      --n-samples 8 --max-new 512 --batch 8 --epochs 3 --accum 32 \
      ${KNOB[$i]} \
      --out "$D/uf18_${TAGS[$i]}"
echo "DONE rc=$?"
