#!/bin/bash
#SBATCH --job-name=s1gsm12
#SBATCH --partition=dbrown-gpu-grn
#SBATCH --account=dbrown
#SBATCH --requeue
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --qos=dbrown-gpu-grn
#SBATCH --mem=28g
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --output=%x_%A_%a.out
# ENDOGENOUS INTENSITY SWEEP. Three intensities per mechanism, so the four endogenous arms can be
# compared to each other and to the rotations on a common scale.
#
# GSM8K, 12 rounds. The whole closed-loop program so far is UltraFeedback; the GSM8K runs that
# exist used only oracle and stationary at 3 rounds. This is the first drift series on the one task
# here where GOLD IS VERIFIABLE, so a harmful drift can be checked against whether the policy got
# the answers right rather than against another model's opinion. --gold-model is dropped for the
# same reason: gold is exact match, not ArmoRM.
#
# WHY GRANITE AND rtx6000. These arms compare only against each other, never against the a6000 UF
# runs, so there is no cross-hardware reproducibility to protect, and dbrown-gpu-grn is our OWNER
# queue there: no preemption, 14-day wall. l40s nodes are excluded deliberately -- sm_89 is in
# NEITHER environment's arch list (bpref runs sm_50..sm_90 skipping 89, vllmgen sm_75/80/86/90/100/
# 120) so they would fall back to JIT. rtx6000 is sm_75 and h100 is sm_90, both covered.
#
# INDEX 6 IS THE ORACLE, added after the first six were already queued. Without it the series can
# say how much gold a drift costs relative to no drift, but not how much was available to lose.
# The earlier three-round LLaMA3-SFT pair on this grader says the headroom is small: oracle 0.780,
# 0.816, 0.850, 0.848 against stationary 0.776, 0.818, 0.850, 0.838, a gap of 0.010 at round 3 and
# inside this task's 0.022 binomial standard error. If that holds at twelve rounds it is itself a
# result -- GSM8K could then only show "drift makes an already-flat process worse", never "drift
# turns a beneficial process harmful", and the report should say so rather than imply the stronger
# claim. The oracle replaces the grader's choice with the answer key; the grader never moves, so
# this is a selection-quality control and not a drift arm.
#
# ARMS. Gold is exact-match accuracy on 500 problems, so the binomial standard error is about 0.022
# per run, five times the UF loop's 0.0048. Small effects are not measurable here and the arms are
# chosen for size: two stationary runs to pin the baseline and give one replicate, then max89
# (+0.052 on UF), rot105 (+0.207), mom_eta50 (+0.532) and sat_eta40. matched75 is omitted because
# its UF effect, +0.027, sits inside this task's noise.
#
# NO --gold-filter. The first attempt (job 2126950, cancelled 21 h in, output under
# _aborted_goldfilter/) carried it over from the old three-round GSM8K script. It keeps only LIVE
# pools, those holding both a correct and an incorrect sample, and that creates a feedback loop the
# UltraFeedback arms do not have: as the policy improves, pools go all-correct and are discarded, so
# training support shrinks onto the hardest prompts. Live went 0.559, 0.447, 0.397 over rounds 1-3
# while all-correct rose 0.411 -> 0.587; DPO fit the remnant at margin>0 0.950 and loss 0.178
# against UltraFeedback's 0.855 and 0.337; and BOTH stationary arms collapsed at round 4, 0.832 to
# 0.604 and 0.848 to 0.504. Their replicate spread reached 0.168, eight times the binomial standard
# error, so the baseline was less reproducible than the arms it had to calibrate.
#
# The drift arms survived that only by being inert: max89's DPO loss sat at 0.69, which is ln 2, so
# its pairs carried no separable signal and its policy never moved. Harm against stationary came
# out NEGATIVE, which would have read as drift protecting against over-optimisation. UltraFeedback
# runs the same 3 epochs without the filter and is stable for twelve rounds, so dropping it is also
# what makes the two venues comparable.
#
# BETA 0.6, from the calibration (job 2150332, five rounds, stationary only, four values):
#   0.1  chars 661 642 716 1883 1532 2189   acc 0.772 0.850 0.796 0.138 0.330 0.060   <- the failure
#   0.3  chars 665 638 663  726  909  537   acc 0.758 0.828 0.840 0.816 0.730 0.816   <- wobbles
#   0.6  chars 664 674 682  709  706  701   acc 0.764 0.814 0.834 0.840 0.840 0.840   <- flat, best
#   1.0  chars 665 684 694  683  700  708   acc 0.784 0.806 0.818 0.810 0.816 0.808   <- flat, lower
# 0.1 reproduces the collapse exactly, which confirms beta was the cause rather than the gold
# filter or the grader. 0.3 would have been chosen on round-2 evidence and is NOT safe: the
# separation only appears at rounds 4-5, where its length swings 663 -> 909 -> 537 and accuracy
# dips to 0.730. 0.6 holds length within 45 chars across all six measurements and is also the most
# accurate arm, so the tighter anchor is not costing learning here -- it is preventing the thing
# that destroys it. 1.0 is equally stable but plateaus lower, so 0.6 is the smallest value that
# works and the best one.
#
# ROUNDS AND WALL. 12 to match UF, where momentum peaked at round 7 and rot105 was still falling at
# 12. rtx6000 is roughly half an a6000 and a GSM8K round moves a similar token count to a UF one
# (3000 x 8 x 400 against 1500 x 8 x 512), so budget 5-6 h per round and 120 h of wall. --resume is
# NOT set, matching the UF arms: the endogenous arms accumulate a criterion profile that no saved
# adapter reconstructs.
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
ARMS=(stationary       stationary      matched_random  matched_random  momentum        satiation oracle)
KNOB=(""              ""              "--drift-deg 89" "--drift-deg 105" "--drift-eta 0.50" "--drift-eta 0.40" "")
TAGS=(stationary      stationary_rep  max89           rot105          mom_eta50       sat_eta40 oracle)
i=$SLURM_ARRAY_TASK_ID
A=${ARMS[$i]}
# No --resume anywhere: every tag is a fresh output tree, and the endogenous arms accumulate a
# criterion profile across rounds that no saved adapter can reconstruct, so a resume would silently
# restart the mechanism from a blank profile.
echo "=== host $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) arm=$A tag=${TAGS[$i]} knob=${KNOB[$i]} rounds=12 $(date) ==="
COMMON="--task gsm8k --policy RLHFlow/LLaMA3-SFT --policy-step dpo --reward-model none \
        --grader qrm_v1 --grader-drifted qrm_v2 \
        --drift-round 2 --gen-backend ${BACKEND:-vllm}"
"$PY" stage1.py --arm "$A" $COMMON --rounds 12 --n-train 3000 --n-eval 500 \
      --n-samples 8 --max-new 400 --batch 8 --epochs 3 --accum 64 --n-shot 8 --dpo-beta 0.6 \
      ${KNOB[$i]} \
      --out "$D/gsm12_${TAGS[$i]}"
echo "DONE rc=$?"
