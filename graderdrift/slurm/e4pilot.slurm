#!/bin/bash
#SBATCH --job-name=gd_e4pilot
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out

# E4 pilot: {S1,S2,S5,S7a,S8t} + controls {S0,S7null} x 10 seeds, informed tracker.
set -euo pipefail
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/graderdrift/bin/python
REPO=/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/graderdrift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$PY" -m graderdrift.experiments.e4_pilot --out "$REPO/runs/e4pilot" --seeds 0-9 --nproc 14
echo "DONE"
