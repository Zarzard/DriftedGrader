#!/bin/bash
#SBATCH --job-name=gd_e5
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out

# E5: arms {accumulate,window,refresh} x {S1,S2,S5,S0} x 20 seeds, budget B=100.
set -euo pipefail
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/graderdrift/bin/python
REPO=/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/graderdrift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$PY" -m graderdrift.experiments.e5 --out "$REPO/runs/e5" --seeds 0-19 --nproc 16
echo "DONE"
