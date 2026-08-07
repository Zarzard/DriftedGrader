#!/bin/bash
#SBATCH --job-name=gd_m6dod
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=10G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out

# M6 DoD: {S0,S1,S5} x seeds 0-2 x knob variants {default, tau_pol=0.02, n_pairs=50}.
set -euo pipefail
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/graderdrift/bin/python
REPO=/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/graderdrift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$PY" -m graderdrift.experiments.m6_dod --out "$REPO/runs/m6dod" --seeds 0-2 --nproc 14
echo "DONE"
