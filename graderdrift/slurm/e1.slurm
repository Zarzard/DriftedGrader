#!/bin/bash
#SBATCH --job-name=gd_e1
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out

# E1 (stationarity null): A0/S0 x 20 seeds. Pure-CPU numpy; runs off the login node.
set -euo pipefail
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/graderdrift/bin/python
REPO=/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/graderdrift
OUT=${E1_OUT:-$REPO/runs/e1}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # each seed is single-threaded; parallelism is across seeds
cd "$REPO"
echo "host=$(hostname) out=$OUT"
"$PY" -m graderdrift.experiments.e1 --out "$OUT" --seeds 0-19 --K 60 --nproc 10
echo "DONE"
