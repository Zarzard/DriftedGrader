#!/bin/bash
#SBATCH --job-name=gd_e6rate
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=0:40:00
#SBATCH --output=%x_%j.out

# E6 rate replication: the 2x2 at eta = 1deg and 4deg/round (160 A1 runs).
set -euo pipefail
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/graderdrift/bin/python
REPO=/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/graderdrift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$PY" -m graderdrift.experiments.e6 --out "$REPO/runs/e6" --seeds 0-19 --nproc 16 --eta 1
"$PY" -m graderdrift.experiments.e6 --out "$REPO/runs/e6" --seeds 0-19 --nproc 16 --eta 4
echo DONE
