#!/bin/bash
#SBATCH --job-name=gd_e4full
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out

# E4 full: (a) A1 harm hierarchy 19 conditions x 20 seeds; (b) A0 attenuation 13 cells x 20.
set -euo pipefail
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/graderdrift/bin/python
REPO=/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/graderdrift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "host=$(hostname)"
"$PY" -m graderdrift.experiments.e4_full --out "$REPO/runs/e4full" --seeds 0-19 --nproc 16
"$PY" -m graderdrift.experiments.e4b   --out "$REPO/runs/e4b"    --seeds 0-19 --nproc 16
echo "DONE"
