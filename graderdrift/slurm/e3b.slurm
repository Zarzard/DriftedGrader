#!/bin/bash
#SBATCH --job-name=gd_e3b
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.out

# E3 completion batch: the generator-matched "informed" mode over all six schedules,
# plus the S3 population cell for the two original modes. Skip-if-done, so
# resubmission is safe.
set -euo pipefail
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/graderdrift/bin/python
REPO=/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/graderdrift
OUT=${E3_OUT:-$REPO/runs/e3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "host=$(hostname) out=$OUT"
"$PY" -m graderdrift.experiments.e3 --out "$OUT" --seeds 0-19 --K 60 --nproc 16 \
      --modes informed --schedules S0,S1,S2,S3,S4,S5
"$PY" -m graderdrift.experiments.e3 --out "$OUT" --seeds 0-19 --K 60 --nproc 16 \
      --modes known_calendar,blind --schedules S3
echo "DONE"
