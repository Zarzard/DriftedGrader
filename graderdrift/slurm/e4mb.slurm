#!/bin/bash
#SBATCH --job-name=gd_e4mb
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out

# Model-based (oracle-free) harm predictor: ~13k soft-VI solves over logged m_k.
set -euo pipefail
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/graderdrift/bin/python
REPO=/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/graderdrift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$PY" -m graderdrift.analysis.e4_modelbased --compute --nproc 16
echo "=== equivalence: point estimates + H2-H2.5 CI ==="
"$PY" -m graderdrift.analysis.e4_predictors
echo "DONE"
