#!/bin/bash
#SBATCH --job-name=cap_sweep
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cap_%A_%a.out
#SBATCH --error=logs/cap_%A_%a.err
#
# One (method, capacity) configuration per array task. Each task fits 25 seeds
# and writes its own suffixed CSVs, so tasks never contend for a file.
# Merge afterwards with:  python scripts/merge_sweep.py --dir outputs/metrics
#
#   sbatch slurm/slurm_capacity_sweep.sh
#
# Capacities are perfect squares so SOM lattices match K-means budgets exactly
# (4=2x2, 9=3x3, ... 400=20x20). Non-square capacities are skipped for SOMs.

#SBATCH --partition=YOUR_PARTITION_NAME
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stable-subtypes

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs/metrics

PREPPED=${PREPPED:-./data/fd001}
CAPACITIES=(4 9 16 25 49 100 225 400)
METHODS=(kmeans som)

N_CAP=${#CAPACITIES[@]}
M_IDX=$(( SLURM_ARRAY_TASK_ID / N_CAP ))
C_IDX=$(( SLURM_ARRAY_TASK_ID % N_CAP ))
METHOD=${METHODS[$M_IDX]}
CAPACITY=${CAPACITIES[$C_IDX]}

echo "task ${SLURM_ARRAY_TASK_ID}: ${METHOD} @ capacity ${CAPACITY}"

srun python scripts/01_capacity_sweep.py \
    --prepped   "$PREPPED" \
    --out       ./outputs/metrics \
    --method    "$METHOD" \
    --capacity  "$CAPACITY" \
    --seeds     25 \
    --train-only
