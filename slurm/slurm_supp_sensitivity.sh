#!/bin/bash
#SBATCH --job-name=supp_sens
#SBATCH --array=0-23
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/supp_%A_%a.out
#SBATCH --error=logs/supp_%A_%a.err
#
# Held-out sweeps for Supplementary Tables S1 and S2, one (setting, cohort)
# per array task so nothing contends for an output file.
#
#   6 settings x 4 cohorts = 24 tasks
#
# Run the staging first, on a login node -- it is cheap and the array tasks
# all read its output:
#
#     ./run_supp_sensitivity.sh stage
#     sbatch slurm/slurm_supp_sensitivity.sh
#     ./run_supp_sensitivity.sh tables
#
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stable-subtypes

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SETTINGS=(base thr10 thr15 thr20 win20 win40)
COHORTS=(fd001_bal fd002_bal fd003_bal fd004_bal)

N_COH=${#COHORTS[@]}
S_IDX=$(( SLURM_ARRAY_TASK_ID / N_COH ))
C_IDX=$(( SLURM_ARRAY_TASK_ID % N_COH ))
SETTING=${SETTINGS[$S_IDX]}
COHORT=${COHORTS[$C_IDX]}

PREPPED="data/supp/${SETTING}/${COHORT}"
OUTDIR="outputs/supp/${SETTING}"
RESULT="${OUTDIR}/holdout_sweep_${COHORT}.csv"

echo "task ${SLURM_ARRAY_TASK_ID}: ${SETTING} / ${COHORT}"

if [ ! -f "${PREPPED}/metadata.csv" ]; then
    echo "ERROR: ${PREPPED} not staged. Run './run_supp_sensitivity.sh stage' first." >&2
    exit 1
fi

if [ -f "$RESULT" ]; then
    echo "already done, skipping: $RESULT"
    exit 0
fi

mkdir -p "$OUTDIR"

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} srun python -u scripts/02_holdout_sweep.py \
    --prepped     "$PREPPED" \
    --capacities  400 \
    --seeds       10 \
    --split-seeds 5 \
    --n-init      1 \
    --out         "$OUTDIR"
