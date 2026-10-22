#!/usr/bin/env bash
# Queue the remaining reviewer-requested runs behind the two live jobs.
# Each block writes to its own directory, so nothing collides with the
# sigma_paired or early_oof outputs already in flight.
set -u
cd "$(dirname "$0")"

echo "=== waiting for sigma_paired and early_oof to finish ==="
while pgrep -u "$USER" -f "sigma_paired.py|04b_early_warning_eval.py" >/dev/null; do
  sleep 60
done
echo "=== node free at $(date) ==="

# ---- review point 12: K-means restart control where the penalty actually is.
# The 10- and 20-restart control was run on FD001, which shows no penalty.
# FD003 and FD004 are where the negative gaps are. One capacity, the
# pre-specified primary budget.
for d in fd004_bal fd003_bal; do
  for ni in 1 10 20; do
    echo "--- restart control: $d n_init=$ni ---"
    OMP_NUM_THREADS=4 python -u scripts/02_holdout_sweep.py \
      --prepped "data/$d" --capacities 400 --seeds 5 --split-seeds 5 \
      --n-init "$ni" --out "outputs/metrics_ninit${ni}_${d}"
  done
done

# ---- review point 11: geometry on a pooled scaler and a pooled PCA basis.
# All four cohorts now share the same 360 features, so a common coordinate
# system is available and removes the objection that silhouettes are compared
# across four independently fitted transforms.
echo "--- pooled-basis geometry control ---"
OMP_NUM_THREADS=4 python -u scripts/terminal_mode_pooled.py \
  --cohorts data/fd001_bal data/fd002_bal data/fd003_bal data/fd004_bal \
  --n-pcs 10 --boot 500 --out outputs/metrics

echo "=== queue complete at $(date) ==="
