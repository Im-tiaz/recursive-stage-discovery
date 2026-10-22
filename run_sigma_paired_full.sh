#!/usr/bin/env bash
# Re-run sigma_paired at the same seed count as the primary analysis so that
# Table 5 and Table 2 are comparably powered.
#
#   primary analysis : 5 engine splits x 10 initialisations = 50 fits per cell
#   first paired run : 3 splits x 3 seeds = 9 fits per cell  (underpowered)
#
# Writes to outputs/metrics_sigma_full so the earlier run is preserved for
# comparison. Roughly 5 sigma values x 5 splits x 10 seeds x 4 cohorts of SOM
# fits at 20x20 plus one K-means reference per split and seed.
set -u
cd "$(dirname "$0")"
mkdir -p logs outputs/metrics_sigma_full

for d in fd004_bal fd003_bal fd001_bal fd002_bal; do
  echo "=== $d  $(date) ==="
  OMP_NUM_THREADS=4 python -u scripts/sigma_paired.py \
    --prepped "data/$d" --capacity 400 \
    --split-seeds 5 --seeds 10 \
    --out outputs/metrics_sigma_full
done
echo "=== complete $(date) ==="
