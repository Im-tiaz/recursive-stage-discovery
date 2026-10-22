# Recursive Stage Discovery for Early Warning in Multi-Sensor Degradation Monitoring

Analysis code for Ahmed, I. and Soliman, H., "Recursive Stage Discovery for
Early Warning in Multi-Sensor Degradation Monitoring", submitted to Sensors
(MDPI), Intelligent Sensors Section, 2026.

Department of Computer Science and Engineering, New Mexico Institute of Mining
and Technology, Socorro, NM 87801, USA.

## Method

A partition of windowed multi-sensor readings is refined recursively until its
nodes are homogeneous in remaining useful life. Terminal nodes are ranked by
descending median training RUL to form an ordered stage index: a low index
means the asset is far from failure, a high index that it is close.

Evaluation is on engines held out entirely. Adjacent windows from one engine
share 25 of their 30 cycles, so any split separating windows from the same
engine leaks. 00_stage_cmapss.py asserts against this.

## Data

The NASA C-MAPSS benchmark is not redistributed here. Download it from the
NASA Prognostics Center of Excellence Data Set Repository and unzip so that
data/raw/CMAPSSData/ holds train_FD00X.txt, test_FD00X.txt and RUL_FD00X.txt.

## Reproducing

Stage a cohort, then run the stage index and the held-out sweep:

    python scripts/00_stage_cmapss.py --data-dir data/raw/CMAPSSData \
        --subsets FD001 --pairwise --max-engines 200 --out data/fd001_bal
    python scripts/04_recursive_early_prediction.py --prepped data/fd001_bal
    python scripts/04b_early_warning_eval.py --prepped data/fd001_bal
    python scripts/02_holdout_sweep.py --prepped data/fd001_bal

FD002 and FD004 need --conditions 6; without per-regime normalisation,
clustering recovers operating regime instead of engine health.

Supplementary Tables S1 and S2 come from run_supp_sensitivity.sh. Run its
verify step first: it re-stages one cohort at the primary settings and compares
window count and prevalence against the existing cohort, so a recipe mismatch
is caught before the sweep spends node hours.

## Licence

MIT, see LICENSE. The C-MAPSS dataset is distributed by NASA under its own
terms and is not covered by this licence.
