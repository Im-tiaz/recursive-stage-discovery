#!/usr/bin/env bash
# =====================================================================
# Supplementary Tables S1 and S2
#
#   S1  near-failure threshold sensitivity : 10, 12, 15, 20 cycles
#   S2  window-length sensitivity          : 20, 30, 40 cycles
#
# Each non-primary setting re-stages all four cohorts with ONE parameter
# changed and re-runs the held-out sweep at the primary 400-prototype
# budget, with the same 5 splits x 10 initialisations as the main
# analysis, so the supplementary numbers are comparably powered to
# Table 6 rather than merely adjacent to it.
#
# The 12-cycle / 30-cycle cell IS the primary setting. It is not
# re-staged and not re-swept from scratch: "stage" symlinks it to the
# existing data/fd00X_bal directories, so S1's 12-cycle row and S2's
# 30-cycle row come from exactly the staging the main text used.
#
#   ./run_supp_sensitivity.sh verify   # confirm the staging recipe matches
#   ./run_supp_sensitivity.sh stage    # cheap; login node is fine
#   ./run_supp_sensitivity.sh sweep    # heavy; prefer the SLURM array
#   ./run_supp_sensitivity.sh tables   # writes paper/supp_S{1,2}.tex
#
# Run "verify" FIRST. It re-stages one cohort at the primary settings
# into a scratch directory and compares window count and rare-state
# prevalence against your existing data/fd001_bal. If those disagree,
# the recipe below differs from whatever produced your primary cohorts,
# and the whole sweep would be incomparable -- fix the recipe before
# spending node hours on it.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

RAW=${RAW:-data/raw/CMAPSSData}
SENSORS="sensor_2 sensor_3 sensor_4 sensor_6 sensor_7 sensor_8 sensor_9 \
sensor_11 sensor_12 sensor_13 sensor_14 sensor_15 sensor_17 sensor_20 sensor_21"
MAXENG=${MAXENG:-200}
STRIDE=${STRIDE:-5}
TESTFRAC=${TESTFRAC:-0.25}
SEED=${SEED:-1}

COHORTS=(FD001 FD002 FD003 FD004)

# setting -> "near-failure-cut window"; base is the primary setting
SETTING_NAMES=(base thr10 thr15 thr20 win20 win40)
setting_params () {
  case "$1" in
    base)  echo "12 30" ;;
    thr10) echo "10 30" ;;
    thr15) echo "15 30" ;;
    thr20) echo "20 30" ;;
    win20) echo "12 20" ;;
    win40) echo "12 40" ;;
    *) echo "unknown setting: $1" >&2; return 1 ;;
  esac
}

conditions_of () { case "$1" in FD002|FD004) echo 6 ;; *) echo 1 ;; esac; }
tag_of ()        { echo "$(echo "$1" | tr 'A-Z' 'a-z')_bal"; }

stage_one () {   # $1=subset  $2=cut  $3=window  $4=outdir
  python -u scripts/00_stage_cmapss.py \
    --data-dir "$RAW" \
    --subsets "$1" \
    --conditions "$(conditions_of "$1")" \
    --window "$3" --stride "$STRIDE" \
    --pairwise \
    --bins "$2" 50 125 \
    --max-engines "$MAXENG" \
    --sensors $SENSORS \
    --test-frac "$TESTFRAC" --seed "$SEED" \
    --out "$4"
}

summarise () {   # $1=metadata.csv -> "windows prevalence"
  python - "$1" <<'PY'
import sys, pandas as pd
md = pd.read_csv(sys.argv[1], usecols=["health_state"])
print(len(md), f'{(md["health_state"] == "imminent").mean():.4f}')
PY
}

# --------------------------------------------------------------- verify
do_verify () {
  ref="data/fd001_bal/metadata.csv"
  if [ ! -f "$ref" ]; then
    echo "ERROR: $ref not found. Point this script at the directory that holds"
    echo "       your primary staged cohorts, or stage them first." >&2
    exit 1
  fi
  scratch="data/supp/_verify/fd001_bal"
  rm -rf "$scratch"; mkdir -p "$scratch"
  echo "=== re-staging FD001 at the primary settings (cut 12, window 30) ==="
  stage_one FD001 12 30 "$scratch" >/dev/null

  read -r n_ref p_ref  <<< "$(summarise "$ref")"
  read -r n_new p_new  <<< "$(summarise "$scratch/metadata.csv")"

  printf '\n%-34s %10s %12s\n' "" "windows" "prevalence"
  printf '%-34s %10s %12s\n' "your existing data/fd001_bal" "$n_ref" "$p_ref"
  printf '%-34s %10s %12s\n' "this script's recipe"         "$n_new" "$p_new"

  if [ "$n_ref" = "$n_new" ] && [ "$p_ref" = "$p_new" ]; then
    echo
    echo "MATCH. The recipe reproduces your primary staging; the sweep will be"
    echo "comparable with Table 6. Continue with: $0 stage"
    rm -rf data/supp/_verify
  else
    echo
    echo "MISMATCH. Do NOT run the sweep yet." >&2
    echo "Something in the recipe differs from whatever produced your primary" >&2
    echo "cohorts -- most likely --max-engines, --seed, --test-frac, --stride," >&2
    echo "or the sensor list. Check the top of this script against the command" >&2
    echo "you originally used, or send both rows above to be diagnosed." >&2
    exit 1
  fi
}

# ---------------------------------------------------------------- stage
do_stage () {
  for name in "${SETTING_NAMES[@]}"; do
    read -r CUT WIN <<< "$(setting_params "$name")"
    mkdir -p "data/supp/$name"
    for sub in "${COHORTS[@]}"; do
      tag="$(tag_of "$sub")"
      out="data/supp/$name/$tag"

      if [ "$name" = "base" ]; then
        # The primary setting: reuse the existing staging rather than
        # recompute it, so the shared row of both tables is identical to
        # the main text by construction.
        if [ -e "$out" ]; then echo "skip (linked): $out"; continue; fi
        if [ ! -f "data/$tag/metadata.csv" ]; then
          echo "ERROR: data/$tag not found; cannot link the primary setting." >&2
          exit 1
        fi
        ln -s "../../$tag" "$out"
        echo "linked $out -> data/$tag"
        continue
      fi

      if [ -f "$out/metadata.csv" ]; then echo "skip (exists): $out"; continue; fi
      echo "=== stage $name  $sub  cut=$CUT window=$WIN ==="
      stage_one "$sub" "$CUT" "$WIN" "$out"
    done
  done
  echo
  echo "staged settings and their rare-state prevalence:"
  for name in "${SETTING_NAMES[@]}"; do
    f="data/supp/$name/fd001_bal/metadata.csv"
    [ -f "$f" ] || continue
    read -r n p <<< "$(summarise "$f")"
    printf '  %-6s FD001: %7s windows, prevalence %s\n' "$name" "$n" "$p"
  done
}

# ---------------------------------------------------------------- sweep
do_sweep () {
  for name in "${SETTING_NAMES[@]}"; do
    for sub in "${COHORTS[@]}"; do
      tag="$(tag_of "$sub")"
      res="outputs/supp/$name/holdout_sweep_${tag}.csv"
      if [ -f "$res" ]; then echo "skip (exists): $res"; continue; fi
      echo "=== sweep $name  $tag  $(date) ==="
      mkdir -p "outputs/supp/$name"
      OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} python -u scripts/02_holdout_sweep.py \
        --prepped "data/supp/$name/$tag" \
        --capacities 400 \
        --seeds 10 --split-seeds 5 --n-init 1 \
        --out "outputs/supp/$name"
    done
  done
}

# --------------------------------------------------------------- tables
do_tables () {
  missing=0
  for name in "${SETTING_NAMES[@]}"; do
    for sub in "${COHORTS[@]}"; do
      f="outputs/supp/$name/holdout_sweep_$(tag_of "$sub").csv"
      [ -f "$f" ] || { echo "missing: $f"; missing=$((missing+1)); }
    done
  done
  if [ "$missing" -gt 0 ]; then
    echo
    echo "$missing of 24 sweeps are missing. Finish them first, or the tables" >&2
    echo "will be written with gaps." >&2
  fi
  python -u scripts/make_supp_tables.py \
    --results outputs/supp --staged data/supp --capacity 400 --out paper
}

case "${1:-}" in
  verify) do_verify ;;
  stage)  do_stage  ;;
  sweep)  do_sweep  ;;
  tables) do_tables ;;
  *) echo "usage: $0 {verify|stage|sweep|tables}" >&2; exit 2 ;;
esac
echo "=== $1 complete $(date) ==="
