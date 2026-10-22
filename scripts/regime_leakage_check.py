#!/usr/bin/env python3
"""
regime_leakage_check.py
=======================
Quantifies the preprocessing leakage in the six-regime cohorts.

In FD002 and FD004 the operating-setting triple is clustered into six regimes
and each sensor is standardised within its regime. That transform is fitted
during staging, on the whole cohort, before any engine-level split. Held-out
engines therefore contribute to the regime centres and to the per-regime sensor
means and standard deviations.

Refitting the transform inside every fold would require restructuring the
pipeline so that staging emits raw windows and normalisation moves into the
sweep. Before paying that cost, this script measures how much the leak actually
changes the features. For each engine-level split it fits the transform twice:

    full   regime k-means and per-regime moments from all engines
           (what the analysed cohorts used)
    fold   regime k-means and per-regime moments from training engines only,
           then applied to held-out engines

and compares the resulting standardised sensor values on the held-out rows.
Because the values are already in standard-deviation units, a mean absolute
difference of 0.01 means the two pipelines place a held-out reading one
hundredth of a standard deviation apart.

Regime assignment agreement is reported separately, matched by the Hungarian
algorithm since cluster labels are arbitrary.

    python regime_leakage_check.py --data-dir data/raw/CMAPSSData --subset FD002
    python regime_leakage_check.py --data-dir data/raw/CMAPSSData --subset FD004
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "stage", os.path.join(HERE, "00_stage_cmapss.py"))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

COMMON = ["sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
          "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
          "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]


def fit_apply(df, train_mask, n_cond, seed=0):
    """Fit regime clustering and per-regime moments on train_mask, apply to all."""
    ops = df[stage.OP_COLS].to_numpy()
    if n_cond > 1:
        km = KMeans(n_clusters=n_cond, n_init=10, random_state=seed)
        km.fit(ops[train_mask])
        regime = km.predict(ops)
    else:
        regime = np.zeros(len(df), dtype=int)

    out = {}
    for col in COMMON:
        v = df[col].to_numpy(dtype=float)
        z = np.zeros_like(v)
        for r in np.unique(regime):
            m_all = regime == r
            m_tr = m_all & train_mask
            ref = v[m_tr] if m_tr.sum() > 1 else v[m_all]
            mu, sd = ref.mean(), ref.std()
            z[m_all] = (v[m_all] - mu) / sd if sd > 1e-8 else 0.0
        out[col] = z
    return regime, pd.DataFrame(out)


def agreement(a, b, k):
    """Best-case share of rows agreeing after optimal label matching."""
    conf = np.zeros((k, k), dtype=int)
    for i, j in zip(a, b):
        conf[i, j] += 1
    r, c = linear_sum_assignment(-conf)
    return conf[r, c].sum() / len(a)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subset", required=True,
                   choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--conditions", type=int, default=6)
    p.add_argument("--max-engines", type=int, default=200)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--out", default="./outputs/metrics")
    a = p.parse_args()

    df = stage.load_subset(a.data_dir, a.subset)
    if a.max_engines:
        df = stage.subsample_engines(df, a.max_engines, 1)
    units = df["unit"].to_numpy()
    uniq = np.array(sorted(set(units)))
    print(f"{a.subset}: {len(df):,} cycles, {len(uniq)} engines, "
          f"{a.conditions} regimes, {len(COMMON)} sensors\n")

    rows = []
    for split in range(a.splits):
        rng = np.random.default_rng(1000 + split)
        u = uniq.copy(); rng.shuffle(u)
        held = set(u[: max(1, int(round(len(u) * a.test_frac)))])
        te = np.isin(units, list(held))
        tr = ~te

        reg_full, z_full = fit_apply(df, np.ones(len(df), bool), a.conditions)
        reg_fold, z_fold = fit_apply(df, tr, a.conditions)

        agree = (agreement(reg_full, reg_fold, a.conditions)
                 if a.conditions > 1 else 1.0)
        d = (z_full[COMMON].to_numpy()[te] - z_fold[COMMON].to_numpy()[te])
        rows.append({"split": split, "regime_agreement": agree,
                     "mean_abs_diff": float(np.abs(d).mean()),
                     "p99_abs_diff": float(np.percentile(np.abs(d), 99)),
                     "max_abs_diff": float(np.abs(d).max()),
                     "n_heldout_rows": int(te.sum())})
        print(f"  split {split}: regime agreement {agree*100:5.2f}%  "
              f"mean |diff| {rows[-1]['mean_abs_diff']:.5f}  "
              f"p99 {rows[-1]['p99_abs_diff']:.5f}  "
              f"max {rows[-1]['max_abs_diff']:.5f}")

    r = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    r.to_csv(os.path.join(a.out, f"regime_leakage_{a.subset}.csv"), index=False)
    print(f"\nacross {a.splits} splits, on held-out rows only:")
    print(f"  regime assignment agreement  {r.regime_agreement.mean()*100:.2f}%")
    print(f"  mean absolute difference     {r.mean_abs_diff.mean():.5f} SD")
    print(f"  99th percentile              {r.p99_abs_diff.mean():.5f} SD")
    print(f"  maximum                      {r.max_abs_diff.max():.5f} SD")
    print("\nValues are in standard-deviation units of the standardised sensor.")
    print("A mean absolute difference far below 0.01 indicates that fitting the")
    print("transform on training engines alone would place held-out readings in")
    print("essentially the same position, and that the leak is numerically")
    print("immaterial. A larger value indicates the pipeline must be refitted")
    print("inside each fold and the affected cohorts re-run.")
    print(f"\nwrote {a.out}/regime_leakage_{a.subset}.csv")


if __name__ == "__main__":
    sys.exit(main())
