#!/usr/bin/env python3
"""
02b_holdout_sweep_foldnorm.py
=============================
The held-out sweep, with the operating-regime transform fitted inside every
fold rather than during staging.

Why this exists
---------------
In FD002 and FD004 the operating-setting triple is clustered into six regimes
and each sensor is standardised within its regime. In `00_stage_cmapss.py` that
transform is fitted on the whole cohort, before any engine-level split, so
held-out engines contribute to the regime centres and to the per-regime sensor
moments. `regime_leakage_check.py` measured the consequence: regime assignment
is identical either way, but per-regime moments displace a held-out reading by
0.021 to 0.023 standard deviations on average, with a 99th percentile near 0.12.

That is small. This script measures whether it matters, by rebuilding the
feature matrix from raw cycles with the transform fitted on training engines
only, and rerunning the same sweep. The comparison of interest is whether the
fault-mode contrast moves by more than its confidence interval.

Pipeline per split
------------------
    raw cycles
      -> engine-level split
      -> regime k-means fitted on TRAINING engines, applied to all
      -> per-regime sensor moments from TRAINING rows, applied to all
      -> windowing and feature construction
      -> feature standardisation from TRAINING windows
      -> fit, freeze, score on held-out engines

Everything downstream of the transform matches `02_holdout_sweep.py`, including
ownership from training labels and size-weighted purity as primary.

    python 02b_holdout_sweep_foldnorm.py --data-dir data/raw/CMAPSSData \
        --subset FD002 --conditions 6 --capacities 100 400 \
        --seeds 5 --split-seeds 5 --out outputs/metrics_foldnorm
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "stage", os.path.join(HERE, "00_stage_cmapss.py"))
stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage)

sys.path.insert(0, HERE)
from _common import RARE, dominant_map, fit_som, side_of  # noqa: E402

COMMON = ["sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
          "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
          "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]


def normalise_fold(df, train_mask, n_cond, seed=0):
    """Regime clustering and per-regime moments from training rows only."""
    df = df.copy()
    ops = df[stage.OP_COLS].to_numpy()
    if n_cond > 1:
        km = KMeans(n_clusters=n_cond, n_init=10, random_state=seed)
        km.fit(ops[train_mask])
        df["regime"] = km.predict(ops)
    else:
        df["regime"] = 0
    for col in COMMON:
        v = df[col].to_numpy(dtype=float)
        z = np.zeros_like(v)
        for r in df["regime"].unique():
            m_all = (df["regime"] == r).to_numpy()
            m_tr = m_all & train_mask
            ref = v[m_tr] if m_tr.sum() > 1 else v[m_all]
            mu, sd = ref.mean(), ref.std()
            z[m_all] = (v[m_all] - mu) / sd if sd > 1e-8 else 0.0
        df[col] = z
    return df


def recovery(y_fit, lab_fit, y_ev, lab_ev, klass):
    dom = dominant_map(y_fit, lab_fit)
    owned = dom.index[dom == klass].to_numpy()
    n = int((y_ev == klass).sum())
    if len(owned) == 0 or n == 0:
        return 0.0, 0.0, 0.0
    m = np.isin(lab_ev, owned)
    hit = int(((y_ev == klass) & m).sum())
    cap = hit / n
    pur = hit / int(m.sum()) if m.any() else 0.0
    return cap, pur, cap * pur


def run(a):
    raw = stage.load_subset(a.data_dir, a.subset)
    if a.max_engines:
        raw = stage.subsample_engines(raw, a.max_engines, 1)
    units_all = np.array(sorted(raw["unit"].unique()))
    print(f"{a.subset}: {len(raw):,} cycles, {len(units_all)} engines, "
          f"{a.conditions} regime(s), fold-fitted normalisation")

    rows = []
    for split in range(a.split_seeds):
        rng = np.random.default_rng(1000 + split)
        u = units_all.copy(); rng.shuffle(u)
        held = set(u[: max(1, int(round(len(u) * a.test_frac)))])
        row_tr = (~raw["unit"].isin(held)).to_numpy()

        dfn = normalise_fold(raw, row_tr, a.conditions)
        X, meta = stage.extract_windows(dfn, COMMON, a.window, a.stride,
                                        a.pairwise)
        meta["health_state"] = stage.label_health(meta["rul"], a.bins,
                                                  stage.DEFAULT_LABELS)
        te = meta["unit"].isin(held).to_numpy()
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = (X - mu) / sd
        ytr, yte = meta.loc[tr, "health_state"].to_numpy(), \
                   meta.loc[te, "health_state"].to_numpy()
        Xtr, Xte = Xs[tr], Xs[te]
        print(f"  split {split}: {tr.sum():,} train / {te.sum():,} held-out "
              f"windows, {X.shape[1]} features")

        for cap, method in itertools.product(a.capacities, ["kmeans", "som"]):
            side = side_of(cap) if method == "som" else None
            for seed in range(a.seeds):
                t0 = time.time()
                if method == "kmeans":
                    m = KMeans(n_clusters=cap, n_init=a.n_init,
                               random_state=seed).fit(Xtr)
                    ltr, lte = m.labels_, m.predict(Xte)
                else:
                    som = fit_som(Xtr, side, seed, a.iters)
                    def bmu(A):
                        return np.array([np.ravel_multi_index(som.winner(x),
                                                              (side, side))
                                         for x in A])
                    ltr, lte = bmu(Xtr), bmu(Xte)
                cap_, pur_, eff_ = recovery(ytr, ltr, yte, lte, a.klass)
                rows.append({"split": split, "method": method, "capacity": cap,
                             "seed": seed, "capture_holdout": cap_,
                             "purity_holdout": pur_,
                             "effcov_holdout_pooled": eff_,
                             "seconds": round(time.time() - t0, 1)})
            g = [r["effcov_holdout_pooled"] for r in rows[-a.seeds:]]
            print(f"    {method:<7} cap={cap:<4} held-out {np.mean(g):.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    tag = a.subset.lower() + "_bal"
    df.to_csv(os.path.join(a.out, f"holdout_sweep_{tag}.csv"), index=False)

    print(f"\nheld-out effective coverage, SOM minus K-means")
    rng = np.random.default_rng(0)
    for cap in a.capacities:
        d = df[df.capacity == cap]
        per = {s: (g[g.method == "som"].set_index("seed").effcov_holdout_pooled
                   - g[g.method == "kmeans"].set_index("seed").effcov_holdout_pooled
                   ).to_numpy() for s, g in d.groupby("split")}
        sp = list(per)
        boot = []
        for _ in range(5000):
            chosen = rng.integers(0, len(sp), len(sp))
            boot.append(np.mean([rng.choice(per[sp[c]], len(per[sp[c]])).mean()
                                 for c in chosen]))
        boot = np.asarray(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        obs = float(np.mean([v.mean() for v in per.values()]))
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  cap={cap:<4} diff {obs:+.4f}  [{lo:+.4f}, {hi:+.4f}] {star}")
    print(f"\nwrote {a.out}/holdout_sweep_{tag}.csv")
    print("Compare against outputs/metrics_2x2_s10/ for the staged-normalisation"
          "\nversion of the same cohort.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subset", required=True,
                   choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--conditions", type=int, default=6)
    p.add_argument("--max-engines", type=int, default=200)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--pairwise", action="store_true", default=True)
    p.add_argument("--bins", type=int, nargs=3, default=[12, 50, 125])
    p.add_argument("--capacities", type=int, nargs="+", default=[100, 400])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--split-seeds", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--klass", default=RARE)
    p.add_argument("--out", default="./outputs/metrics_foldnorm")
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
