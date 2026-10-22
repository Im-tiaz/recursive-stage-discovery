#!/usr/bin/env python3
"""
label_permutation_control.py
============================
Negative control for the recovery metric, in the mould of Paper 3's
`negative_controls.py`.

Effective coverage is bounded below by chance rather than by zero: with many
clusters, some will show the rare class as plurality purely by accident. This
shuffles the health-state labels while holding the partition fixed, giving the
null distribution of effective coverage, and reports the observed value
against it.

It also checks whether the SOM's advantage over K-means persists under shuffled
labels. If it does, that advantage is an artefact of cluster-size distribution
rather than of real label geometry.

    python label_permutation_control.py --prepped data/fd001
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from _common import (RARE, add_common_args, bmu_flat, effcov, fit_som,
                     load_prepped, save, side_of)

N_NULL = 200


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--klass", default=RARE)
    args = p.parse_args()
    side = side_of(args.capacity)
    X, md = load_prepped(args.prepped)
    y = md["health_state"].to_numpy()
    tag = os.path.basename(args.prepped.rstrip("/"))
    rng = np.random.default_rng(0)
    print(f"{tag}: capacity {args.capacity}, class '{args.klass}', "
          f"{N_NULL} label permutations per partition")

    rows = []
    for seed in range(args.seeds):
        som = fit_som(X, side, seed, args.iters)
        parts = {
            "som": bmu_flat(som, X, side),
            "kmeans": KMeans(n_clusters=args.capacity, n_init=1,
                             random_state=seed).fit_predict(X),
        }
        for method, lab in parts.items():
            obs = effcov(y, lab, args.klass)
            null = np.array([effcov(rng.permutation(y), lab, args.klass)
                             for _ in range(N_NULL)])
            p_val = (np.sum(null >= obs) + 1) / (N_NULL + 1)
            rows.append({"seed": seed, "method": method, "observed": obs,
                         "null_mean": float(null.mean()),
                         "null_sd": float(null.std()),
                         "null_max": float(null.max()),
                         "z": (obs - null.mean()) / max(null.std(), 1e-9),
                         "p": p_val})
            print(f"  seed {seed} {method:<7}: observed {obs:.3f}, "
                  f"null {null.mean():.3f}+/-{null.std():.3f} "
                  f"(max {null.max():.3f}), z={rows[-1]['z']:.1f}, "
                  f"p={p_val:.4f}")

    df = pd.DataFrame(rows)
    agg = df.groupby("method", as_index=False).agg(
        observed=("observed", "mean"), null_mean=("null_mean", "mean"),
        z=("z", "mean"))
    print("\nmean over seeds:")
    print(agg.round(3).to_string(index=False))
    gap_real = agg.set_index("method")["observed"].get("som", 0) - \
        agg.set_index("method")["observed"].get("kmeans", 0)
    gap_null = agg.set_index("method")["null_mean"].get("som", 0) - \
        agg.set_index("method")["null_mean"].get("kmeans", 0)
    print(f"\nSOM minus K-means: {gap_real:+.3f} observed, "
          f"{gap_null:+.3f} under shuffled labels")
    save(df, args.out, "label_permutation_control", tag)


if __name__ == "__main__":
    main()
