#!/usr/bin/env python3
"""
capacity_utilization.py
=======================
Verifies that a capacity-matched comparison really is matched.

If a large share of SOM nodes are dead (never the best-matching unit for any
sample), its effective capacity is smaller than its nominal capacity and the
comparison against K-means at the same budget is unfair. This reports live
node counts for both methods, plus mean distance from each sample to its
assigned prototype, broken down by class.

The per-class breakdown matters: within-cluster distance is K-means' own
objective, so K-means is expected to win on it. What is informative is whether
the SOM's penalty falls disproportionately on the rare class.

    python capacity_utilization.py --prepped data/fd001
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from _common import (STATES, add_common_args, bmu_flat, fit_som, load_prepped,
                     save, side_of)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = p.parse_args()
    side = side_of(args.capacity)
    X, md = load_prepped(args.prepped)
    y = md["health_state"].to_numpy()
    tag = os.path.basename(args.prepped.rstrip("/"))
    print(f"{tag}: {X.shape[0]:,} windows, capacity {args.capacity}")

    rows = []
    for seed in range(args.seeds):
        som = fit_som(X, side, seed, args.iters)
        lab_som = bmu_flat(som, X, side)
        w = som.get_weights().reshape(-1, X.shape[1])
        d_som = np.linalg.norm(X - w[lab_som], axis=1)

        km = KMeans(n_clusters=args.capacity, n_init=1, random_state=seed).fit(X)
        d_km = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)

        live_som = len(np.unique(lab_som))
        live_km = len(np.unique(km.labels_))
        print(f"  seed {seed}: SOM {live_som}/{args.capacity} live, "
              f"K-means {live_km}/{args.capacity} live")

        for klass in STATES + ["ALL"]:
            m = np.ones(len(y), bool) if klass == "ALL" else (y == klass)
            if not m.any():
                continue
            rows.append({
                "seed": seed, "class": klass, "n": int(m.sum()),
                "live_som": live_som, "live_kmeans": live_km,
                "dist_som": float(d_som[m].mean()),
                "dist_kmeans": float(d_km[m].mean()),
                "penalty": float(d_som[m].mean() / d_km[m].mean()),
            })

    df = pd.DataFrame(rows)
    agg = (df.groupby("class", as_index=False)
             .agg(n=("n", "first"),
                  live_som=("live_som", "mean"),
                  live_kmeans=("live_kmeans", "mean"),
                  dist_som=("dist_som", "mean"),
                  dist_kmeans=("dist_kmeans", "mean"),
                  penalty=("penalty", "mean"),
                  penalty_sd=("penalty", "std")))
    order = {c: i for i, c in enumerate(STATES + ["ALL"])}
    agg = agg.sort_values("class", key=lambda s: s.map(order))
    print(f"\nmean over {args.seeds} seeds:")
    print(agg.round(3).to_string(index=False))
    save(df, args.out, "capacity_utilization", tag)


if __name__ == "__main__":
    main()
