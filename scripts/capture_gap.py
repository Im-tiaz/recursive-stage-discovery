#!/usr/bin/env python3
"""
capture_gap.py
==============
Answers the obvious objection: is the SOM simply better on this dataset?

Reports capture and effective coverage for EVERY class, not just the rare one,
and the SOM-minus-K-means gap per class. If the SOM leads everywhere, the
rare-class result says nothing about rarity or geometry. If it trails on the
majority classes and leads on the rare one, the two methods are trading
majority coverage for minority coverage at a fixed prototype budget, which is
the claim the paper actually makes.

    python capture_gap.py --prepped data/fd001
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from _common import (STATES, add_common_args, bmu_flat, dominant_map, fit_som,
                     load_prepped, save, side_of)


def per_class(y: np.ndarray, lab: np.ndarray) -> dict:
    dom = dominant_map(y, lab)
    out = {}
    for klass in np.unique(y):
        owned = dom.index[dom == klass].to_numpy()
        n_class = int((y == klass).sum())
        if len(owned) == 0 or n_class == 0:
            out[klass] = (0.0, 0.0, 0)
            continue
        in_owned = np.isin(lab, owned)
        hit = int(((y == klass) & in_owned).sum())
        capture = hit / n_class
        purity = float(np.mean([
            ((y == klass) & (lab == cid)).sum() / (lab == cid).sum()
            for cid in owned]))
        out[klass] = (capture, capture * purity, len(owned))
    return out


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = p.parse_args()
    side = side_of(args.capacity)
    X, md = load_prepped(args.prepped)
    y = md["health_state"].to_numpy()
    tag = os.path.basename(args.prepped.rstrip("/"))
    print(f"{tag}: capacity {args.capacity}, {args.seeds} seeds")

    rows = []
    for seed in range(args.seeds):
        som = fit_som(X, side, seed, args.iters)
        res = {
            "som": per_class(y, bmu_flat(som, X, side)),
            "kmeans": per_class(y, KMeans(n_clusters=args.capacity, n_init=1,
                                          random_state=seed).fit_predict(X)),
        }
        for klass in np.unique(y):
            rows.append({
                "seed": seed, "class": klass, "n": int((y == klass).sum()),
                "prevalence": float((y == klass).mean()),
                "capture_som": res["som"][klass][0],
                "capture_kmeans": res["kmeans"][klass][0],
                "effcov_som": res["som"][klass][1],
                "effcov_kmeans": res["kmeans"][klass][1],
                "nodes_som": res["som"][klass][2],
                "nodes_kmeans": res["kmeans"][klass][2],
            })

    df = pd.DataFrame(rows)
    agg = df.groupby("class", as_index=False).mean(numeric_only=True)
    agg["capture_gap"] = agg["capture_som"] - agg["capture_kmeans"]
    agg["effcov_gap"] = agg["effcov_som"] - agg["effcov_kmeans"]
    order = {c: i for i, c in enumerate(STATES)}
    agg = agg.sort_values("class", key=lambda s: s.map(order))
    cols = ["class", "n", "prevalence", "capture_kmeans", "capture_som",
            "capture_gap", "effcov_kmeans", "effcov_som", "effcov_gap",
            "nodes_kmeans", "nodes_som"]
    print("\nmean over seeds (classes ordered by severity):")
    print(agg[cols].round(3).to_string(index=False))

    wins = (agg["capture_gap"] > 0).sum()
    print(f"\nSOM leads on capture in {wins}/{len(agg)} classes.")
    if wins == len(agg):
        print("  SOM leads everywhere: the rare-class result is NOT about rarity.")
    else:
        print("  Mixed: consistent with a trade at fixed prototype budget.")
    save(df, args.out, "capture_gap", tag)


if __name__ == "__main__":
    main()
