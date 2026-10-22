#!/usr/bin/env python3
"""
contiguity_test.py
==================
Tests the proposed mechanism: topology preservation helps the rare class when
that class occupies a contiguous region of the lattice.

Counts connected components formed by the nodes where the rare class is the
plurality, against a null in which the same number of owned nodes is scattered
at random over the lattice. Fewer components than the null means the class
sits in a coherent patch rather than in isolated fragments.

Note the SOM guarantees that neighbouring nodes hold similar weights, so some
contiguity is structural. The informative comparison is BETWEEN cohorts: a
cohort whose rare class fragments more is one where the degradation path is
not single-valued, which is what the recovery results should then reflect.

    python contiguity_test.py --prepped data/fd001
    python contiguity_test.py --prepped data/fd003
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.ndimage import label

from _common import (RARE, add_common_args, bmu_flat, fit_som, load_prepped,
                     owned_clusters, save, side_of)

N_NULL = 1_000


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--klass", default=RARE)
    args = p.parse_args()
    side = side_of(args.capacity)
    X, md = load_prepped(args.prepped)
    y = md["health_state"].to_numpy()
    tag = os.path.basename(args.prepped.rstrip("/"))
    rng = np.random.default_rng(0)
    print(f"{tag}: class '{args.klass}' "
          f"({(y == args.klass).sum():,} of {len(y):,} windows, "
          f"{100 * (y == args.klass).mean():.1f}%)")

    rows = []
    for seed in range(args.seeds):
        som = fit_som(X, side, seed, args.iters)
        lab = bmu_flat(som, X, side)
        owned = owned_clusters(y, lab, args.klass)

        grid = np.zeros(args.capacity, bool)
        grid[owned] = True
        obs = label(grid.reshape(side, side))[1]
        null = np.array([label(rng.permutation(grid).reshape(side, side))[1]
                         for _ in range(N_NULL)])
        p_val = (np.sum(null <= obs) + 1) / (N_NULL + 1)
        rows.append({"seed": seed, "n_owned_nodes": len(owned),
                     "components": obs, "null_mean": float(null.mean()),
                     "null_sd": float(null.std()),
                     "ratio": obs / max(null.mean(), 1e-9), "p": p_val})
        print(f"  seed {seed}: {len(owned):>3} nodes, {obs} component(s), "
              f"null {null.mean():.1f}, ratio {rows[-1]['ratio']:.3f}, "
              f"p={p_val:.4f}")

    df = pd.DataFrame(rows)
    print(f"\nmedian components {df['components'].median():.1f}, "
          f"median ratio {df['ratio'].median():.3f}, "
          f"{(df['p'] < 0.05).sum()}/{len(df)} seeds with p < 0.05")
    print("Compare the RATIO across cohorts; lower means a more coherent patch.")
    save(df, args.out, "contiguity_test", tag)


if __name__ == "__main__":
    main()
