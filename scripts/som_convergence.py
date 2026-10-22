#!/usr/bin/env python3
"""
som_convergence.py
==================
Rules out the under-training confound.

If large SOM lattices were simply under-converged, they would recover the rare
class poorly for a reason that has nothing to do with topology preservation.
This trains cumulatively and reports quantisation error at increasing
iteration budgets. A flat curve means the training budget used throughout the
study is sufficient and the recovery results are not an artefact of stopping
early.

    python som_convergence.py --prepped data/fd001
"""

import argparse
import os

import numpy as np
import pandas as pd

from _common import add_common_args, fit_som, load_prepped, save, side_of

CHECKPOINTS = [20_000, 30_000, 50_000, 100_000, 200_000]


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = p.parse_args()
    side = side_of(args.capacity)
    X, _ = load_prepped(args.prepped)
    tag = os.path.basename(args.prepped.rstrip("/"))
    print(f"{tag}: {X.shape[0]:,} training windows, {side}x{side} lattice")

    rows = []
    for seed in range(args.seeds):
        som = fit_som(X, side, seed, iters=CHECKPOINTS[0])
        total = CHECKPOINTS[0]
        qe = som.quantization_error(X)
        rows.append({"seed": seed, "iterations": total, "qe": qe})
        print(f"  seed {seed}: {total:>8,} -> QE {qe:.4f}")
        for step in CHECKPOINTS[1:]:
            som.train_random(X, step, verbose=False)   # resumes, does not reset
            total += step
            qe = som.quantization_error(X)
            rows.append({"seed": seed, "iterations": total, "qe": qe})
            print(f"           {total:>8,} -> QE {qe:.4f}")

    df = pd.DataFrame(rows)
    piv = df.pivot(index="iterations", columns="seed", values="qe")
    first, last = piv.mean(axis=1).iloc[0], piv.mean(axis=1).iloc[-1]
    drift = (first - last) / first * 100
    print(f"\nmean QE {first:.4f} -> {last:.4f} "
          f"across {piv.index[0]:,}-{piv.index[-1]:,} iterations "
          f"({drift:.2f}% change)")
    print("converged" if abs(drift) < 2 else "NOT converged: raise --iters")
    save(df, args.out, "som_convergence", tag)


if __name__ == "__main__":
    main()
