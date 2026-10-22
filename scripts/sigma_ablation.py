#!/usr/bin/env python3
"""
sigma_ablation.py
=================
Isolates topology preservation (review point 2).

The K-means/SOM contrast confounds the lattice constraint with the objective
function, the update rule, the learning rate, the initialisation scheme and the
optimisation dynamics. A difference between them cannot be attributed to
topology preservation alone.

This script varies the neighbourhood width sigma WITHIN the self-organising map
and holds everything else fixed: same update rule, same learning rate, same
initialisation, same iteration count, same prototype budget, same data. As
sigma falls toward zero the neighbourhood coupling vanishes and the map
degenerates to online vector quantisation on a lattice that no longer
constrains anything. Sigma is therefore a direct handle on the strength of the
constraint.

The prediction under test: if the cohort difference is caused by topology
preservation, the recovery penalty should scale with sigma on a cohort whose
terminal state is separated, and stay flat on one whose terminal state is not.
A dose-response within each cohort is mechanistic evidence that two cohort
observations cannot supply.

K-means at the same budget is fitted alongside as an external reference point.

    python sigma_ablation.py --prepped data/fd001 --capacity 400
    python sigma_ablation.py --prepped data/fd003 --capacity 400 --split-seeds 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RARE, dominant_map, side_of  # noqa: E402

# Multipliers of the lattice side. 0.25 is the value used throughout the
# study; the smallest approaches no coupling at all.
SIGMA_FRACTIONS = [0.25, 0.125, 0.0625, 0.03125, 0.015625]


def engine_split(units, frac, seed):
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return set(u[: max(1, int(round(len(u) * frac)))])


def recovery(y_fit, lab_fit, y_eval, lab_eval, klass):
    """Ownership from the fitting set, scored on the eval set, pooled purity."""
    dom = dominant_map(y_fit, lab_fit)
    owned = dom.index[dom == klass].to_numpy()
    n_class = int((y_eval == klass).sum())
    if len(owned) == 0 or n_class == 0:
        return 0.0, 0.0, 0.0
    in_owned = np.isin(lab_eval, owned)
    hit = int(((y_eval == klass) & in_owned).sum())
    capture = hit / n_class
    purity = hit / int(in_owned.sum()) if in_owned.any() else 0.0
    return capture, purity, capture * purity


def fit_som_sigma(X, side, sigma, seed, iters):
    from minisom import MiniSom
    som = MiniSom(side, side, X.shape[1], sigma=sigma, learning_rate=0.5,
                  random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, iters, verbose=False)
    return som


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--split-seeds", type=int, default=3)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--klass", default=RARE)
    a = p.parse_args()

    side = side_of(a.capacity)
    X = np.load(os.path.join(a.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(a.prepped, "metadata.csv"))
    y_all = md["health_state"].to_numpy()
    units = md["unit"].to_numpy()
    tag = os.path.basename(a.prepped.rstrip("/"))
    sigmas = [round(side * f, 4) for f in SIGMA_FRACTIONS]
    print(f"{tag}: {side}x{side} lattice, sigma in {sigmas} "
          f"(study value {sigmas[0]}), {a.split_seeds} splits x {a.seeds} seeds\n")

    rows = []
    for split in range(a.split_seeds):
        held = engine_split(units, a.test_frac, 1000 + split)
        te = np.isin(units, list(held)); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = (X - mu) / sd
        Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y_all[tr], y_all[te]

        for sigma in sigmas:
            for seed in range(a.seeds):
                t0 = time.time()
                som = fit_som_sigma(Xtr, side, sigma, seed, a.iters)
                def bmu(A):
                    return np.array([np.ravel_multi_index(som.winner(x),
                                                          (side, side)) for x in A])
                ltr, lte = bmu(Xtr), bmu(Xte)
                cap_, pur_, eff_ = recovery(ytr, ltr, yte, lte, a.klass)
                rows.append({"split": split, "method": "som", "sigma": sigma,
                             "sigma_frac": sigma / side, "seed": seed,
                             "live_nodes": int(len(np.unique(ltr))),
                             "capture": cap_, "purity": pur_, "effcov": eff_,
                             "seconds": round(time.time() - t0, 1)})
            d = [r["effcov"] for r in rows[-a.seeds:]]
            print(f"  split {split} sigma={sigma:<7} effcov {np.mean(d):.3f} "
                  f"(live {rows[-1]['live_nodes']}/{a.capacity})", flush=True)

        for seed in range(a.seeds):
            km = KMeans(n_clusters=a.capacity, n_init=1, random_state=seed).fit(Xtr)
            cap_, pur_, eff_ = recovery(ytr, km.labels_, yte, km.predict(Xte), a.klass)
            rows.append({"split": split, "method": "kmeans", "sigma": np.nan,
                         "sigma_frac": np.nan, "seed": seed,
                         "live_nodes": int(len(np.unique(km.labels_))),
                         "capture": cap_, "purity": pur_, "effcov": eff_,
                         "seconds": 0.0})
        d = [r["effcov"] for r in rows[-a.seeds:]]
        print(f"  split {split} K-means      effcov {np.mean(d):.3f}\n")

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"sigma_ablation_{tag}.csv"), index=False)

    som = df[df.method == "som"]
    print("held-out effective coverage by neighbourhood width")
    print(f"{'sigma':>8}{'frac':>8}{'effcov':>9}{'capture':>9}{'purity':>9}{'live':>7}")
    for sg, g in som.groupby("sigma"):
        print(f"{sg:>8.3f}{g.sigma_frac.iloc[0]:>8.4f}{g.effcov.mean():>9.3f}"
              f"{g.capture.mean():>9.3f}{g.purity.mean():>9.3f}"
              f"{g.live_nodes.mean():>7.0f}")
    km = df[df.method == "kmeans"]
    print(f"{'K-means':>8}{'--':>8}{km.effcov.mean():>9.3f}"
          f"{km.capture.mean():>9.3f}{km.purity.mean():>9.3f}"
          f"{km.live_nodes.mean():>7.0f}")

    lo, hi = som[som.sigma == min(sigmas)], som[som.sigma == max(sigmas)]
    trend = lo.effcov.mean() - hi.effcov.mean()
    rng = np.random.default_rng(0)
    boot = [rng.choice(lo.effcov.values, len(lo)).mean()
            - rng.choice(hi.effcov.values, len(hi)).mean() for _ in range(5000)]
    ci = np.percentile(boot, [2.5, 97.5])
    print(f"\nweakest minus strongest coupling: {trend:+.3f} "
          f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print("A positive, interval-excluding-zero trend means recovery improves as")
    print("the lattice constraint is relaxed: the constraint itself carries the")
    print("cost. A flat trend means the cost is not attributable to topology.")
    print(f"\nwrote {a.out}/sigma_ablation_{tag}.csv")


if __name__ == "__main__":
    sys.exit(main())
