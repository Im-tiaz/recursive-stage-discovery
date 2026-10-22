#!/usr/bin/env python3
"""
equal_budget_control.py
=======================
A matched-restart comparison between $K$-means and the self-organising map.

The optimisation control reported in the body of the paper grants $K$-means ten
and twenty restarts while the self-organising map receives a single
initialisation. That is not a matched comparison in either direction: at one
restart each the effort is equal but neither method is tuned, and at ten or
twenty restarts $K$-means is deliberately advantaged. A reviewer can reasonably
object that the two algorithms were never given comparable selection
opportunity.

This script gives both methods the same number of restarts and selects among
them by each method's own unsupervised objective:

    K-means   the fit with the lowest within-cluster sum of squares (inertia),
              which is the quantity K-means minimises
    SOM       the fit with the lowest quantisation error, the analogous
              unsupervised criterion for a map

Selection uses training data only. Health-state labels play no part in it. The
selected fit of each method is then frozen and scored on held-out engines by
the protocol of 02_holdout_sweep.py, so the reported difference is between two
methods that each had n restarts and each chose its own best.

Topographic error of the selected map is recorded alongside, since a map
selected by quantisation error alone may preserve topology poorly.

    python equal_budget_control.py --prepped data/fd004_bal --restarts 1 5 10
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


def split_units(units, frac, seed):
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return set(u[: max(1, int(round(len(u) * frac)))])


def _blocks(W, X, block=512):
    wn = (W ** 2).sum(1)
    for i in range(0, len(X), block):
        c = X[i:i + block]
        yield i, (c ** 2).sum(1)[:, None] - 2 * c @ W.T + wn


def bmu_te_qe(W, X, side):
    best = np.empty(len(X), dtype=int)
    second = np.empty(len(X), dtype=int)
    qe = 0.0
    for i, d in _blocks(W, X):
        o = np.argpartition(d, 1, axis=1)[:, :2]
        take = np.take_along_axis(d, o, axis=1)
        flip = take[:, 0] > take[:, 1]
        o[flip] = o[flip][:, ::-1]
        best[i:i + len(o)], second[i:i + len(o)] = o[:, 0], o[:, 1]
        qe += np.sqrt(np.maximum(d.min(axis=1), 0)).sum()
    b = np.stack(np.unravel_index(best, (side, side)), axis=1)
    s = np.stack(np.unravel_index(second, (side, side)), axis=1)
    return best, float((np.abs(b - s).max(axis=1) > 1).mean()), float(qe / len(X))


def fit_som(X, side, sigma0, seed, iters):
    from minisom import MiniSom
    som = MiniSom(side, side, X.shape[1], sigma=sigma0, learning_rate=0.5,
                  random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, iters, verbose=False)
    return som


def effcov(y_fit, lab_fit, y_ev, lab_ev, klass):
    dom = dominant_map(y_fit, lab_fit)
    owned = dom.index[dom == klass].to_numpy()
    n = int((y_ev == klass).sum())
    if len(owned) == 0 or n == 0:
        return 0.0
    m = np.isin(lab_ev, owned)
    hit = int(((y_ev == klass) & m).sum())
    return (hit / n) * (hit / int(m.sum()) if m.any() else 0.0)


def run(a):
    side = side_of(a.capacity)
    sigma0 = side / 4
    X = np.load(os.path.join(a.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(a.prepped, "metadata.csv"))
    y_all = md["health_state"].to_numpy()
    units = md["unit"].to_numpy()
    tag = os.path.basename(a.prepped.rstrip("/"))
    print(f"{tag}: capacity {a.capacity}, sigma_0 = {sigma0}, "
          f"restarts {a.restarts}, {a.split_seeds} splits x {a.outer} outer seeds")
    print("  K-means selected by inertia, SOM selected by quantisation error, "
          "both on training data only\n")

    rows = []
    for split in range(a.split_seeds):
        held = split_units(units, a.test_frac, 1000 + split)
        te = np.isin(units, list(held)); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = ((X - mu) / sd).astype(np.float64)
        Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y_all[tr], y_all[te]

        for n_restart in a.restarts:
            for outer in range(a.outer):
                t0 = time.time()
                base = 1000 * outer + 7 * n_restart

                # --- K-means: n restarts, keep the lowest inertia ----------
                best_km, best_in = None, np.inf
                for r in range(n_restart):
                    m = KMeans(n_clusters=a.capacity, n_init=1,
                               random_state=base + r).fit(Xtr)
                    if m.inertia_ < best_in:
                        best_km, best_in = m, m.inertia_
                ec_km = effcov(ytr, best_km.labels_, yte,
                               best_km.predict(Xte), a.klass)

                # --- SOM: n restarts, keep the lowest quantisation error ---
                best_W, best_qe = None, np.inf
                for r in range(n_restart):
                    som = fit_som(Xtr, side, sigma0, base + r, a.iters)
                    W = som.get_weights().reshape(-1, Xtr.shape[1]).astype(np.float64)
                    _, _, qe = bmu_te_qe(W, Xtr, side)
                    if qe < best_qe:
                        best_W, best_qe = W, qe
                ltr, _, _ = bmu_te_qe(best_W, Xtr, side)
                lte, te_v, qe_te = bmu_te_qe(best_W, Xte, side)
                ec_som = effcov(ytr, ltr, yte, lte, a.klass)

                rows.append({"cohort": tag, "split": split, "outer": outer,
                             "restarts": n_restart, "ec_kmeans": ec_km,
                             "ec_som": ec_som, "ec_diff": ec_som - ec_km,
                             "km_inertia": float(best_in),
                             "som_qe_train": best_qe, "som_qe_holdout": qe_te,
                             "som_te_holdout": te_v,
                             "seconds": round(time.time() - t0, 1)})
            g = [r for r in rows if r["split"] == split
                 and r["restarts"] == n_restart]
            print(f"  split {split} restarts={n_restart:<3} "
                  f"KM {np.mean([r['ec_kmeans'] for r in g]):.3f}  "
                  f"SOM {np.mean([r['ec_som'] for r in g]):.3f}  "
                  f"diff {np.mean([r['ec_diff'] for r in g]):+.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"equal_budget_{tag}.csv")
    df.to_csv(path, index=False)

    rng = np.random.default_rng(0)
    print(f"\nmatched-restart comparison, {tag}")
    print(f"{'restarts':>9}{'KM':>8}{'SOM':>8}{'diff':>9}{'95% CI':>21}{'TE':>8}")
    for n_restart, g in df.groupby("restarts"):
        per = {s: gg.ec_diff.to_numpy() for s, gg in g.groupby("split")}
        sp = list(per); boot = []
        for _ in range(5000):
            pick = rng.integers(0, len(sp), len(sp))
            boot.append(np.mean([rng.choice(per[sp[c]], len(per[sp[c]])).mean()
                                 for c in pick]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"{n_restart:>9}{g.ec_kmeans.mean():>8.3f}{g.ec_som.mean():>8.3f}"
              f"{g.ec_diff.mean():>+9.4f}{f'[{lo:+.3f},{hi:+.3f}]':>21}"
              f"{g.som_te_holdout.mean():>8.4f} {star}")
    print(f"\nwrote {path}")
    print("\nIf the difference is stable across restart counts, neither method's")
    print("selection opportunity explains it. If it narrows as both are given")
    print("more restarts, the earlier asymmetric control overstated the gap.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--restarts", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--split-seeds", type=int, default=5)
    p.add_argument("--outer", type=int, default=3,
                   help="independent repetitions of the whole selection")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--klass", default=RARE)
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
