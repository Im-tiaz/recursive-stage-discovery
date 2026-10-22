#!/usr/bin/env python3
"""
sigma_paired.py
===============
Recovery, topology preservation and quantisation fidelity measured on the SAME
held-out engines, for every combination of cohort, split, initialisation and
neighbourhood width.

The earlier analysis took recovery from one experiment and topographic error
from another, on different folds, and compared two separately averaged columns.
That prevents any statement about the relationship between them. Here every fit
yields all three quantities on the same held-out windows, so the
recovery-topology trade-off can be examined directly and plotted as a frontier.

Three quantities per fit, all on held-out engines:

    EC   effective coverage of the near-failure state, size-weighted purity,
         with cluster ownership assigned from training labels
    TE   topographic error, the share of samples whose two closest prototypes
         are non-adjacent on the lattice (Kiviluoto 1996). Eight-neighbour
         adjacency on the rectangular lattice.
    QE   quantisation error, the mean distance from a sample to its closest
         prototype

K-means at the same prototype budget is fitted on each split as the recovery
reference, so the reported difference is paired within split and
initialisation.

NEIGHBOURHOOD SCHEDULE
----------------------
MiniSom decays sigma during training rather than holding it fixed, so the value
passed is the INITIAL width, written sigma_0 here and in the manuscript. The
script records the library version and the decay function in use so the
schedule is reproducible from the output alone.

    python sigma_paired.py --prepped data/fd001_bal
    python sigma_paired.py --prepped data/fd004_bal --split-seeds 5 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RARE, dominant_map, side_of  # noqa: E402

SIGMA_FRACTIONS = [0.25, 0.125, 0.0625, 0.03125, 0.015625]


def schedule_note():
    """Record the library version and decay behaviour of the neighbourhood."""
    import minisom
    ver = getattr(minisom, "__version__", "unknown")
    fn = None
    for name in ("_decay_function", "_sigma_decay_function"):
        f = getattr(minisom.MiniSom, name, None)
        if f is not None:
            fn = getattr(f, "__name__", str(f))
            break
    return {"minisom_version": ver, "sigma_decay": fn or "not introspectable",
            "note": "sigma passed to MiniSom is the INITIAL width; it decays "
                    "over training under the library default."}


def split_units(units, frac, seed):
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return set(u[: max(1, int(round(len(u) * frac)))])


def fit_som(X, side, sigma0, seed, iters):
    from minisom import MiniSom
    som = MiniSom(side, side, X.shape[1], sigma=sigma0, learning_rate=0.5,
                  random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, iters, verbose=False)
    return som


def _blocks(W, X, block=512):
    wn = (W ** 2).sum(1)
    for i in range(0, len(X), block):
        c = X[i:i + block]
        yield i, (c ** 2).sum(1)[:, None] - 2 * c @ W.T + wn


def bmu_te_qe(W, X, side):
    """Nearest prototype, topographic error and quantisation error in one pass."""
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
    te = float((np.abs(b - s).max(axis=1) > 1).mean())
    return best, te, float(qe / len(X))


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
    X = np.load(os.path.join(a.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(a.prepped, "metadata.csv"))
    y_all = md["health_state"].to_numpy()
    units = md["unit"].to_numpy()
    tag = os.path.basename(a.prepped.rstrip("/"))
    sig = [round(side * f, 4) for f in SIGMA_FRACTIONS]
    sched = schedule_note()
    print(f"{tag}: {side}x{side} lattice, initial sigma grid {sig}")
    print(f"  minisom {sched['minisom_version']}, "
          f"sigma decay: {sched['sigma_decay']}")
    print(f"  {a.split_seeds} splits x {a.seeds} seeds, all three metrics on "
          f"the same held-out engines\n")

    rows = []
    for split in range(a.split_seeds):
        held = split_units(units, a.test_frac, 1000 + split)
        te_m = np.isin(units, list(held)); tr_m = ~te_m
        mu, sd = X[tr_m].mean(0), X[tr_m].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = ((X - mu) / sd).astype(np.float64)
        Xtr, Xte = Xs[tr_m], Xs[te_m]
        ytr, yte = y_all[tr_m], y_all[te_m]

        km_ec = []
        for seed in range(a.seeds):
            km = KMeans(n_clusters=a.capacity, n_init=1,
                        random_state=seed).fit(Xtr)
            km_ec.append(effcov(ytr, km.labels_, yte, km.predict(Xte), a.klass))
        km_mean = float(np.mean(km_ec))

        for s0 in sig:
            for seed in range(a.seeds):
                t0 = time.time()
                som = fit_som(Xtr, side, s0, seed, a.iters)
                W = som.get_weights().reshape(-1, Xtr.shape[1]).astype(np.float64)
                ltr, _, _ = bmu_te_qe(W, Xtr, side)
                lte, te_v, qe_v = bmu_te_qe(W, Xte, side)
                ec = effcov(ytr, ltr, yte, lte, a.klass)
                rows.append({"cohort": tag, "split": split, "seed": seed,
                             "sigma0": s0, "sigma0_frac": s0 / side,
                             "ec_som": ec, "ec_kmeans": km_ec[seed],
                             "ec_diff": ec - km_ec[seed],
                             "te": te_v, "qe": qe_v,
                             "live_nodes": int(len(np.unique(ltr))),
                             "seconds": round(time.time() - t0, 1)})
            g = rows[-a.seeds:]
            print(f"  split {split} sigma0={s0:<7} "
                  f"EC {np.mean([r['ec_som'] for r in g]):.3f} "
                  f"(KM {km_mean:.3f})  TE {np.mean([r['te'] for r in g]):.4f}  "
                  f"QE {np.mean([r['qe'] for r in g]):.3f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"sigma_paired_{tag}")
    df.to_csv(stem + ".csv", index=False)
    with open(stem + "_schedule.json", "w") as fh:
        json.dump(sched, fh, indent=2)

    rng = np.random.default_rng(0)
    print(f"\npaired within split and initialisation, {tag}")
    print(f"{'sigma0':>9}{'EC diff':>10}{'95% CI':>22}{'TE':>9}{'QE':>9}")
    for s0, g in df.groupby("sigma0"):
        per = {s: gg.ec_diff.to_numpy() for s, gg in g.groupby("split")}
        sp = list(per)
        boot = []
        for _ in range(5000):
            pick = rng.integers(0, len(sp), len(sp))
            boot.append(np.mean([rng.choice(per[sp[c]], len(per[sp[c]])).mean()
                                 for c in pick]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"{s0:>9.3f}{g.ec_diff.mean():>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{g.te.mean():>9.4f}"
              f"{g.qe.mean():>9.3f} {star}")
    r = df[["ec_diff", "te", "qe"]].corr().loc["ec_diff"]
    print(f"\ncorrelation across all fits: EC diff vs TE {r['te']:+.3f}, "
          f"vs QE {r['qe']:+.3f}")
    print(f"wrote {stem}.csv and {stem}_schedule.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--split-seeds", type=int, default=3)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--klass", default=RARE)
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
