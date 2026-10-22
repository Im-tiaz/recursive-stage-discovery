#!/usr/bin/env python3
"""
sigma_train_selection.py
========================
Selects the initial neighbourhood width by quantisation error computed on
TRAINING engines alone, freezes it, and evaluates rare-state recovery once on
outer held-out engines.

Why this exists
---------------
Section 4.4 observes that quantisation error is minimised at sigma_0 = d/8 in
every cohort, and infers that an unsupervised selection step would have chosen
that width. But the quantisation error in that table is computed on the same
held-out engines used to score recovery. Using an unsupervised criterion does
not make selection on the evaluation set harmless: the width was still chosen
after looking at the test engines.

The protocol here is clean:

    outer training engines
      -> inner validation split, within training engines only
      -> sigma_0 minimising quantisation error on the inner validation fold
      -> freeze, refit on all outer training engines
      -> score rare-state effective coverage once on outer held-out engines

Labels play no part in the selection. The inner fold is drawn from training
engines, so outer held-out engines are untouched until scoring.

Reports how often each width is selected, and the recovery difference against
K-means at the selected width. A fixed sigma_0 = d/4 arm and an oracle arm that
selects by held-out recovery are included for reference; the oracle uses labels
and outer data and is not operationally available.

    python sigma_train_selection.py --prepped data/fd004_bal
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RARE, dominant_map, side_of  # noqa: E402

FRACS = [0.25, 0.125, 0.0625, 0.03125, 0.015625]


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


def bmu_qe(W, X):
    lab = np.empty(len(X), dtype=int)
    qe = 0.0
    for i, d in _blocks(W, X):
        lab[i:i + len(d)] = d.argmin(axis=1)
        qe += np.sqrt(np.maximum(d.min(axis=1), 0)).sum()
    return lab, float(qe / len(X))


def fit_W(X, side, sigma0, seed, iters):
    from minisom import MiniSom
    som = MiniSom(side, side, X.shape[1], sigma=sigma0, learning_rate=0.5,
                  random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, iters, verbose=False)
    return som.get_weights().reshape(-1, X.shape[1]).astype(np.float64)


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
    sig = [round(side * f, 4) for f in FRACS]
    X = np.load(os.path.join(a.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(a.prepped, "metadata.csv"))
    y_all = md["health_state"].to_numpy()
    units = md["unit"].to_numpy()
    tag = os.path.basename(a.prepped.rstrip("/"))
    print(f"{tag}: sigma_0 grid {sig}, {a.split_seeds} outer splits x "
          f"{a.seeds} seeds, inner validation fraction {a.inner_frac}\n")

    rows = []
    for split in range(a.split_seeds):
        held = split_units(units, a.test_frac, 1000 + split)
        te = np.isin(units, list(held)); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = ((X - mu) / sd).astype(np.float64)
        Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y_all[tr], y_all[te]
        utr = units[tr]

        inner = split_units(utr, a.inner_frac, 5000 + split)
        iv = np.isin(utr, list(inner)); it = ~iv
        Xit, Xiv = Xtr[it], Xtr[iv]

        for seed in range(a.seeds):
            # ---- selection: inner-validation QE, no labels, no outer data --
            qes = []
            for s0 in sig:
                W = fit_W(Xit, side, s0, seed, a.iters)
                qes.append(bmu_qe(W, Xiv)[1])
            pick = sig[int(np.argmin(qes))]

            # ---- K-means reference on this split --------------------------
            km = KMeans(n_clusters=a.capacity, n_init=1,
                        random_state=seed).fit(Xtr)
            ec_km = effcov(ytr, km.labels_, yte, km.predict(Xte), a.klass)

            # ---- selected, fixed d/4, and an oracle arm -------------------
            ecs = {}
            for name, s0 in (("selected", pick), ("fixed_d4", sig[0])):
                W = fit_W(Xtr, side, s0, seed, a.iters)
                ltr, _ = bmu_qe(W, Xtr)
                lte, _ = bmu_qe(W, Xte)
                ecs[name] = (s0, effcov(ytr, ltr, yte, lte, a.klass))
            best_o, best_ec = None, -np.inf
            for s0 in sig:
                W = fit_W(Xtr, side, s0, seed, a.iters)
                ltr, _ = bmu_qe(W, Xtr)
                lte, _ = bmu_qe(W, Xte)
                e = effcov(ytr, ltr, yte, lte, a.klass)
                if e > best_ec:
                    best_o, best_ec = s0, e
            ecs["oracle"] = (best_o, best_ec)

            for name, (s0, ec) in ecs.items():
                rows.append({"cohort": tag, "split": split, "seed": seed,
                             "arm": name, "sigma0": s0, "ec_som": ec,
                             "ec_kmeans": ec_km, "ec_diff": ec - ec_km,
                             "inner_qe_min": float(min(qes))})
            print(f"  split {split} seed {seed}: inner QE selects "
                  f"sigma_0={pick} | selected {ecs['selected'][1]:.3f}  "
                  f"d/4 {ecs['fixed_d4'][1]:.3f}  KM {ec_km:.3f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"sigma_train_selection_{tag}.csv")
    df.to_csv(path, index=False)

    sel = df[df.arm == "selected"]
    freq = sel.sigma0.value_counts(normalize=True).sort_index(ascending=False)
    print(f"\nselected sigma_0 frequency ({len(sel)} fits)")
    for s0, f in freq.items():
        print(f"  {s0:>7.3f}  {100 * f:5.1f}%")

    rng = np.random.default_rng(0)
    print(f"\noutcome on outer held-out engines")
    print(f"{'arm':<10}{'SOM':>8}{'KM':>8}{'diff':>9}{'95% CI':>21}")
    for arm in ("fixed_d4", "selected", "oracle"):
        g = df[df.arm == arm]
        per = {s: gg.ec_diff.to_numpy() for s, gg in g.groupby("split")}
        sp = list(per); boot = []
        for _ in range(5000):
            pk = rng.integers(0, len(sp), len(sp))
            boot.append(np.mean([rng.choice(per[sp[c]], len(per[sp[c]])).mean()
                                 for c in pk]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"{arm:<10}{g.ec_som.mean():>8.3f}{g.ec_kmeans.mean():>8.3f}"
              f"{g.ec_diff.mean():>+9.4f}{f'[{lo:+.3f},{hi:+.3f}]':>21} {star}")
    print("\n'oracle' selects by outer held-out recovery. It uses labels and")
    print("outer data, is not operationally available, and bounds what the")
    print("training-only criterion gives up.")
    print(f"\nwrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--split-seeds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--inner-frac", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--klass", default=RARE)
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
