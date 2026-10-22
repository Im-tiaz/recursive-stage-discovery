#!/usr/bin/env python3
"""
sigma_nested_selection.py
=========================
Nested, label-free selection of the neighbourhood width sigma.

Two objections motivate this script. First, the ablation reported elsewhere
identifies the best sigma on the same held-out engines used to score it, which
is selection on the evaluation set. Second, the operational setting assumed
throughout this study is one in which failure labels are unavailable, so a
criterion that selects sigma by rare-state effective coverage is not a
procedure a practitioner could run.

Both are addressed here. Outer engines are held out and never touched during
selection. Within the outer training engines, an inner split selects sigma by a
criterion computed WITHOUT any health-state label. The selected sigma is then
frozen, refitted on all outer training engines, and scored once on the outer
held-out engines.

    outer training engines
        -> inner training / inner validation engines
        -> sigma chosen on inner validation by a label-free criterion
    -> sigma frozen, refit on all outer training engines
    -> scored on outer held-out engines

SELECTION CRITERIA
------------------
quantisation error (QE)
    mean distance from a sample to its best-matching unit. Label-free, but
    monotone decreasing in prototype freedom, so it favours the smallest sigma
    and cannot identify an interior optimum. Reported for reference.

topographic error (TE)
    fraction of samples whose best and second-best matching units are not
    adjacent on the lattice. Label-free. It rises when sigma is too small,
    because the map folds and neighbouring nodes stop holding similar weights,
    and also when sigma is too large, because the map underfits the data. It
    therefore admits an interior minimum and is used as the primary criterion.

combined
    minimise QE subject to TE below the median TE across the sigma grid.

An oracle selection by rare-state effective coverage on the inner validation
fold is also reported. It uses labels and is NOT operationally available; it is
included only to bound how much the label-free criteria give up.

    python sigma_nested_selection.py --prepped data/fd003_bal
    python sigma_nested_selection.py --prepped data/fd004_bal --outer-splits 5
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

SIGMA_FRACTIONS = [0.25, 0.125, 0.0625, 0.03125, 0.015625]


def split_units(units, frac, seed):
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return set(u[: max(1, int(round(len(u) * frac)))])


def fit_som(X, side, sigma, seed, iters):
    from minisom import MiniSom
    som = MiniSom(side, side, X.shape[1], sigma=sigma, learning_rate=0.5,
                  random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, iters, verbose=False)
    return som


def _dists(W, X, block=512):
    """Squared distances from every sample to every prototype, in blocks."""
    wn = (W ** 2).sum(1)
    for i in range(0, len(X), block):
        chunk = X[i:i + block]
        yield i, (chunk ** 2).sum(1)[:, None] - 2 * chunk @ W.T + wn


def bmu_pair(som, X, side):
    """Best and second-best matching unit index for each sample."""
    W = som.get_weights().reshape(-1, X.shape[1]).astype(np.float64)
    best = np.empty(len(X), dtype=int)
    second = np.empty(len(X), dtype=int)
    for i, d in _dists(W, X):
        o = np.argpartition(d, 1, axis=1)[:, :2]
        take = np.take_along_axis(d, o, axis=1)
        flip = take[:, 0] > take[:, 1]
        o[flip] = o[flip][:, ::-1]
        best[i:i + len(o)], second[i:i + len(o)] = o[:, 0], o[:, 1]
    return best, second


def bmu_only(W, X):
    """Nearest-prototype index for each sample."""
    out = np.empty(len(X), dtype=int)
    for i, d in _dists(W, X):
        out[i:i + len(d)] = d.argmin(axis=1)
    return out


def topographic_error(best, second, side):
    """Share of samples whose two closest prototypes are non-adjacent."""
    b = np.stack(np.unravel_index(best, (side, side)), axis=1)
    s = np.stack(np.unravel_index(second, (side, side)), axis=1)
    cheb = np.abs(b - s).max(axis=1)
    return float((cheb > 1).mean())


def quantisation_error(som, X):
    W = som.get_weights().reshape(-1, X.shape[1]).astype(np.float64)
    tot = 0.0
    for _, d in _dists(W, X):
        tot += np.sqrt(np.maximum(d.min(axis=1), 0)).sum()
    return float(tot / len(X))


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
    sigmas = [round(side * f, 4) for f in SIGMA_FRACTIONS]
    print(f"{tag}: {side}x{side} lattice, sigma grid {sigmas}")
    print(f"{a.outer_splits} outer splits x {a.seeds} seeds, "
          f"inner validation fraction {a.inner_frac}\n")

    rows, sel, crit_rows = [], [], []
    for outer in range(a.outer_splits):
        held = split_units(units, a.test_frac, 1000 + outer)
        te = np.isin(units, list(held)); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = (X - mu) / sd
        Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y_all[tr], y_all[te]
        utr = units[tr]

        # ---- inner split, entirely within outer training engines ----
        inner_held = split_units(utr, a.inner_frac, 5000 + outer)
        iv = np.isin(utr, list(inner_held)); it = ~iv
        Xit, Xiv, yit, yiv = Xtr[it], Xtr[iv], ytr[it], ytr[iv]

        crit = {c: [] for c in ("qe", "te", "oracle")}
        for sg in sigmas:
            qs, ts, os_ = [], [], []
            for seed in range(a.seeds):
                som = fit_som(Xit, side, sg, seed, a.iters)
                b, s2 = bmu_pair(som, Xiv, side)
                ts.append(topographic_error(b, s2, side))
                qs.append(quantisation_error(som, Xiv))
                bt, _ = bmu_pair(som, Xit, side)
                os_.append(effcov(yit, bt, yiv, b, a.klass))
            crit["qe"].append(np.mean(qs))
            crit["te"].append(np.mean(ts))
            crit["oracle"].append(np.mean(os_))
            crit_rows.append({"cohort": tag, "outer": outer, "sigma": sg,
                              "sigma_frac": sg / side,
                              "qe": float(np.mean(qs)), "qe_sd": float(np.std(qs)),
                              "te": float(np.mean(ts)), "te_sd": float(np.std(ts)),
                              "oracle_effcov": float(np.mean(os_))})
            print(f"  outer {outer} sigma={sg:<7} inner: "
                  f"QE {np.mean(qs):.3f}  TE {np.mean(ts):.4f}  "
                  f"oracle effcov {np.mean(os_):.3f}", flush=True)

        te_arr, qe_arr = np.array(crit["te"]), np.array(crit["qe"])
        pick = {
            "topographic": sigmas[int(np.argmin(te_arr))],
            "quantisation": sigmas[int(np.argmin(qe_arr))],
            "combined": sigmas[int(np.argmin(np.where(
                te_arr <= np.median(te_arr), qe_arr, np.inf)))],
            "oracle": sigmas[int(np.argmax(crit["oracle"]))],
            "fixed_d4": sigmas[0],
        }
        print(f"  outer {outer} selected: "
              + ", ".join(f"{k}={v}" for k, v in pick.items()))

        # ---- refit at the selected sigma, score once on outer held-out ----
        for crit_name, sg in pick.items():
            for seed in range(a.seeds):
                som = fit_som(Xtr, side, sg, seed, a.iters)
                W = som.get_weights().reshape(-1, Xtr.shape[1]).astype(np.float64)
                ltr, lte = bmu_only(W, Xtr), bmu_only(W, Xte)
                rows.append({"cohort": tag, "outer": outer, "criterion": crit_name,
                             "sigma": sg, "seed": seed, "method": "som",
                             "effcov": effcov(ytr, ltr, yte, lte, a.klass)})
        for seed in range(a.seeds):
            km = KMeans(n_clusters=a.capacity, n_init=1, random_state=seed).fit(Xtr)
            rows.append({"cohort": tag, "outer": outer, "criterion": "kmeans",
                         "sigma": np.nan, "seed": seed, "method": "kmeans",
                         "effcov": effcov(ytr, km.labels_, yte, km.predict(Xte),
                                          a.klass)})
        sel.append({"outer": outer, **pick})
        print()

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"sigma_nested_{tag}.csv"), index=False)
    pd.DataFrame(sel).to_csv(os.path.join(a.out, f"sigma_nested_{tag}_selected.csv"),
                             index=False)
    cr = pd.DataFrame(crit_rows)
    cr.to_csv(os.path.join(a.out, f"sigma_criteria_{tag}.csv"), index=False)

    print("\ninner-validation criteria by neighbourhood width")
    print(f"{'sigma':>8}{'QE':>9}{'TE':>9}{'oracle':>9}")
    for sg, g in cr.groupby("sigma"):
        print(f"{sg:>8.3f}{g.qe.mean():>9.3f}{g.te.mean():>9.4f}"
              f"{g.oracle_effcov.mean():>9.3f}")
    print("TE is the share of samples whose two closest prototypes are not")
    print("adjacent on the lattice: lower means better topology preservation.")

    km_mean = df[df.criterion == "kmeans"].effcov.mean()
    rng = np.random.default_rng(0)
    print("outer held-out effective coverage, SOM minus K-means")
    print(f"{'criterion':<14}{'sigma (mode)':>13}{'effcov':>9}{'diff':>9}"
          f"{'95% CI':>22}")
    for c in ("fixed_d4", "topographic", "quantisation", "combined", "oracle"):
        g = df[df.criterion == c]
        if not len(g):
            continue
        k = df[df.criterion == "kmeans"]
        boot = [rng.choice(g.effcov.values, len(g)).mean()
                - rng.choice(k.effcov.values, len(k)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"{c:<14}{g.sigma.mode().iloc[0]:>13.3f}{g.effcov.mean():>9.3f}"
              f"{g.effcov.mean()-km_mean:>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22} {star}")
    print(f"{'kmeans':<14}{'--':>13}{km_mean:>9.3f}")
    print("\n'oracle' selects sigma by rare-state coverage on the inner")
    print("validation fold. It uses labels and is not operationally available.")
    print("It bounds what the label-free criteria give up.")
    print(f"\nwrote {a.out}/sigma_nested_{tag}.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--outer-splits", type=int, default=3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--inner-frac", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--klass", default=RARE)
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
