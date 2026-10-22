#!/usr/bin/env python3
"""
capture_gap_holdout.py
======================
Per-class capture and purity, measured on HELD-OUT engines.

Replaces capture_gap.py, which called load_prepped() with its default
train_only=True and therefore fitted and scored on the same windows. Every
number that script produced described training data. This one follows the
protocol of 02_holdout_sweep.py: fit on training engines, freeze, assign
held-out windows, and take cluster ownership from TRAINING labels so that
held-out labels are used once, to score.

Addresses three review points at once:
  * held-out rather than fitting-set measurement
  * capture and purity reported separately, not only their product, since two
    methods can reach the same product with different recall and false-positive
    behaviour
  * runs on any staged cohort, so the class-wise control extends to all four

Both purity conventions are reported. Size-weighted purity is primary for
held-out evaluation, because the unweighted per-cluster mean is unstable when
an owned cluster receives few held-out windows.

    python capture_gap_holdout.py --prepped data/fd001_bal
    python capture_gap_holdout.py --prepped data/fd004_bal --method som
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import STATES, dominant_map, fit_som, side_of  # noqa: E402


def engine_split(units, frac, seed):
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return set(u[: max(1, int(round(len(u) * frac)))])


def per_class(y_fit, lab_fit, y_eval, lab_eval):
    """Ownership from the fitting set; capture and purity on the eval set."""
    dom = dominant_map(y_fit, lab_fit)
    rows = []
    for klass in np.unique(y_fit):
        owned = dom.index[dom == klass].to_numpy()
        n_eval = int((y_eval == klass).sum())
        if len(owned) == 0 or n_eval == 0:
            rows.append({"class": klass, "n_eval": n_eval, "n_owned": len(owned),
                         "capture": 0.0, "purity_pooled": 0.0, "purity": 0.0,
                         "effcov_pooled": 0.0})
            continue
        in_owned = np.isin(lab_eval, owned)
        hit = int(((y_eval == klass) & in_owned).sum())
        capture = hit / n_eval
        pooled = hit / int(in_owned.sum()) if in_owned.any() else 0.0
        per = [((y_eval == klass) & (lab_eval == c)).sum() / (lab_eval == c).sum()
               for c in owned if (lab_eval == c).any()]
        rows.append({"class": klass, "n_eval": n_eval, "n_owned": len(owned),
                     "capture": capture, "purity_pooled": pooled,
                     "purity": float(np.mean(per)) if per else 0.0,
                     "effcov_pooled": capture * pooled})
    return rows


def run(args):
    X = np.load(os.path.join(args.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(args.prepped, "metadata.csv"))
    y_all = md["health_state"].to_numpy()
    units = md["unit"].to_numpy()
    tag = os.path.basename(args.prepped.rstrip("/"))
    print(f"{tag}: {X.shape[0]:,} windows, {len(set(units)):,} engines, "
          f"capacity {args.capacity}, {args.split_seeds} splits x "
          f"{args.seeds} seeds")

    rows = []
    for split in range(args.split_seeds):
        held = engine_split(units, args.test_frac, 1000 + split)
        te = np.isin(units, list(held)); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = (X - mu) / sd
        Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y_all[tr], y_all[te]
        if split == 0:
            print(f"  split 0: {tr.sum():,} train / {te.sum():,} held-out windows")

        for method, seed in itertools.product(args.methods, range(args.seeds)):
            if method == "kmeans":
                m = KMeans(n_clusters=args.capacity, n_init=args.n_init,
                           random_state=seed).fit(Xtr)
                ltr, lte = m.labels_, m.predict(Xte)
            else:
                side = side_of(args.capacity)
                som = fit_som(Xtr, side, seed, args.iters)
                def bmu(A):
                    return np.array([np.ravel_multi_index(som.winner(x),
                                                          (side, side)) for x in A])
                ltr, lte = bmu(Xtr), bmu(Xte)
            for r in per_class(ytr, ltr, yte, lte):
                r.update({"cohort": tag, "split": split, "method": method,
                          "seed": seed})
                rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"capture_gap_holdout_{tag}.csv")
    df.to_csv(path, index=False)

    agg = df.groupby(["class", "method"]).mean(numeric_only=True).reset_index()
    piv = agg.pivot(index="class", columns="method",
                    values=["capture", "purity_pooled", "effcov_pooled"])
    order = {c: i for i, c in enumerate(STATES)}
    piv = piv.sort_index(key=lambda s: s.map(order))
    n = df.groupby("class")["n_eval"].mean()

    print(f"\nheld-out, mean over {args.split_seeds * args.seeds} fits per method")
    print(f"{'state':<11}{'n':>7}{'KM cap':>9}{'SOM cap':>9}{'gap':>8}"
          f"{'KM pur':>9}{'SOM pur':>9}{'gap':>8}")
    for c in [s for s in STATES if s in piv.index]:
        kc, sc = piv.loc[c, ("capture", "kmeans")], piv.loc[c, ("capture", "som")]
        kp, sp = piv.loc[c, ("purity_pooled", "kmeans")], piv.loc[c, ("purity_pooled", "som")]
        print(f"{c:<11}{n[c]:>7.0f}{kc:>9.3f}{sc:>9.3f}{sc-kc:>+8.3f}"
              f"{kp:>9.3f}{sp:>9.3f}{sp-kp:>+8.3f}")
    print(f"\nwrote {path}")
    print("\nCapture is recall of the class by the clusters it dominates.")
    print("Purity is the share of those clusters' members belonging to it.")
    print("Report both: equal products can hide opposite behaviours.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--methods", nargs="+", default=["kmeans", "som"])
    p.add_argument("--method", dest="methods", action="append_const", const=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--split-seeds", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=50_000)
    a = p.parse_args()
    a.methods = [m for m in a.methods if m]
    run(a)


if __name__ == "__main__":
    sys.exit(main())
