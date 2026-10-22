#!/usr/bin/env python3
"""
02_holdout_sweep.py
===================
Revision of the capacity sweep addressing reviewer points 2, 3, 4 and 5.

What changes relative to 01_capacity_sweep.py
---------------------------------------------
1. HELD-OUT EVALUATION (point 2). Models are fitted on training engines only,
   frozen, and then held-out windows are assigned to the nearest centroid or
   best-matching unit. Cluster ownership -- which class each cluster is taken
   to represent -- is determined from TRAINING labels and then applied
   unchanged to held-out windows. Held-out labels are therefore used once, to
   score, and never to define the partition or its interpretation. Training
   recovery is reported alongside so the two can be compared.

2. MULTIPLE ENGINE SPLITS (point 3). The engine-level split is repeated with
   several split seeds. Variability is then decomposed into sampling
   variability (which engines were drawn) and initialisation variability
   (which seed the fit used), rather than conflating them.

3. n_init CONTROL (point 4). `--n-init` exposes the K-means restart count. The
   original sweep used a single restart, matching the reference study, which
   leaves open whether K-means saturation reflects geometry or weak
   optimisation. Run with 10 or 20 restarts to settle it. Note this is
   expensive: cost scales linearly, so restrict the capacity grid.

4. SEED-LEVEL OUTPUT (point 5). Every fit is written as its own row, so
   bootstrap confidence intervals on the SOM-minus-K-means difference can be
   computed over seeds, over splits, or over both.

Usage
-----
    # points 2, 3, 5: held-out recovery across 5 engine splits
    python 02_holdout_sweep.py --prepped data/fd001 \
        --capacities 25 100 225 400 --seeds 10 --split-seeds 5 \
        --out outputs/metrics_holdout

    # point 4: optimisation control, single split, reduced grid
    python 02_holdout_sweep.py --prepped data/fd001 \
        --capacities 100 225 400 --seeds 10 --split-seeds 1 \
        --n-init 20 --out outputs/metrics_ninit20
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RARE, STATES, dominant_map, fit_som, side_of  # noqa: E402


# --------------------------------------------------------------------------
def engine_split(units: np.ndarray, frac: float, seed: int) -> set:
    """Hold out whole engines. Returns the set of held-out unit ids."""
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return set(u[: max(1, int(round(len(u) * frac)))])


def recovery(y_fit: np.ndarray, lab_fit: np.ndarray,
             y_eval: np.ndarray, lab_eval: np.ndarray,
             klass: str) -> dict:
    """Ownership from the fitting set; capture and purity on the eval set."""
    dom = dominant_map(y_fit, lab_fit)
    owned = dom.index[dom == klass].to_numpy()
    n_class = int((y_eval == klass).sum())
    if len(owned) == 0 or n_class == 0:
        return {"n_owned": len(owned), "capture": 0.0, "purity": 0.0,
                "purity_pooled": 0.0, "effcov": 0.0, "effcov_pooled": 0.0}

    in_owned = np.isin(lab_eval, owned)
    hit = int(((y_eval == klass) & in_owned).sum())
    capture = hit / n_class
    # Unweighted per-cluster purity (primary), over owned clusters that
    # actually received at least one eval window.
    per = [((y_eval == klass) & (lab_eval == c)).sum() / (lab_eval == c).sum()
           for c in owned if (lab_eval == c).any()]
    purity = float(np.mean(per)) if per else 0.0
    purity_pooled = hit / int(in_owned.sum()) if in_owned.any() else 0.0
    return {"n_owned": len(owned), "capture": capture, "purity": purity,
            "purity_pooled": purity_pooled,
            "effcov": capture * purity,
            "effcov_pooled": capture * purity_pooled}


# --------------------------------------------------------------------------
def run(args) -> None:
    X = np.load(os.path.join(args.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(args.prepped, "metadata.csv"))
    y_all = md[args.label_col].to_numpy()
    units = md["unit"].to_numpy()
    tag = os.path.basename(args.prepped.rstrip("/"))
    print(f"{tag}: {X.shape[0]:,} windows x {X.shape[1]:,} features, "
          f"{len(set(units)):,} engines")
    print(f"grid: {args.capacities} x {args.seeds} seeds x "
          f"{args.split_seeds} split(s), n_init={args.n_init}\n")

    rows = []
    for split in range(args.split_seeds):
        held = engine_split(units, args.test_frac, 1000 + split)
        te = np.isin(units, list(held))
        tr = ~te
        # Standardise on the training engines of THIS split, not globally.
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = (X - mu) / sd
        Xtr, Xte = Xs[tr], Xs[te]
        ytr, yte = y_all[tr], y_all[te]
        print(f"split {split}: {tr.sum():,} train / {te.sum():,} held-out "
              f"windows, rare {100*(yte==args.klass).mean():.1f}% held out")

        for cap, method in itertools.product(args.capacities,
                                             ["kmeans", "som"]):
            side = None
            if method == "som":
                try:
                    side = side_of(cap)
                except ValueError:
                    continue
            for seed in range(args.seeds):
                t0 = time.time()
                if method == "kmeans":
                    m = KMeans(n_clusters=cap, n_init=args.n_init,
                               random_state=seed).fit(Xtr)
                    ltr, lte = m.labels_, m.predict(Xte)
                else:
                    m = fit_som(Xtr, side, seed, args.iters)
                    def bmu(A):
                        return np.array([np.ravel_multi_index(m.winner(x),
                                                              (side, side))
                                         for x in A])
                    ltr, lte = bmu(Xtr), bmu(Xte)

                ho = recovery(ytr, ltr, yte, lte, args.klass)
                tri = recovery(ytr, ltr, ytr, ltr, args.klass)
                rows.append({
                    "split": split, "method": method, "capacity": cap,
                    "seed": seed, "n_init": args.n_init,
                    "n_owned": ho["n_owned"],
                    "capture_holdout": ho["capture"],
                    "purity_holdout": ho["purity"],
                    "effcov_holdout": ho["effcov"],
                    "effcov_holdout_pooled": ho["effcov_pooled"],
                    "effcov_train": tri["effcov"],
                    "seconds": round(time.time() - t0, 1),
                })
            last = rows[-1]
            print(f"    {method:<7} cap={cap:<4} "
                  f"held-out {np.mean([r['effcov_holdout'] for r in rows[-args.seeds:]]):.3f}  "
                  f"train {np.mean([r['effcov_train'] for r in rows[-args.seeds:]]):.3f}  "
                  f"({last['seconds']:.1f}s/fit)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, f"holdout_sweep_{tag}.csv"), index=False)

    # ---- summary with bootstrap CI on the SOM - K-means difference --------
    print("\nheld-out effective coverage, mean [95% bootstrap CI on SOM - KM]")
    rng = np.random.default_rng(0)
    out = []
    for cap in sorted(df.capacity.unique()):
        d = df[df.capacity == cap]
        s = d[d.method == "som"]["effcov_holdout"].to_numpy()
        k = d[d.method == "kmeans"]["effcov_holdout"].to_numpy()
        if len(s) == 0:
            continue
        boot = [rng.choice(s, len(s)).mean() - rng.choice(k, len(k)).mean()
                for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"  cap={cap:<4} KM {k.mean():.3f}  SOM {s.mean():.3f}  "
              f"diff {s.mean()-k.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}")
        out.append({"capacity": cap, "kmeans": k.mean(), "som": s.mean(),
                    "diff": s.mean() - k.mean(), "ci_lo": lo, "ci_hi": hi,
                    "n_fits_per_method": len(s)})
    pd.DataFrame(out).to_csv(
        os.path.join(args.out, f"holdout_summary_{tag}.csv"), index=False)

    if args.split_seeds > 1:
        print("\nvariance decomposition of held-out effcov (SOM, largest cap):")
        d = df[(df.method == "som") & (df.capacity == max(args.capacities))]
        if len(d):
            print(f"  across splits (mean per split): "
                  f"sd = {d.groupby('split')['effcov_holdout'].mean().std():.4f}")
            print(f"  across seeds within split:      "
                  f"sd = {d.groupby('split')['effcov_holdout'].std().mean():.4f}")
    print(f"\nwrote {args.out}/holdout_sweep_{tag}.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics_holdout")
    p.add_argument("--capacities", type=int, nargs="+",
                   default=[25, 100, 225, 400])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--split-seeds", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--label-col", default="health_state")
    p.add_argument("--klass", default=RARE)
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
