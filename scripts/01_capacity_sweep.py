#!/usr/bin/env python3
"""
01_capacity_sweep.py
====================
Capacity-matched comparison of K-means and SOMs on the staged C-MAPSS matrix,
replicating the design of Paper 2 (AI Eng. 2026, aieng1020010) on multi-sensor
time-series data.

This script tests the four predictions Paper 2 made for engineering sensor data
(Sec. 5.5.3) and states, per prediction, whether it held. That framing is the
paper: the method is already published, what is new is the transfer.

Design (matched to Paper 2 so the tables are directly comparable)
-----------------------------------------------------------------
* K-means across cluster counts k, SOMs across lattice sizes, compared at
  MATCHED PROTOTYPE BUDGETS (k == n_nodes). Comparing a 20x20 SOM to k=20
  K-means is the error this design exists to avoid.
* `--seeds 25` independent fits per configuration.
* Stability = mean PAIRWISE NMI and ARI across seed pairs. Not a single run.
* Recovery = capture / purity / effective coverage per health state, so a
  method that hides the rare near-failure class inside a large healthy cluster
  cannot look good.

Metric definitions (RESOLVED — do not change without re-running everything)
--------------------------------------------------------------------------
`effective_coverage()` reports TWO purity conventions on every row:

  purity          per-cluster unweighted mean over owned clusters. This is
                  Paper 2's definition, verified against
                  cancer-som-paper/scripts/09_consensus_kmeans.py L121.
                  `effective_coverage` = capture * purity uses it, so numbers
                  in this study sit on the same axis as Paper 2's.

  purity_pooled   pooled hits / pooled members over owned clusters, i.e.
                  size-weighted. `effcov_pooled` = capture * purity_pooled.

They agree only when owned clusters are equally sized. The unweighted mean lets
a 3-member pure cluster count as much as a 40-member one, which can favour
methods that fragment into many small clusters — so report the headline result
under both and state that it is robust to the convention.

Outputs
-------
    outputs/metrics/capacity_sweep.csv      one row per (method, capacity, seed)
    outputs/metrics/stability_pairwise.csv  one row per (method, capacity)
    outputs/metrics/recovery_by_class.csv   one row per (method, capacity, class)

Usage
-----
    python 01_capacity_sweep.py --self-test
    python 01_capacity_sweep.py --prepped ./data/fd001 --out ./outputs/metrics \
        --capacities 4 8 19 50 100 200 400 --seeds 25
    # one configuration per SLURM array task:
    python 01_capacity_sweep.py --prepped ./data/fd001 --capacity 400 --method kmeans
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# --------------------------------------------------------------------------
# Recovery decomposition
# --------------------------------------------------------------------------
def effective_coverage(labels: np.ndarray, clusters: np.ndarray) -> pd.DataFrame:
    """Per-class capture / purity / effective coverage.

    For class c, let D(c) be the clusters in which c is the plurality label.
        capture(c)  = |c inside D(c)| / |c|
        purity(c)   = |c inside D(c)| / |all members of D(c)|
        effcov(c)   = capture(c) * purity(c)

    A class with no dominated cluster scores zero on all three: it has been
    absorbed into someone else's cluster, which is exactly the failure mode
    that matters for rare critical states.
    """
    out = []
    cl = pd.DataFrame({"y": labels, "c": clusters})
    dominant = cl.groupby("c")["y"].agg(lambda s: s.value_counts().idxmax())

    for klass in np.unique(labels):
        owned = dominant.index[dominant == klass]
        n_class = int((labels == klass).sum())
        if len(owned) == 0 or n_class == 0:
            out.append({"class": klass, "n": n_class, "n_owned_clusters": 0,
                        "capture": 0.0, "purity": 0.0, "purity_pooled": 0.0,
                        "effective_coverage": 0.0, "effcov_pooled": 0.0})
            continue

        in_owned = np.isin(clusters, owned)
        hit = int(((labels == klass) & in_owned).sum())
        capture = hit / n_class

        # PRIMARY: per-cluster unweighted mean, matching Paper 2
        # (cancer-som-paper/scripts/09_consensus_kmeans.py L121).
        purity = float(np.mean([
            ((labels == klass) & (clusters == cid)).sum() / (clusters == cid).sum()
            for cid in owned
        ]))
        # SECONDARY: pooled over all owned clusters. Size-weighted, so it does
        # not let a 3-member pure cluster count as much as a 40-member one.
        # Kept so the headline result can be shown robust to the convention.
        purity_pooled = hit / int(in_owned.sum())

        out.append({"class": klass, "n": n_class, "n_owned_clusters": len(owned),
                    "capture": capture,
                    "purity": purity,
                    "purity_pooled": purity_pooled,
                    "effective_coverage": capture * purity,
                    "effcov_pooled": capture * purity_pooled})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_kmeans(X: np.ndarray, capacity: int, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=capacity, n_init=1, random_state=seed, max_iter=300)
    return km.fit_predict(X)


def fit_som(X: np.ndarray, capacity: int, seed: int, iters: int = 20_000) -> np.ndarray:
    """Square SOM whose node count matches `capacity`; BMU index is the label."""
    try:
        from minisom import MiniSom
    except ImportError:  # pragma: no cover
        raise ImportError(
            "minisom is not installed here. It IS installed in the "
            "`stable-subtypes` env on the cluster — run this there."
        )
    side = int(round(np.sqrt(capacity)))
    if side * side != capacity:
        raise ValueError(
            f"capacity {capacity} is not a perfect square; SOM lattices must be "
            f"square for a matched budget. Use {side * side} or {(side + 1) ** 2}."
        )
    som = MiniSom(side, side, X.shape[1], sigma=max(1.0, side / 4),
                  learning_rate=0.5, random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, iters, verbose=False)
    return np.array([np.ravel_multi_index(som.winner(x), (side, side)) for x in X])


FITTERS = {"kmeans": fit_kmeans, "som": fit_som}


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
def run_configuration(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    capacity: int,
    seeds: int,
    max_pairs: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    fitter = FITTERS[method]
    partitions, per_seed = [], []

    for seed in range(seeds):
        t0 = time.time()
        lab = fitter(X, capacity, seed)
        partitions.append(lab)
        per_seed.append({
            "method": method, "capacity": capacity, "seed": seed,
            "n_nonempty": int(len(np.unique(lab))),
            "nmi_vs_truth": normalized_mutual_info_score(y, lab),
            "ari_vs_truth": adjusted_rand_score(y, lab),
            "seconds": round(time.time() - t0, 2),
        })
        print(f"    {method} cap={capacity} seed={seed} "
              f"({per_seed[-1]['seconds']:.1f}s)", flush=True)

    # Pairwise agreement between independent fits = stability.
    pairs = list(itertools.combinations(range(seeds), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    nmis = [normalized_mutual_info_score(partitions[a], partitions[b]) for a, b in pairs]
    aris = [adjusted_rand_score(partitions[a], partitions[b]) for a, b in pairs]
    stability = {
        "method": method, "capacity": capacity, "n_seeds": seeds,
        "n_pairs": len(pairs),
        "pairwise_nmi": float(np.mean(nmis)), "pairwise_nmi_sd": float(np.std(nmis)),
        "pairwise_ari": float(np.mean(aris)), "pairwise_ari_sd": float(np.std(aris)),
    }

    # Recovery is averaged over seeds so it is not a lucky-run artefact.
    recs = []
    for seed, lab in enumerate(partitions):
        r = effective_coverage(y, lab)
        r["seed"] = seed
        recs.append(r)
    rec = (pd.concat(recs)
             .groupby("class", as_index=False)
             .agg(n=("n", "first"),
                  n_owned_clusters=("n_owned_clusters", "mean"),
                  capture=("capture", "mean"),
                  capture_sd=("capture", "std"),
                  purity=("purity", "mean"),
                  purity_pooled=("purity_pooled", "mean"),
                  effective_coverage=("effective_coverage", "mean"),
                  effective_coverage_sd=("effective_coverage", "std"),
                  effcov_pooled=("effcov_pooled", "mean"),
                  effcov_pooled_sd=("effcov_pooled", "std")))
    rec.insert(0, "capacity", capacity)
    rec.insert(0, "method", method)
    return pd.DataFrame(per_seed), stability, rec


def run(args) -> None:
    X = np.load(os.path.join(args.prepped, "X.npy"))
    meta = pd.read_csv(os.path.join(args.prepped, "metadata.csv"))
    if args.train_only:
        keep = (meta["split"] == "train").to_numpy()
        X, meta = X[keep], meta.loc[keep].reset_index(drop=True)
    y = meta[args.label_col].to_numpy()

    print(f"X: {X.shape[0]:,} windows x {X.shape[1]:,} features")
    counts = pd.Series(y).value_counts()
    print(f"classes: {dict(counts)}")
    print(f"imbalance: {counts.max() / counts.min():.1f}x\n")

    methods = [args.method] if args.method else ["kmeans", "som"]
    capacities = [args.capacity] if args.capacity else args.capacities

    seed_rows, stab_rows, rec_rows = [], [], []
    for method, cap in itertools.product(methods, capacities):
        if method == "som" and int(round(np.sqrt(cap))) ** 2 != cap:
            print(f"  skipping som cap={cap} (not a perfect square)")
            continue
        print(f"  == {method} @ capacity {cap} ==")
        s, st, rc = run_configuration(X, y, method, cap, args.seeds, args.max_pairs)
        seed_rows.append(s)
        stab_rows.append(st)
        rec_rows.append(rc)

    os.makedirs(args.out, exist_ok=True)
    suffix = f"_{args.method}_{args.capacity}" if args.capacity else ""
    pd.concat(seed_rows).to_csv(
        os.path.join(args.out, f"capacity_sweep{suffix}.csv"), index=False)
    pd.DataFrame(stab_rows).to_csv(
        os.path.join(args.out, f"stability_pairwise{suffix}.csv"), index=False)
    pd.concat(rec_rows).to_csv(
        os.path.join(args.out, f"recovery_by_class{suffix}.csv"), index=False)

    print(f"\nwrote 3 CSVs to {args.out}")
    print("\nrarest class, effective coverage by configuration:")
    rare = counts.idxmin()
    allrec = pd.concat(rec_rows)
    sub = allrec[allrec["class"] == rare].sort_values(["method", "capacity"])
    for _, r in sub.iterrows():
        print(f"  {r['method']:<7} cap={int(r['capacity']):<4} "
              f"{rare}: effcov={r['effective_coverage']:.3f}")


def self_test() -> None:
    """Synthetic imbalanced blobs: capacity should help the rare class."""
    from sklearn.datasets import make_blobs

    rng = np.random.default_rng(0)
    X, y = make_blobs(n_samples=[600, 300, 120, 25], n_features=40,
                      cluster_std=3.0, random_state=0)
    names = np.array(["healthy", "degrading", "critical", "imminent"])
    y = names[y]

    for cap in (4, 25, 100):
        _, st, rec = run_configuration(X, y, "kmeans", cap, seeds=5, max_pairs=10)
        rare = rec[rec["class"] == "imminent"].iloc[0]
        print(f"  cap={cap:<4} pairwise_ARI={st['pairwise_ari']:.3f} "
              f"imminent_effcov={rare['effective_coverage']:.3f}")

    assert 0.0 <= st["pairwise_nmi"] <= 1.0
    assert set(rec.columns) >= {"capture", "purity", "effective_coverage"}
    print("\nself-test passed: metrics well-formed, sweep runs end to end.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", help="output directory of 00_stage_cmapss.py")
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--label-col", default="health_state")
    p.add_argument("--capacities", type=int, nargs="+",
                   default=[4, 9, 16, 25, 49, 100, 225, 400])
    p.add_argument("--capacity", type=int, help="single capacity (SLURM array mode)")
    p.add_argument("--method", choices=["kmeans", "som"],
                   help="single method (SLURM array mode)")
    p.add_argument("--seeds", type=int, default=25)
    p.add_argument("--max-pairs", type=int, default=300)
    p.add_argument("--train-only", action="store_true",
                   help="fit on training engines only; keeps held-out engines clean")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.prepped:
        p.error("--prepped is required (or use --self-test)")
    run(args)


if __name__ == "__main__":
    sys.exit(main())