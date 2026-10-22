#!/usr/bin/env python3
"""
_common.py
==========
Shared helpers for the validation scripts. Keeping the SOM fit, the BMU
mapping and the recovery metric in one place means every control measures the
same thing the sweep measures.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

STATES = ["healthy", "degrading", "critical", "imminent"]
RARE = "imminent"


def load_prepped(prepped: str, train_only: bool = True):
    """Return (X, metadata) from a 00_stage_cmapss.py output directory."""
    X = np.load(os.path.join(prepped, "X.npy"))
    md = pd.read_csv(os.path.join(prepped, "metadata.csv"))
    if train_only:
        keep = (md["split"] == "train").to_numpy()
        X, md = X[keep], md.loc[keep].reset_index(drop=True)
    return X, md


def fit_som(X: np.ndarray, side: int, seed: int, iters: int = 50_000):
    """SOM with the hyperparameters used throughout this study."""
    from minisom import MiniSom

    som = MiniSom(side, side, X.shape[1], sigma=max(1.0, side / 4),
                  learning_rate=0.5, random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, iters, verbose=False)
    return som


def bmu_flat(som, X: np.ndarray, side: int) -> np.ndarray:
    """Flat node index (0..side*side-1) of each sample's best-matching unit."""
    return np.array([np.ravel_multi_index(som.winner(x), (side, side))
                     for x in X])


def dominant_map(labels: np.ndarray, clusters: np.ndarray) -> pd.Series:
    """Plurality class of each cluster, indexed by cluster id."""
    return (pd.DataFrame({"y": labels, "c": clusters})
              .groupby("c")["y"].agg(lambda s: s.value_counts().idxmax()))


def owned_clusters(labels: np.ndarray, clusters: np.ndarray,
                   klass: str) -> np.ndarray:
    dom = dominant_map(labels, clusters)
    return dom.index[dom == klass].to_numpy()


def effcov(labels: np.ndarray, clusters: np.ndarray, klass: str,
           pooled: bool = False) -> float:
    """Effective coverage of one class. Matches 01_capacity_sweep.py.

    pooled=False -> Paper 2's per-cluster unweighted purity.
    pooled=True  -> size-weighted purity.
    """
    owned = owned_clusters(labels, clusters, klass)
    n_class = int((labels == klass).sum())
    if len(owned) == 0 or n_class == 0:
        return 0.0
    in_owned = np.isin(clusters, owned)
    hit = int(((labels == klass) & in_owned).sum())
    capture = hit / n_class
    if pooled:
        purity = hit / int(in_owned.sum())
    else:
        purity = float(np.mean([
            ((labels == klass) & (clusters == cid)).sum() / (clusters == cid).sum()
            for cid in owned
        ]))
    return capture * purity


def add_common_args(p, capacity_default: int = 400):
    p.add_argument("--prepped", required=True,
                   help="output directory of 00_stage_cmapss.py")
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--capacity", type=int, default=capacity_default)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--iters", type=int, default=50_000)
    return p


def side_of(capacity: int) -> int:
    side = int(round(np.sqrt(capacity)))
    if side * side != capacity:
        raise ValueError(f"capacity {capacity} is not a perfect square")
    return side


def save(df: pd.DataFrame, out: str, name: str, tag: str) -> None:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}_{tag}.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")
