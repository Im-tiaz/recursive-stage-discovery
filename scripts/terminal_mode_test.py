#!/usr/bin/env python3
"""
terminal_mode_test.py
=====================
Addresses the circularity objection (reviewer point 1).

The contiguity and terminal-assignment analyses both read structure off the
SOM lattice, so using them to explain why the SOM behaves as it does is
partly circular. This script asks the same question WITHOUT any SOM: in the
standardised feature space itself, do engines converge on one terminal region
or several?

Method
------
Each run-to-failure engine is reduced to a single point: the centroid of its
terminal windows (those labelled with the most severe health state, or the
last `--n-terminal` windows if that state is unavailable). With ~100 engines
in 360-392 dimensions, density estimates are meaningless without reduction, so
centroids are projected onto their leading principal components before any
clustering.

Three independent readings, none of which involve a lattice:

  1. MODEL SELECTION. Gaussian mixtures with 1..K components, scored by BIC.
     A cohort converging on one destination should select one component.
  2. CONNECTIVITY. Single-linkage merge heights. If the terminal centroids
     form one blob, the largest merge height is small relative to the spread;
     if several, one merge joins distant groups late.
  3. SILHOUETTE. At the BIC-selected component count, how well separated the
     groups are. Near zero means the split is not real.

Bootstrap over engines gives the stability of the selected component count.

    python terminal_mode_test.py --prepped data/fd001
    python terminal_mode_test.py --prepped data/fd003
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RARE  # noqa: E402


def terminal_centroids(X, md, klass, n_terminal):
    """One point per run-to-failure engine, from its most-degraded windows."""
    rows, units = [], []
    src = md[md["origin"] == "train"] if "origin" in md else md
    for u, g in src.groupby("unit"):
        sel = g[g["health_state"] == klass]
        if len(sel) < 2:                       # fall back to the last windows
            sel = g.nsmallest(n_terminal, "rul")
        rows.append(X[sel.index.to_numpy()].mean(axis=0))
        units.append(u)
    return np.asarray(rows), np.asarray(units)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--klass", default=RARE)
    p.add_argument("--n-terminal", type=int, default=5)
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--n-pcs", type=int, default=3)
    p.add_argument("--boot", type=int, default=500)
    a = p.parse_args()

    X = np.load(os.path.join(a.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(a.prepped, "metadata.csv"))
    tag = os.path.basename(a.prepped.rstrip("/"))

    C, units = terminal_centroids(X, md, a.klass, a.n_terminal)
    C = (C - C.mean(0)) / np.where(C.std(0) < 1e-8, 1.0, C.std(0))
    npc = min(a.n_pcs, C.shape[0] - 1, C.shape[1])
    pca = PCA(n_components=npc, random_state=0)
    Z = pca.fit_transform(C)
    print(f"{tag}: {len(units)} run-to-failure engines, "
          f"{npc} PCs retaining {100*pca.explained_variance_ratio_.sum():.1f}% "
          f"of centroid variance\n")

    # -- 1. model selection ------------------------------------------------
    ks = range(1, min(a.max_k, len(units) // 8) + 1)
    bic = {k: GaussianMixture(k, covariance_type="diag", n_init=5,
                              random_state=0).fit(Z).bic(Z) for k in ks}
    best = min(bic, key=bic.get)
    print("GMM BIC (lower is better):")
    for k, v in bic.items():
        print(f"  k={k}: {v:9.1f}{'   <- selected' if k == best else ''}")
    delta = bic[1] - bic[best]
    print(f"  selected k = {best}; BIC improvement over k=1: {delta:.1f}")

    # -- 2. connectivity ---------------------------------------------------
    Zl = linkage(Z, method="single")
    heights = Zl[:, 2]
    gap = heights[-1] / np.median(heights)
    print(f"\nsingle-linkage: final merge {heights[-1]:.3f}, "
          f"median merge {np.median(heights):.3f}, ratio {gap:.2f}")

    # -- 3a. silhouette scan (better powered than BIC at n ~ 100) ---------
    sil_by_k = {}
    for k in ks:
        if k < 2:
            continue
        lb = GaussianMixture(k, covariance_type="diag", n_init=5,
                             random_state=0).fit_predict(Z)
        if len(set(lb)) > 1:
            sil_by_k[k] = silhouette_score(Z, lb)
    if sil_by_k:
        k_sil = max(sil_by_k, key=sil_by_k.get)
        print("\nsilhouette by k:")
        for k, v in sil_by_k.items():
            print(f"  k={k}: {v:6.3f}{'   <- best' if k == k_sil else ''}")
        print(f"  best separated split: k={k_sil}, silhouette {sil_by_k[k_sil]:.3f}")
    else:
        k_sil, sil_by_k = 1, {}

    # -- 3b. silhouette at the BIC choice ----------------------------------
    if best > 1:
        lab = GaussianMixture(best, covariance_type="diag", n_init=5,
                              random_state=0).fit_predict(Z)
        sil = silhouette_score(Z, lab)
        sizes = dict(pd.Series(lab).value_counts().sort_index())
        print(f"silhouette at k={best}: {sil:.3f}; group sizes {sizes}")
    else:
        sil, sizes = float("nan"), {0: len(units)}
        print("silhouette: undefined at k=1 (single mode)")

    # -- bootstrap stability of k ------------------------------------------
    rng = np.random.default_rng(0)
    picks = []
    for _ in range(a.boot):
        idx = rng.choice(len(Z), len(Z), replace=True)
        Zb = Z[idx]
        try:
            b = {k: GaussianMixture(k, covariance_type="diag", n_init=1,
                                    random_state=0).fit(Zb).bic(Zb) for k in ks}
            picks.append(min(b, key=b.get))
        except Exception:
            continue
    vc = pd.Series(picks).value_counts(normalize=True).sort_index()
    print(f"\nbootstrap over engines ({len(picks)} resamples), selected k:")
    for k, f in vc.items():
        print(f"  k={k}: {100*f:5.1f}%")

    pd.DataFrame([{
        "cohort": tag, "n_engines": len(units), "n_pcs": npc,
        "var_explained": float(pca.explained_variance_ratio_.sum()),
        "best_k": best, "bic_gain_over_k1": delta,
        "silhouette": sil, "linkage_ratio": gap,
        "boot_frac_k1": float(vc.get(1, 0.0)),
        "boot_modal_k": int(vc.idxmax()),
        "k_max_silhouette": int(k_sil),
        "max_silhouette": float(max(sil_by_k.values())) if sil_by_k else float("nan"),
    }]).to_csv(os.path.join(a.out, f"terminal_mode_test_{tag}_pc{npc}.csv"),
               index=False)
    print(f"\nwrote {a.out}/terminal_mode_test_{tag}_pc{npc}.csv")
    print("\nRead the SILHOUETTE SCAN first: at ~100 engines, BIC is weakly")
    print("powered and its selection is unstable under resampling. A")
    print("single-destination fleet should show max silhouette near or below")
    print("0.25 (no separated split available) and a low linkage ratio; a")
    print("multi-destination fleet should show a clearly better-separated k.")
    print("The comparison that matters is BETWEEN cohorts, not against a")
    print("threshold.")


if __name__ == "__main__":
    sys.exit(main())
