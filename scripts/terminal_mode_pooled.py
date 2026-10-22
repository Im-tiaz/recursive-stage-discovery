#!/usr/bin/env python3
"""
terminal_mode_pooled.py
=======================
The terminal-state geometry comparison, computed in one shared coordinate
system rather than four separate ones.

`terminal_mode_test.py` standardises each cohort's engine centroids on that
cohort and fits a separate principal component basis to it. Silhouette values
obtained after four independently fitted transforms are then compared across
cohorts, which invites the objection that the comparison is not like for like.
Since all four cohorts are now staged on the same fifteen sensor channels and
carry identical 360-dimensional features, a common transform is available.

Here the terminal centroids of every cohort are pooled, one scaler and one
principal component basis are fitted to the pooled set, and every cohort is
projected through that shared basis. Separation statistics are then computed
per cohort in coordinates that are directly comparable.

Reports, per cohort, in the pooled basis:

    best_k, silhouette   the best-separated partition over k = 2..6, by mean
                         silhouette of a diagonal-covariance Gaussian mixture
    bic_gain             improvement of the best mixture over a single
                         component, in BIC units
    linkage_ratio        final single-linkage merge height over the median
                         merge height
    bootstrap support    stability of the selected component count over
                         resampled engines

Run alongside `terminal_mode_test.py` and report as a sensitivity analysis. If
the ordering between cohorts is preserved, the per-cohort transforms were not
driving it.

    python terminal_mode_pooled.py --cohorts data/fd001_bal data/fd002_bal \
        data/fd003_bal data/fd004_bal
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


def centroids(prepped, klass, n_terminal):
    """One point per run-to-failure engine, from its most degraded windows."""
    X = np.load(os.path.join(prepped, "X.npy"))
    md = pd.read_csv(os.path.join(prepped, "metadata.csv"))
    src = md[md["origin"] == "train"] if "origin" in md else md
    rows = []
    for _, g in src.groupby("unit"):
        sel = g[g["health_state"] == klass]
        if len(sel) < 2:
            sel = g.nsmallest(n_terminal, "rul")
        rows.append(X[sel.index.to_numpy()].mean(axis=0))
    return np.asarray(rows)


def stats(Z, ks, boot, rng):
    bic = {k: GaussianMixture(k, covariance_type="diag", n_init=5,
                              random_state=0).fit(Z).bic(Z) for k in ks}
    best_bic = min(bic, key=bic.get)

    sil = {}
    for k in ks:
        if k < 2:
            continue
        lab = GaussianMixture(k, covariance_type="diag", n_init=5,
                              random_state=0).fit_predict(Z)
        if len(set(lab)) > 1:
            sil[k] = silhouette_score(Z, lab)
    k_sil = max(sil, key=sil.get) if sil else 1

    h = linkage(Z, method="single")[:, 2]
    ratio = float(h[-1] / np.median(h))

    picks = []
    for _ in range(boot):
        idx = rng.choice(len(Z), len(Z), replace=True)
        Zb = Z[idx]
        try:
            b = {k: GaussianMixture(k, covariance_type="diag", n_init=1,
                                    random_state=0).fit(Zb).bic(Zb) for k in ks}
            picks.append(min(b, key=b.get))
        except Exception:
            continue
    vc = pd.Series(picks).value_counts(normalize=True) if picks else pd.Series()

    return {"best_k_bic": best_bic,
            "bic_gain_over_k1": float(bic[1] - bic[best_bic]),
            "k_max_silhouette": int(k_sil),
            "max_silhouette": float(max(sil.values())) if sil else float("nan"),
            "linkage_ratio": ratio,
            "boot_frac_k1": float(vc.get(1, 0.0)) if len(vc) else float("nan"),
            "boot_modal_k": int(vc.idxmax()) if len(vc) else -1}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohorts", nargs="+", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--klass", default=RARE)
    p.add_argument("--n-terminal", type=int, default=5)
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--boot", type=int, default=500)
    a = p.parse_args()
    rng = np.random.default_rng(0)

    blocks, tags = [], []
    for c in a.cohorts:
        C = centroids(c, a.klass, a.n_terminal)
        blocks.append(C)
        tags.append(os.path.basename(c.rstrip("/")))
        print(f"  {tags[-1]}: {C.shape[0]} engines, {C.shape[1]} features")
    if len({b.shape[1] for b in blocks}) != 1:
        print("cohorts have different feature counts; a pooled basis is not "
              "meaningful. Restage on a common sensor set first.")
        return 1

    pooled = np.vstack(blocks)
    mu, sd = pooled.mean(0), pooled.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    pooled_z = (pooled - mu) / sd
    npc = min(a.n_pcs, pooled_z.shape[0] - 1, pooled_z.shape[1])
    pca = PCA(n_components=npc, random_state=0).fit(pooled_z)
    print(f"\npooled basis: {len(pooled)} engines, {npc} components retaining "
          f"{100 * pca.explained_variance_ratio_.sum():.1f}% of pooled variance")

    ks = range(1, a.max_k + 1)
    rows = []
    off = 0
    for tag, C in zip(tags, blocks):
        Z = pca.transform((C - mu) / sd)
        r = stats(Z, ks, a.boot, rng)
        r.update({"cohort": tag, "n_engines": len(C), "n_pcs": npc,
                  "basis": "pooled"})
        rows.append(r)
        off += len(C)

    df = pd.DataFrame(rows)[
        ["cohort", "n_engines", "n_pcs", "basis", "best_k_bic",
         "bic_gain_over_k1", "k_max_silhouette", "max_silhouette",
         "linkage_ratio", "boot_frac_k1", "boot_modal_k"]]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"terminal_mode_pooled_pc{npc}.csv")
    df.to_csv(path, index=False)
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nwrote {path}")
    print("\nCompare against terminal_mode_test_*_pc10.csv, which uses a")
    print("per-cohort scaler and basis. If the ordering of max_silhouette and")
    print("linkage_ratio between cohorts is preserved, the separation is not an")
    print("artefact of fitting four separate transforms.")


if __name__ == "__main__":
    sys.exit(main())
