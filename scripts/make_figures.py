#!/usr/bin/env python3
"""
make_figures.py
===============
Generates every figure in the manuscript from the analysis outputs on disk.

Each figure is written as a vector PDF into paper/Figures/, at a width matching
a single-column journal page, with type sizes chosen so that axis labels remain
legible at final scale. Missing inputs are skipped with a warning rather than
raising, so the script can be run before every analysis has completed.

    python make_figures.py --out paper/Figures

Figures
-------
    fig1_design        the four cohorts as a 2x2, with prevalence and window
                       counts; no data files required
    fig2_contrast      SOM minus K-means held-out recovery by cohort and
                       prototype budget, with hierarchical bootstrap intervals
    fig3_sigma         held-out recovery against neighbourhood width, one line
                       per cohort, with the study setting marked
    fig4_terminal      terminal-centroid projections, one panel per cohort
    fig5_geometry      silhouette and single-linkage ratio by cohort
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

COHORTS = [("fd001_bal", "FD001", "1 cond, 1 mode"),
           ("fd002_bal", "FD002", "6 cond, 1 mode"),
           ("fd003_bal", "FD003", "1 cond, 2 modes"),
           ("fd004_bal", "FD004", "6 cond, 2 modes")]
C1, C2 = "#1F5FA9", "#B45309"          # one fault mode, two fault modes
COL = {"fd001_bal": C1, "fd002_bal": C1, "fd003_bal": C2, "fd004_bal": C2}


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def paired(df, metric):
    out = {}
    for s, g in df.groupby("split"):
        a = g[g.method == "som"].set_index("seed")[metric]
        b = g[g.method == "kmeans"].set_index("seed")[metric]
        k = a.index.intersection(b.index)
        if len(k):
            out[s] = (a.loc[k] - b.loc[k]).to_numpy()
    return out


def hier_ci(d, rng, n=5000):
    sp = list(d)
    reps = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(sp), len(sp))
        reps[i] = np.mean([rng.choice(d[sp[c]], len(d[sp[c]])).mean()
                           for c in pick])
    return float(np.mean([v.mean() for v in d.values()])), \
        *np.percentile(reps, [2.5, 97.5])


# --------------------------------------------------------------------------
def fig1_design(out):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    cells = {(0, 0): ("FD001", "5,665 win\n4.6% rare", C1),
             (1, 0): ("FD003", "7,183 win\n3.8% rare", C2),
             (0, 1): ("FD002", "5,681 win\n4.8% rare", C1),
             (1, 1): ("FD004", "7,349 win\n3.8% rare", C2)}
    for (cx, cy), (name, sub, col) in cells.items():
        ax.add_patch(Rectangle((cx, cy), .92, .92, facecolor=col, alpha=.12,
                               edgecolor=col, lw=1.2))
        ax.text(cx + .46, cy + .62, name, ha="center", fontsize=9,
                fontweight="bold", color=col)
        ax.text(cx + .46, cy + .30, sub, ha="center", fontsize=6.5, color="#333")
    ax.set_xlim(-.15, 2.05); ax.set_ylim(-.15, 2.05)
    ax.set_xticks([.46, 1.46]); ax.set_xticklabels(["1 fault mode", "2 fault modes"])
    ax.set_yticks([.46, 1.46]); ax.set_yticklabels(["1 operating\ncondition",
                                                    "6 operating\nconditions"])
    ax.set_aspect("equal"); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cohorts, balanced to 200 engines and 360 shared features",
                 fontsize=7.5, pad=8)
    save(fig, out, "fig1_design")


def fig2_contrast(out, d2x2):
    rows = []
    rng = np.random.default_rng(0)
    for tag, name, design in COHORTS:
        f = os.path.join(d2x2, f"holdout_sweep_{tag}.csv")
        if not os.path.exists(f):
            print(f"  skip fig2: missing {f}"); return
        df = pd.read_csv(f)
        for cap in sorted(df.capacity.unique()):
            m, lo, hi = hier_ci(paired(df[df.capacity == cap],
                                       "effcov_holdout_pooled"), rng)
            rows.append({"tag": tag, "name": name, "design": design,
                         "cap": cap, "m": m, "lo": lo, "hi": hi})
    r = pd.DataFrame(rows)
    caps = sorted(r.cap.unique())
    fig, axes = plt.subplots(1, len(caps), figsize=(5.6, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cap in zip(axes, caps):
        s = r[r.cap == cap]
        y = np.arange(len(s))[::-1]
        ax.axvline(0, color="#999", lw=.8, zorder=0)
        ax.errorbar(s.m, y, xerr=[s.m - s.lo, s.hi - s.m], fmt="o", ms=4,
                    lw=1.2, capsize=2.5,
                    ecolor=[COL[t] for t in s.tag][0], color="none")
        for yi, (_, row) in zip(y, s.iterrows()):
            ax.errorbar(row.m, yi, xerr=[[row.m - row.lo], [row.hi - row.m]],
                        fmt="o", ms=4, lw=1.2, capsize=2.5,
                        color=COL[row.tag], ecolor=COL[row.tag])
        ax.set_yticks(y); ax.set_yticklabels(
            [f"{n}\n{d}" for n, d in zip(s.name, s.design)])
        ax.set_title(f"{cap} prototypes")
        ax.set_xlabel("SOM $-$ $K$-means")
    axes[0].set_ylabel("")
    fig.suptitle("Held-out recovery of the near-failure state", y=1.02,
                 fontsize=9)
    save(fig, out, "fig2_contrast")


def fig3_sigma(out, dmet):
    """Recovery and quantisation error against initial neighbourhood width."""
    import glob as _g
    fs = sorted(_g.glob(os.path.join(dmet, "sigma_paired_*.csv")))
    if not fs:
        print("  skip fig3: no sigma_paired_*.csv"); return
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
    for f in fs:
        tag = os.path.basename(f)[13:-4]
        if not tag.endswith("_bal"):
            tag += "_bal"
        d = pd.read_csv(f)
        g = d.groupby("sigma0")
        col = COL.get(tag, "#666")
        name = tag.replace("_bal", "").upper()
        ls = "-" if tag in ("fd001_bal", "fd003_bal") else "--"
        axes[0].plot(g.sigma0.first(), g.ec_diff.mean(), "o" + ls, ms=3.5,
                     lw=1.3, color=col, label=name)
        axes[1].plot(g.sigma0.first(), g.qe.mean(), "o" + ls, ms=3.5,
                     lw=1.3, color=col, label=name)
    axes[0].axhline(0, color="#999", lw=.8)
    for ax, lab in zip(axes, ["SOM $-$ $K$-means", "quantisation error"]):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"initial width $\sigma_0$")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, ncol=2, fontsize=6)
    fig.suptitle("Recovery and quantisation fidelity against neighbourhood width",
                 fontsize=8.5, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, out, "fig3_sigma")


def fig4_terminal(out, ddata):
    from sklearn.decomposition import PCA
    panels = []
    for tag, name, design in COHORTS:
        xp, mp = os.path.join(ddata, tag, "X.npy"), \
                 os.path.join(ddata, tag, "metadata.csv")
        if not (os.path.exists(xp) and os.path.exists(mp)):
            continue
        X = np.load(xp); md = pd.read_csv(mp)
        src = md[md.origin == "train"] if "origin" in md else md
        C = []
        for _, g in src.groupby("unit"):
            sel = g[g.health_state == "imminent"]
            if len(sel) < 2:
                sel = g.nsmallest(5, "rul")
            C.append(X[sel.index.to_numpy()].mean(0))
        C = np.asarray(C)
        C = (C - C.mean(0)) / np.where(C.std(0) < 1e-8, 1, C.std(0))
        Z = PCA(n_components=2, random_state=0).fit_transform(C)
        panels.append((name, design, Z, COL[tag]))
    if not panels:
        print("  skip fig4: no staged cohorts found"); return
    fig, axes = plt.subplots(1, len(panels), figsize=(1.55 * len(panels), 2.35),
                             sharex=False, sharey=False)
    for ax, (name, design, Z, col) in zip(np.atleast_1d(axes), panels):
        ax.scatter(Z[:, 0], Z[:, 1], s=9, color=col, alpha=.75, lw=0)
        ax.set_title(f"{name}\n{design}", fontsize=7, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#bbb")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("Terminal-state centroids, one point per run-to-failure engine",
                 fontsize=8, y=0.99)
    save(fig, out, "fig4_terminal")


def fig5_geometry(out, dmet):
    rows = []
    for tag, name, design in COHORTS:
        f = os.path.join(dmet, f"terminal_mode_test_{tag}_pc10.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f).iloc[0]
        rows.append({"name": name, "design": design, "tag": tag,
                     "sil": d.max_silhouette, "link": d.linkage_ratio})
    if not rows:
        print("  skip fig5: no terminal_mode_test_*_pc10.csv"); return
    r = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))
    x = np.arange(len(r))
    for ax, col, lab in zip(axes, ["sil", "link"],
                            ["max silhouette", "single-linkage ratio"]):
        ax.bar(x, r[col], color=[COL[t] for t in r.tag], width=.62)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{d}" for n, d in zip(r.name, r.design)],
                           fontsize=6, rotation=30, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel(lab)
        ax.margins(x=0.08)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("Terminal-state separation, measured without a lattice",
                 fontsize=8.5, y=0.99)
    save(fig, out, "fig5_geometry")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="paper/Figures")
    p.add_argument("--metrics", default="outputs/metrics")
    p.add_argument("--sigma", default="outputs/metrics_sigma_full",
                   help="directory holding sigma_paired_*.csv")
    p.add_argument("--twobytwo", default="outputs/metrics_2x2_s10")
    p.add_argument("--data", default="data")
    a = p.parse_args()
    print(f"writing figures to {a.out}")
    fig1_design(a.out)
    fig2_contrast(a.out, a.twobytwo)
    fig3_sigma(a.out, a.sigma)
    fig4_terminal(a.out, a.data)
    fig5_geometry(a.out, a.metrics)
    print("done")


if __name__ == "__main__":
    sys.exit(main())