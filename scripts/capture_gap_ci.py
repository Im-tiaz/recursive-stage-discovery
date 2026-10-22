#!/usr/bin/env python3
"""
capture_gap_ci.py
=================
Confidence intervals for the per-class capture and purity gaps.

The class-wise decomposition is now used mechanistically: the manuscript reads
the near-failure capture gap as positive in the single-fault-mode cohorts and
negative in the two-fault-mode cohorts, and locates the cohort contrast in
recall rather than in purity. Differences of 0.058 and 0.041 are interpreted as
directional without any statement of uncertainty. This script supplies it.

Resampling matches `hierarchical_bootstrap.py`: splits are drawn with
replacement, then initialisations within each drawn split, because
initialisations sharing a split share the same training and held-out engines.
Gaps are formed paired within split and initialisation, since the two methods
saw identical data there.

    python capture_gap_ci.py
    python capture_gap_ci.py --class-name imminent --n-boot 20000
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ORDER = ["fd001_bal", "fd002_bal", "fd003_bal", "fd004_bal"]
DESIGN = {"fd001_bal": "1c, 1m", "fd002_bal": "6c, 1m",
          "fd003_bal": "1c, 2m", "fd004_bal": "6c, 2m"}


def paired(df, metric):
    """d[split] = array over seeds of (SOM - K-means) for one class."""
    out = {}
    for s, g in df.groupby("split"):
        a = g[g.method == "som"].set_index("seed")[metric]
        b = g[g.method == "kmeans"].set_index("seed")[metric]
        k = a.index.intersection(b.index)
        if len(k):
            out[s] = (a.loc[k] - b.loc[k]).to_numpy()
    return out


def hier(d, rng, n):
    sp = list(d)
    reps = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(sp), len(sp))
        reps[i] = np.mean([rng.choice(d[sp[c]], len(d[sp[c]])).mean()
                           for c in pick])
    obs = float(np.mean([v.mean() for v in d.values()]))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", default="outputs/metrics")
    p.add_argument("--class-name", default="imminent")
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for tag in ORDER:
        f = os.path.join(a.metrics, f"capture_gap_holdout_{tag}.csv")
        if not os.path.exists(f):
            print(f"missing {f}")
            continue
        d = pd.read_csv(f)
        d = d[d["class"] == a.class_name]
        if not len(d):
            print(f"{tag}: no rows for class '{a.class_name}'")
            continue
        for metric, label in [("capture", "capture"),
                              ("purity_pooled", "purity")]:
            obs, lo, hi = hier(paired(d, metric), rng, a.n_boot)
            rows.append({"cohort": tag, "design": DESIGN[tag],
                         "quantity": label, "gap": obs, "lo": lo, "hi": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
    if not rows:
        return 1

    r = pd.DataFrame(rows)
    out = os.path.join(a.metrics, f"capture_gap_ci_{a.class_name}.csv")
    r.to_csv(out, index=False)

    print(f"\nclass '{a.class_name}', SOM minus K-means, hierarchical bootstrap")
    print(f"{'cohort':<11}{'design':<9}{'quantity':<9}{'gap':>9}{'95% CI':>22}")
    for _, x in r.iterrows():
        star = "*" if x.excludes_zero else " "
        print(f"{x.cohort:<11}{x.design:<9}{x.quantity:<9}{x.gap:>+9.4f}"
              f"{f'[{x.lo:+.4f}, {x.hi:+.4f}]':>22} {star}")

    cap = r[r.quantity == "capture"]
    sig = cap.excludes_zero.sum()
    print(f"\ncapture gap excludes zero in {sig} of {len(cap)} cohorts")
    print("If the two single-fault-mode capture gaps are positive with intervals")
    print("excluding zero, and the two two-fault-mode gaps negative likewise,")
    print("the recall reading is supported. Where an interval spans zero, report")
    print("the point estimate as a direction only and say so.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
