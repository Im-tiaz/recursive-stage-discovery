#!/usr/bin/env python3
"""
factorial_contrasts.py
======================
Uncertainty for the cross-cohort contrasts themselves (review point 2).

The manuscript reported point estimates for the fault-mode contrast, the
operating-condition contrast, and their interaction, with no interval for any
of them, and argued that one factor matters more than another because cells in
one column were individually significant and cells in the other were not.
"Significant in A but not in B" does not establish that A and B differ. This
script forms each contrast directly and bootstraps it.

WHAT THIS DOES AND DOES NOT ESTIMATE
------------------------------------
There is exactly one C-MAPSS subset per factor combination. Engine splits and
initialisations are repeated observations WITHIN each subset, not independent
replications of the treatments. The intervals below therefore describe
sampling variability in the observed contrast across engines and
initialisations within these four benchmark subsets. They do NOT support
inference to a population of datasets with one or two fault modes, and the
quantities should be described as benchmark contrasts rather than as factorial
main effects. This limitation is structural and no resampling scheme can
remove it.

Resampling is hierarchical and paired, matching hierarchical_bootstrap.py:
splits are drawn with replacement within each cohort, then initialisations
within each drawn split, and the SOM-minus-K-means difference is formed within
split and initialisation before aggregation.

Also reports held-out capture and purity separately (review point 6), since a
product of the two can be reached by operationally different behaviours.

    python factorial_contrasts.py --dir outputs/metrics_2x2 --capacity 400
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# cohort -> (fault modes, operating conditions)
DESIGN = {"fd001_bal": (1, 1), "fd002_bal": (1, 6),
          "fd003_bal": (2, 1), "fd004_bal": (2, 6)}


def paired(df, metric):
    """d[split] = array over seeds of (SOM - K-means) for one cohort/capacity."""
    out = {}
    for split, g in df.groupby("split"):
        s = g[g.method == "som"].set_index("seed")[metric]
        k = g[g.method == "kmeans"].set_index("seed")[metric]
        common = s.index.intersection(k.index)
        if len(common):
            out[split] = (s.loc[common] - k.loc[common]).to_numpy()
    return out


def draw(d, rng):
    """One hierarchical bootstrap replicate of a cohort's mean difference."""
    splits = list(d)
    chosen = rng.choice(len(splits), len(splits), replace=True)
    return float(np.mean([rng.choice(d[splits[c]], len(d[splits[c]])).mean()
                          for c in chosen]))


def ci(v, lo=2.5, hi=97.5):
    return np.percentile(v, [lo, hi])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="outputs/metrics_2x2")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--metric", default="effcov_holdout_pooled")
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args()
    rng = np.random.default_rng(0)

    cells, frames = {}, {}
    for tag in DESIGN:
        f = os.path.join(a.dir, f"holdout_sweep_{tag}.csv")
        if not os.path.exists(f):
            print(f"MISSING: {f}")
            return 1
        df = pd.read_csv(f)
        df = df[df.capacity == a.capacity]
        frames[tag] = df
        cells[tag] = paired(df, a.metric)

    print(f"capacity {a.capacity}, metric {a.metric}, "
          f"{a.n_boot:,} hierarchical replicates\n")

    # ---- per-cohort means with intervals ---------------------------------
    reps = {t: np.array([draw(cells[t], rng) for _ in range(a.n_boot)])
            for t in DESIGN}
    print("per-cohort SOM minus K-means")
    print(f"{'cohort':<11}{'modes':>6}{'cond':>6}{'diff':>9}{'95% CI':>22}")
    for t, (fm, oc) in DESIGN.items():
        obs = float(np.mean([v.mean() for v in cells[t].values()]))
        lo, hi = ci(reps[t])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"{t:<11}{fm:>6}{oc:>6}{obs:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22} {star}")

    # ---- contrasts, formed within each replicate -------------------------
    fm_c = (reps["fd003_bal"] + reps["fd004_bal"]) / 2 - \
           (reps["fd001_bal"] + reps["fd002_bal"]) / 2
    oc_c = (reps["fd002_bal"] + reps["fd004_bal"]) / 2 - \
           (reps["fd001_bal"] + reps["fd003_bal"]) / 2
    inter = (reps["fd004_bal"] - reps["fd003_bal"]) - \
            (reps["fd002_bal"] - reps["fd001_bal"])
    diff_of = fm_c - oc_c          # is the fault-mode contrast the larger one?

    print("\nbenchmark contrasts (NOT factorial main effects -- see header)")
    rows = []
    for name, v in [("fault-mode contrast", fm_c),
                    ("op-condition contrast", oc_c),
                    ("interaction", inter),
                    ("fault-mode minus op-condition", diff_of)]:
        lo, hi = ci(v)
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {name:<32}{v.mean():>+9.4f}  [{lo:+.4f}, {hi:+.4f}] {star}")
        rows.append({"quantity": name, "capacity": a.capacity,
                     "metric": a.metric, "estimate": v.mean(),
                     "lo95": lo, "hi95": hi,
                     "excludes_zero": bool(lo > 0 or hi < 0)})

    print("\n  The last row is the test the manuscript needs: it asks directly")
    print("  whether the fault-mode contrast is larger than the")
    print("  operating-condition contrast, rather than inferring it from one")
    print("  being significant and the other not.")

    # ---- capture and purity separately (review point 6) ------------------
    print(f"\nheld-out capture and purity at capacity {a.capacity}")
    print(f"{'cohort':<11}{'KM cap':>9}{'SOM cap':>9}{'KM pur':>9}{'SOM pur':>9}")
    for t in DESIGN:
        d = frames[t]
        k, s = d[d.method == "kmeans"], d[d.method == "som"]
        print(f"{t:<11}{k.capture_holdout.mean():>9.3f}"
              f"{s.capture_holdout.mean():>9.3f}"
              f"{k.purity_holdout.mean():>9.3f}{s.purity_holdout.mean():>9.3f}")

    # ---- prevalence confound, made explicit (review point 3) -------------
    print("\nrare-state prevalence by cohort (held-out windows)")
    for t in DESIGN:
        d = frames[t]
        print(f"  {t:<11}n_owned mean {d.n_owned.mean():>6.1f}")
    print("  NOTE: prevalence is 4.6-4.8% in the one-fault-mode cohorts and")
    print("  3.8% in both two-fault-mode cohorts, so the contrast above is")
    print("  confounded with rarity. A prevalence-matched analysis is required")
    print("  before the fault-mode reading can be preferred.")

    out = os.path.join(a.dir, f"factorial_contrasts_cap{a.capacity}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
