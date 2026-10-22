#!/usr/bin/env python3
"""
early_paired_tests.py
=====================
Paired engine-level tests on the out-of-fold early-warning predictions.

The appendix currently compares strategies by their median lead times and notes
that the gaps are small relative to run-to-run variation. That is weaker than
the data allow. Because every engine now receives one out-of-fold prediction
under every strategy, the comparison can be made PAIRED: the same engine, the
same fold, the same held-out windows, differing only in whether the partition
was refined recursively.

Three comparisons per cohort and operating point:

    recursive - flat        does recursive refinement improve lead time over
                            the root partition alone
    recursive - centroid    does it improve over a distance-to-terminal-centroid
                            health indicator
    som - kmeans            does the root method matter

For each, the per-engine difference in lead time is bootstrapped over engines.
Engines flagged under one strategy and not the other are handled explicitly
rather than silently dropped: the default treats an unflagged engine as having
zero lead time, since an alert that never fires provides no warning. Use
--complete-only to restrict to engines flagged under both, and report whichever
you use.

Premature-warning rates are compared the same way, on truncated engines, which
never reach failure and therefore should not be flagged at all.

    python early_paired_tests.py
    python early_paired_tests.py --rho 75 --complete-only
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ORDER = ["fd001_bal", "fd002_bal", "fd003_bal", "fd004_bal"]
PAIRS = [("recursive", "flat"), ("recursive", "centroid"), ("flat", "centroid")]


def load(metrics):
    rows = []
    for f in sorted(glob.glob(os.path.join(metrics, "early_oof_*_engines.csv"))):
        m = re.search(r"early_oof_(fd\d+_bal)_(kmeans|som)_engines", f)
        if not m:
            continue
        d = pd.read_csv(f)
        d["cohort"], d["method"] = m.group(1), m.group(2)
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else None


def paired_diff(df, a, b, rho, complete_only, value="lead_time"):
    """Per-engine difference between two strategies at one operating point."""
    s = df[(df.rho == rho) & (df.run_to_failure)]
    A = s[s.strategy == a].set_index("unit")
    B = s[s.strategy == b].set_index("unit")
    idx = A.index.intersection(B.index)
    if not len(idx):
        return np.array([])
    va, vb = A.loc[idx, value], B.loc[idx, value]
    fa, fb = A.loc[idx, "flagged"], B.loc[idx, "flagged"]
    if complete_only:
        keep = fa & fb
        return (va[keep] - vb[keep]).to_numpy()
    # an alert that never fires gives no warning
    return (va.fillna(0.0).where(fa, 0.0)
            - vb.fillna(0.0).where(fb, 0.0)).to_numpy()


def boot_ci(x, rng, n=10_000):
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    reps = np.array([rng.choice(x, len(x), replace=True).mean()
                     for _ in range(n)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", default="outputs/metrics")
    p.add_argument("--rho", type=int, nargs="+", default=[50, 75, 100])
    p.add_argument("--method", default="kmeans", choices=["kmeans", "som"])
    p.add_argument("--complete-only", action="store_true",
                   help="restrict to engines flagged under both strategies")
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args()

    df = load(a.metrics)
    if df is None:
        print("no early_oof_*_engines.csv found")
        return 1
    rng = np.random.default_rng(0)
    mode = "flagged under both" if a.complete_only else "unflagged counted as 0"
    print(f"paired lead-time differences, {a.method} root, {mode}\n")

    rows = []
    for coh in ORDER:
        d = df[(df.cohort == coh) & (df.method == a.method)]
        if not len(d):
            continue
        print(f"=== {coh} ===")
        print(f"{'rho':>5}{'comparison':<24}{'n':>5}{'mean diff':>11}{'95% CI':>22}")
        for rho in a.rho:
            for x, y in PAIRS:
                v = paired_diff(d, x, y, rho, a.complete_only)
                m_, lo, hi = boot_ci(v, rng, a.n_boot)
                if m_ != m_:
                    continue
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"{rho:>5}{f'{x} - {y}':<24}{len(v):>5}{m_:>+11.2f}"
                      f"{f'[{lo:+.2f}, {hi:+.2f}]':>22} {star}")
                rows.append({"cohort": coh, "method": a.method, "rho": rho,
                             "comparison": f"{x}-{y}", "n_engines": len(v),
                             "mean_diff": m_, "lo": lo, "hi": hi,
                             "excludes_zero": bool(lo > 0 or hi < 0),
                             "pairing": mode})
        print()

    # ---- premature warnings on truncated engines --------------------------
    print("premature-warning rate on truncated engines, which never fail")
    print(f"{'cohort':<11}{'rho':>5}{'flat':>8}{'recursive':>11}{'centroid':>10}"
          f"{'n':>6}")
    for coh in ORDER:
        d = df[(df.cohort == coh) & (df.method == a.method) & (~df.run_to_failure)]
        if not len(d):
            continue
        for rho in a.rho:
            s = d[d.rho == rho]
            g = s.groupby("strategy").flagged.mean()
            n = int(s[s.strategy == "recursive"].shape[0])
            print(f"{coh:<11}{rho:>5}{g.get('flat', np.nan):>8.3f}"
                  f"{g.get('recursive', np.nan):>11.3f}"
                  f"{g.get('centroid', np.nan):>10.3f}{n:>6}")

    out = os.path.join(a.metrics, f"early_paired_{a.method}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nA comparison whose interval excludes zero is a difference. One that")
    print("spans zero should be reported as a direction, not a finding.")


if __name__ == "__main__":
    sys.exit(main())
