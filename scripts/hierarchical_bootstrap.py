#!/usr/bin/env python3
"""
hierarchical_bootstrap.py
=========================
Addresses review points 1 and 3.

POINT 3 --- nested resampling. The fits in a holdout sweep are not 50
independent observations. Ten initialisations within one split share the same
training and held-out engines, so resampling all fits as though exchangeable
understates sampling uncertainty. This script resamples SPLITS with
replacement, then resamples INITIALISATIONS within each drawn split, which
respects the nesting. Differences are computed PAIRED: within a split and
seed, the SOM and K-means fits saw identical data, so the difference is taken
before aggregation rather than between two independently averaged means.

The naive interval is reported alongside so the effect of the correction is
visible. Expect the hierarchical interval to be wider, sometimes considerably,
when between-split variance is large relative to between-seed variance.

POINT 1 --- equivalence rather than absence of evidence. A confidence interval
containing zero establishes only that a difference was not detected. To claim
the two methods are practically equivalent requires an equivalence test
against a pre-specified margin. Two one-sided tests are performed here through
the interval correspondence: equivalence at level alpha holds when the
100(1-2*alpha) percent interval lies entirely inside (-margin, +margin). With
alpha = 0.05 that is the 90 percent interval.

THE MARGIN MUST BE JUSTIFIED ON OPERATIONAL GROUNDS, NOT CHOSEN TO PASS. State
in the paper what a given loss in effective coverage of the near-failure state
would mean for a maintenance programme, and set the margin from that. A margin
selected after seeing these intervals is not an equivalence test.

    python hierarchical_bootstrap.py --sweep outputs/metrics_holdout/holdout_sweep_fd001.csv
    python hierarchical_bootstrap.py --sweep ... --margin 0.05 --metric effcov_holdout_pooled
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def paired_differences(df: pd.DataFrame, metric: str) -> dict:
    """d[split] = array of (SOM - K-means) over seeds sharing that split."""
    out = {}
    for split, g in df.groupby("split"):
        s = g[g.method == "som"].set_index("seed")[metric]
        k = g[g.method == "kmeans"].set_index("seed")[metric]
        common = s.index.intersection(k.index)
        if len(common):
            out[split] = (s.loc[common] - k.loc[common]).to_numpy()
    return out


def hierarchical_boot(d: dict, n: int, rng) -> np.ndarray:
    """Resample splits with replacement, then seeds within each drawn split."""
    splits = list(d)
    draws = np.empty(n)
    for i in range(n):
        chosen = rng.choice(len(splits), len(splits), replace=True)
        vals = []
        for c in chosen:
            arr = d[splits[c]]
            vals.append(rng.choice(arr, len(arr), replace=True).mean())
        draws[i] = np.mean(vals)
    return draws


def naive_boot(d: dict, n: int, rng) -> np.ndarray:
    """Pool every fit and resample, ignoring the split structure."""
    flat = np.concatenate(list(d.values()))
    return np.array([rng.choice(flat, len(flat), replace=True).mean()
                     for _ in range(n)])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, help="holdout_sweep_*.csv")
    p.add_argument("--metric", default="effcov_holdout_pooled")
    p.add_argument("--margin", type=float, default=0.05,
                   help="equivalence margin; justify operationally")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    df = pd.read_csv(a.sweep)
    tag = os.path.basename(a.sweep).replace("holdout_sweep_", "").replace(".csv", "")
    rng = np.random.default_rng(0)
    nsplit = df.split.nunique()
    print(f"{tag}: metric={a.metric}, {nsplit} split(s), "
          f"{df.seed.nunique()} seeds, margin +/-{a.margin}")
    if nsplit < 3:
        print("  WARNING: hierarchical resampling over fewer than 3 splits is\n"
              "  unreliable. Treat these intervals as indicative only.")

    rows = []
    for cap in sorted(df.capacity.unique()):
        d = paired_differences(df[df.capacity == cap], a.metric)
        if not d:
            continue
        obs = float(np.mean([v.mean() for v in d.values()]))
        h = hierarchical_boot(d, a.n_boot, rng)
        nv = naive_boot(d, a.n_boot, rng)
        h95 = np.percentile(h, [2.5, 97.5])
        h90 = np.percentile(h, [5, 95])          # TOST at alpha = 0.05
        n95 = np.percentile(nv, [2.5, 97.5])
        equiv = (h90[0] > -a.margin) and (h90[1] < a.margin)
        differs = (h95[0] > 0) or (h95[1] < 0)
        between = np.std([v.mean() for v in d.values()], ddof=1) if len(d) > 1 else np.nan
        within = float(np.mean([v.std(ddof=1) for v in d.values() if len(v) > 1]))

        verdict = ("DIFFERENT" if differs else
                   "EQUIVALENT" if equiv else "INCONCLUSIVE")
        print(f"\n  capacity {cap}")
        print(f"    paired difference (SOM - KM): {obs:+.4f}")
        print(f"    hierarchical 95% CI: [{h95[0]:+.4f}, {h95[1]:+.4f}]")
        print(f"    naive        95% CI: [{n95[0]:+.4f}, {n95[1]:+.4f}]"
              f"   (width x{(h95[1]-h95[0])/(n95[1]-n95[0]):.2f})")
        print(f"    TOST 90% CI:         [{h90[0]:+.4f}, {h90[1]:+.4f}] "
              f"vs margin +/-{a.margin}")
        print(f"    between-split sd {between:.4f}, within-split sd {within:.4f}")
        print(f"    --> {verdict}")

        rows.append({
            "cohort": tag, "capacity": cap, "metric": a.metric,
            "diff": obs, "hier_lo95": h95[0], "hier_hi95": h95[1],
            "naive_lo95": n95[0], "naive_hi95": n95[1],
            "tost_lo90": h90[0], "tost_hi90": h90[1], "margin": a.margin,
            "between_split_sd": between, "within_split_sd": within,
            "verdict": verdict, "n_splits": nsplit,
        })

    out = a.out or os.path.join(os.path.dirname(a.sweep),
                                f"hierarchical_bootstrap_{tag}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nDIFFERENT   = 95% CI excludes zero")
    print("EQUIVALENT  = 90% CI inside the margin (TOST passes)")
    print("INCONCLUSIVE= neither; report as 'no detectable difference',")
    print("              NOT as evidence of equivalence.")


if __name__ == "__main__":
    sys.exit(main())
