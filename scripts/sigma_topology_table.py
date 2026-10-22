#!/usr/bin/env python3
"""
sigma_topology_table.py
=======================
Builds the joint table the review asks for: what varying the neighbourhood
width does to rare-state recovery AND to topology preservation, side by side.

The question is no longer whether sigma changes recovery. It is whether an
intermediate sigma improves recovery because it preserves topology better, or
because it stops preserving topology at all. Reporting recovery alone cannot
distinguish those.

Two sources are combined.

    recovery   outputs/metrics/sigma_ablation_{cohort}.csv
               held-out effective coverage of the near-failure state, SOM minus
               K-means at the same prototype budget

    topology   outputs/metrics/sigma_criteria_{cohort}.csv
               inner-validation quantisation error and topographic error, as
               written by sigma_nested_selection.py

Runs of sigma_nested_selection.py that predate the patch adding that CSV print
the same numbers to their log but do not save them. Pass --log to recover them.

    python sigma_topology_table.py
    python sigma_topology_table.py --log logs/sigma_nested.log
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

LINE = re.compile(
    r"outer\s+(\d+)\s+sigma=([\d.]+)\s+inner:\s+QE\s+([\d.]+)\s+TE\s+([\d.]+)"
    r"\s+oracle effcov\s+([\d.]+)")
HEAD = re.compile(r"^(fd\d+_bal):\s+\d+x\d+ lattice")


def from_log(path):
    """Recover per-sigma criteria printed by an unpatched run."""
    rows, cohort = [], None
    with open(path) as fh:
        for ln in fh:
            h = HEAD.match(ln)
            if h:
                cohort = h.group(1)
                continue
            m = LINE.search(ln)
            if m and cohort:
                rows.append({"cohort": cohort, "outer": int(m.group(1)),
                             "sigma": float(m.group(2)), "qe": float(m.group(3)),
                             "te": float(m.group(4)),
                             "oracle_effcov": float(m.group(5))})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", default="outputs/metrics")
    p.add_argument("--log", default=None,
                   help="nested-selection log, for runs predating the CSV")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    crit = [pd.read_csv(f) for f in
            sorted(glob.glob(os.path.join(a.metrics, "sigma_criteria_*.csv")))]
    if a.log and os.path.exists(a.log):
        rec = from_log(a.log)
        if len(rec):
            have = set(pd.concat(crit).cohort) if crit else set()
            rec = rec[~rec.cohort.isin(have)]
            if len(rec):
                print(f"recovered {rec.cohort.nunique()} cohort(s) from {a.log}")
                crit.append(rec)
    if not crit:
        print("no criteria found; run sigma_nested_selection.py first")
        return 1
    crit = pd.concat(crit, ignore_index=True)

    rec_rows = []
    for f in sorted(glob.glob(os.path.join(a.metrics, "sigma_ablation_*.csv"))):
        d = pd.read_csv(f)
        tag = os.path.basename(f)[len("sigma_ablation_"):-4]
        if not tag.endswith("_bal"):
            tag += "_bal"
        km = d[d.method == "kmeans"].effcov.mean()
        for sg, g in d[d.method == "som"].groupby("sigma"):
            rec_rows.append({"cohort": tag, "sigma": sg,
                             "effcov": g.effcov.mean(),
                             "diff": g.effcov.mean() - km})
    rec = pd.DataFrame(rec_rows)

    ct = crit.groupby(["cohort", "sigma"], as_index=False)[["qe", "te"]].mean()
    tab = ct.merge(rec, on=["cohort", "sigma"], how="outer").sort_values(
        ["cohort", "sigma"], ascending=[True, False])

    print(f"\n{'cohort':<11}{'sigma':>8}{'SOM-KM':>9}{'TE':>9}{'QE':>9}")
    for coh, g in tab.groupby("cohort"):
        for _, r in g.iterrows():
            d = f"{r['diff']:+.3f}" if pd.notna(r.get("diff")) else "    --"
            te = f"{r['te']:.4f}" if pd.notna(r.get("te")) else "     --"
            qe = f"{r['qe']:.3f}" if pd.notna(r.get("qe")) else "    --"
            print(f"{coh:<11}{r['sigma']:>8.3f}{d:>9}{te:>9}{qe:>9}")
        print()

    out = a.out or os.path.join(a.metrics, "sigma_topology_table.csv")
    tab.to_csv(out, index=False)
    print(f"wrote {out}\n")
    print("READING THE TABLE")
    print("  If recovery improves as sigma falls while TE stays flat, the")
    print("  intermediate width preserves topology as well as the wide one and")
    print("  recovers more: the penalty at d/4 is a tuning artefact, not the")
    print("  price of topology preservation.")
    print("  If recovery improves while TE rises sharply, the gain comes from")
    print("  abandoning topology, and the trade-off interpretation stands.")


if __name__ == "__main__":
    sys.exit(main())
