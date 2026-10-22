#!/usr/bin/env python3
"""
04_recursive_early_prediction.py
================================
Recursive refinement of a flat partition into a tree whose terminal nodes are
scored by remaining useful life (RUL), then used to raise an early warning on
held-out engines.

Rationale
---------
A flat partition labels a window by its CURRENT health state. Clusters
dominated by a non-critical state are homogeneous in label but heterogeneous in
time-to-failure: two windows may both look healthy while one engine fails in 40
cycles and the other in 300. Recursive sub-clustering, in the manner of Ahmed
and Soliman's first study, splits that heterogeneity. A sub-cluster whose
TRAINING members were mostly close to failure becomes an at-risk node, and a
held-out window routed into it is flagged even when its own label still reads
healthy. That step is prediction rather than detection.

Procedure
---------
1. Fit the root partition on training engines (K-means or SOM, matched budget).
2. For every node, compute the interquartile range of training-window RUL. A
   node is heterogeneous if that spread exceeds `--rul-iqr-stop` cycles.
3. Sub-cluster heterogeneous nodes with K-means into `--branch` children.
   Recurse to `--max-depth`, stopping a node when it holds fewer than
   `--min-node` training windows, when its RUL spread is below the threshold,
   or when a split fails to reduce within-node RUL variance by `--min-gain`.
4. Score each terminal node by the median RUL of its training windows. Nodes at
   or below `--risk-rul` are at-risk.
5. Route held-out windows down the tree. For each held-out engine record the
   first cycle landing in an at-risk node, and the RUL at that cycle.

Endpoints, all at engine level
------------------------------
    detection rate     fraction of held-out run-to-failure engines ever flagged
    lead time          RUL in cycles at the first flag (higher is earlier)
    premature rate     flagged while RUL still exceeds the Heimes change point
    missed rate        never flagged before the final observed cycle

HONEST NOTES
------------
* Training RUL defines the at-risk nodes. That is a label, used exactly as
  cluster ownership is used elsewhere in this study. Held-out RUL is used only
  to score. State this in the paper.
* Truncated engines never reach failure, so detection rate and lead time are
  computed over run-to-failure engines only. Truncated engines are used for the
  premature-flag rate, where their higher RUL range is informative.
* A null result is a result. If sub-clusters of the healthy region do not
  separate on future RUL, lead times will cluster near the change point and the
  correct conclusion is that the sensors carry no pre-change-point signal in
  this benchmark.

    python 04_recursive_early_prediction.py --prepped data/fd001_bal
    python 04_recursive_early_prediction.py --prepped data/fd003_bal \
        --method som --capacity 400 --max-depth 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import fit_som, side_of  # noqa: E402

CHANGE_POINT = 125          # Heimes piecewise-linear RUL convention


# --------------------------------------------------------------------------
class Node:
    __slots__ = ("idx", "depth", "children", "median_rul", "iqr", "n", "at_risk",
                 "model", "node_id")

    def __init__(self, idx, depth, node_id):
        self.idx = idx
        self.depth = depth
        self.node_id = node_id
        self.children = []
        self.model = None
        self.median_rul = np.nan
        self.iqr = np.nan
        self.n = len(idx)
        self.at_risk = False


def grow(X, rul, idx, depth, args, counter, log):
    """Recursively split a node until a stopping rule fires."""
    node = Node(idx, depth, counter[0])
    counter[0] += 1
    r = rul[idx]
    node.median_rul = float(np.median(r))
    node.iqr = float(np.percentile(r, 75) - np.percentile(r, 25))

    stop = None
    if depth >= args.max_depth:
        stop = "max_depth"
    elif len(idx) < args.min_node * args.branch:
        stop = "min_node"
    elif node.iqr <= args.rul_iqr_stop:
        stop = "rul_homogeneous"

    if stop is None:
        km = KMeans(n_clusters=args.branch, n_init=3,
                    random_state=args.seed + depth).fit(X[idx])
        parts = [idx[km.labels_ == c] for c in range(args.branch)]
        if min(len(p) for p in parts) < args.min_node:
            stop = "degenerate_split"
        else:
            # variance of RUL explained by the split
            before = float(np.var(r))
            after = float(np.sum([len(p) * np.var(rul[p]) for p in parts]) / len(idx))
            gain = 0.0 if before <= 0 else (before - after) / before
            if gain < args.min_gain:
                stop = f"gain_{gain:.3f}"
            else:
                node.model = km
                log.append({"node": node.node_id, "depth": depth, "n": len(idx),
                            "median_rul": node.median_rul, "iqr": node.iqr,
                            "split": True, "rul_var_gain": gain})
                node.children = [grow(X, rul, p, depth + 1, args, counter, log)
                                 for p in parts]
                return node

    log.append({"node": node.node_id, "depth": depth, "n": len(idx),
                "median_rul": node.median_rul, "iqr": node.iqr,
                "split": False, "stop_reason": stop})
    return node


def terminals(node):
    if not node.children:
        yield node
    else:
        for c in node.children:
            yield from terminals(c)


def route(node, X, rows):
    """Assign held-out rows to terminal nodes; returns node_id per row."""
    out = np.full(len(rows), -1, dtype=int)
    if not node.children:
        out[:] = node.node_id
        return out
    lab = node.model.predict(X[rows])
    for c, child in enumerate(node.children):
        m = lab == c
        if m.any():
            out[m] = route(child, X, rows[m])
    return out


# --------------------------------------------------------------------------
def run(args):
    X = np.load(os.path.join(args.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(args.prepped, "metadata.csv"))
    tag = os.path.basename(args.prepped.rstrip("/"))
    tr = (md["split"] == "train").to_numpy()
    te = ~tr
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = ((X - mu) / sd).astype(np.float32)
    rul = md["rul"].to_numpy(dtype=float)
    print(f"{tag}: {tr.sum():,} train / {te.sum():,} held-out windows, "
          f"root = {args.method} at capacity {args.capacity}")

    # ---- root partition ---------------------------------------------------
    tr_idx = np.where(tr)[0]
    if args.method == "kmeans":
        root_model = KMeans(n_clusters=args.capacity, n_init=1,
                            random_state=args.seed).fit(Xs[tr_idx])
        root_lab = root_model.labels_
        def root_assign(rows):
            return root_model.predict(Xs[rows])
    else:
        side = side_of(args.capacity)
        som = fit_som(Xs[tr_idx], side, args.seed, args.iters)
        def bmu(rows):
            return np.array([np.ravel_multi_index(som.winner(x), (side, side))
                             for x in Xs[rows]])
        root_lab = bmu(tr_idx)
        root_assign = bmu

    # ---- grow one subtree per root cluster --------------------------------
    counter, log, forest = [0], [], {}
    for c in np.unique(root_lab):
        member = tr_idx[root_lab == c]
        if len(member) < args.min_node:
            continue
        forest[int(c)] = grow(Xs, rul, member, 0, args, counter, log)

    leaves = [t for tree in forest.values() for t in terminals(tree)]
    for lf in leaves:
        lf.at_risk = lf.median_rul <= args.risk_rul
    n_risk = sum(lf.at_risk for lf in leaves)
    depths = [lf.depth for lf in leaves]
    print(f"  tree: {len(forest)} root clusters -> {len(leaves)} terminal nodes, "
          f"depth {min(depths)}-{max(depths)} (mean {np.mean(depths):.2f})")
    print(f"  at-risk nodes (median training RUL <= {args.risk_rul}): "
          f"{n_risk} ({100*n_risk/len(leaves):.1f}%)")
    splits = sum(1 for e in log if e.get("split"))
    print(f"  splits performed: {splits}; "
          f"stop reasons: "
          f"{dict(pd.Series([e.get('stop_reason') for e in log if not e.get('split')]).value_counts())}")

    risk_ids = {lf.node_id for lf in leaves if lf.at_risk}

    # ---- route held-out windows ------------------------------------------
    te_idx = np.where(te)[0]
    root_te = root_assign(te_idx)
    node_of = np.full(len(te_idx), -1, dtype=int)
    for c, tree in forest.items():
        m = root_te == c
        if m.any():
            node_of[m] = route(tree, Xs, te_idx[m])

    ho = md.loc[te_idx].copy()
    ho["node"] = node_of
    ho["flagged"] = np.isin(node_of, list(risk_ids))
    unrouted = int((node_of == -1).sum())
    if unrouted:
        print(f"  note: {unrouted} held-out windows fell in root clusters with "
              f"no training subtree and are treated as unflagged")

    # ---- engine-level endpoints ------------------------------------------
    rows = []
    for unit, g in ho.groupby("unit"):
        g = g.sort_values("cycle_end")
        f = g[g["flagged"]]
        runs_to_failure = g["origin"].iloc[0] == "train"   # C-MAPSS train file
        rows.append({
            "unit": unit, "run_to_failure": runs_to_failure,
            "n_windows": len(g), "min_rul": float(g["rul"].min()),
            "flagged": bool(len(f)),
            "first_flag_cycle": int(f["cycle_end"].iloc[0]) if len(f) else -1,
            "lead_time": float(f["rul"].iloc[0]) if len(f) else np.nan,
        })
    eng = pd.DataFrame(rows)
    rtf = eng[eng.run_to_failure]

    det = rtf.flagged.mean() if len(rtf) else np.nan
    lead = rtf.loc[rtf.flagged, "lead_time"]
    prem = eng.loc[eng.flagged, "lead_time"] > CHANGE_POINT

    print(f"\nengine-level endpoints ({len(rtf)} held-out run-to-failure engines)")
    print(f"  detection rate            {100*det:5.1f}%")
    if len(lead):
        print(f"  lead time, median         {lead.median():6.1f} cycles")
        print(f"  lead time, IQR            {lead.quantile(.25):6.1f} - "
              f"{lead.quantile(.75):.1f} cycles")
        print(f"  lead time, max            {lead.max():6.1f} cycles")
        print(f"  flagged before change pt  {100*(lead > CHANGE_POINT).mean():5.1f}% "
              f"(RUL > {CHANGE_POINT})")
    else:
        print("  no engine was flagged; lead time undefined")
    print(f"  premature flags (all)     {100*prem.mean():5.1f}%" if len(prem)
          else "  premature flags: n/a")
    print(f"  missed (never flagged)    {100*(1-det):5.1f}%")

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"recursive_early_{tag}_{args.method}_"
                                  f"c{args.capacity}_d{args.max_depth}_s{args.seed}")
    eng.to_csv(stem + "_engines.csv", index=False)
    pd.DataFrame(log).to_csv(stem + "_tree.csv", index=False)
    summary = {"cohort": tag, "method": args.method, "capacity": args.capacity,
               "seed": args.seed, "n_terminal": len(leaves),
               "n_at_risk": n_risk, "mean_depth": float(np.mean(depths)),
               "detection_rate": float(det),
               "lead_median": float(lead.median()) if len(lead) else None,
               "lead_iqr_lo": float(lead.quantile(.25)) if len(lead) else None,
               "lead_iqr_hi": float(lead.quantile(.75)) if len(lead) else None,
               "frac_before_change_point":
                   float((lead > CHANGE_POINT).mean()) if len(lead) else None}
    with open(stem + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {stem}_engines.csv / _tree.csv / _summary.json")

    if len(lead) and (lead > CHANGE_POINT).mean() < 0.05:
        print("\nINTERPRETATION: almost no engine is flagged before the "
              f"{CHANGE_POINT}-cycle change point. That is consistent with the "
              "piecewise-linear convention: the sensors carry little "
              "degradation signal earlier, and the recursive refinement finds "
              "no pre-change-point risk structure. Report this as measured.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--method", choices=["kmeans", "som"], default="kmeans")
    p.add_argument("--capacity", type=int, default=100)
    p.add_argument("--branch", type=int, default=2, help="children per split")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-node", type=int, default=30,
                   help="minimum training windows in a node")
    p.add_argument("--rul-iqr-stop", type=float, default=40.0,
                   help="stop splitting when RUL IQR falls below this")
    p.add_argument("--min-gain", type=float, default=0.05,
                   help="minimum fractional reduction in within-node RUL variance")
    p.add_argument("--risk-rul", type=float, default=50.0,
                   help="terminal node is at-risk if median training RUL <= this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=50_000)
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
