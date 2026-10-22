#!/usr/bin/env python3
"""
04c_supervised_warning_baselines.py
===================================
Compares the recursive stage index against supervised remaining-useful-life
regression, on identical folds and at identical operating points.

WHY THIS IS NEEDED

C-MAPSS has a large published literature on supervised RUL regression. Any
reviewer will ask why a partition-based procedure should be preferred to
regressing remaining life directly and thresholding the prediction. The
appendix currently compares the recursive tree against a root-only partition
and a distance-to-centroid rule, both of which are weaker comparators than the
obvious one. Without a regression baseline the comparison is not informative.

Four alert rules, scored on the same engine-level folds:

    recursive   the stage index of Appendix A: recursive refinement of a
                K-means root partition, terminal nodes ranked by median
                training RUL, alert on first arrival in an at-risk node
    flat        the same, with refinement disabled
    ridge       ridge regression on RUL; alert when the predicted RUL for a
                window first falls at or below the operating point
    forest      random-forest regression on RUL, same alert rule

The regressors are given exactly what the tree is given: the same standardised
features, the same training engines, the same persistence requirement, and the
same operating points. They see the same RUL labels the tree's stopping rules
and node scores consume, so the comparison is between two supervised procedures
and not between a supervised and an unsupervised one.

OPERATING POINT

Reporting a threshold chosen after seeing held-out results is selection on the
evaluation set. This script selects one operating point per fold from the
TRAINING engines alone, by a rule stated in advance: the largest rho whose
training premature-warning rate stays at or below --max-premature. The selected
value is reported alongside the full curve.

    python 04c_supervised_warning_baselines.py --prepped data/fd001_bal
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_rec = import_module("04b_early_warning_eval")

RHO_GRID = [30, 50, 75, 100, 125]
CHANGE_POINT = 125


def engine_folds(units, k, seed):
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return [set(x) for x in np.array_split(u, k)]


def first_persistent(flags, k):
    run = 0
    for i, v in enumerate(flags):
        run = run + 1 if v else 0
        if run >= k:
            return i - k + 1
    return -1


def engine_rows(md, idx, flag, tag, strat, rho, fold, persist):
    """One row per engine: did it alert, and with how much life remaining."""
    sub = md.loc[idx].copy()
    sub["f"] = flag
    out = []
    for unit, g in sub.groupby("unit"):
        g = g.sort_values("cycle_end")
        i = first_persistent(g["f"].to_numpy(), persist)
        out.append({"cohort": tag, "strategy": strat, "rho": rho, "fold": fold,
                    "unit": unit,
                    "run_to_failure": g["origin"].iloc[0] == "train",
                    "flagged": i >= 0,
                    "lead": float(g["rul"].iloc[i]) if i >= 0 else np.nan})
    return out


def endpoints(d, rho, horizon=None):
    H = rho if horizon is None else horizon
    rtf = d[d.run_to_failure]
    det = float(rtf.flagged.mean()) if len(rtf) else np.nan
    lead = rtf.loc[rtf.flagged, "lead"]
    alerted = d[d.flagged]
    prem = float((alerted.lead > H).mean()) if len(alerted) else np.nan
    return {"detection": det, "missed": (1 - det) if det == det else np.nan,
            "lead_median": float(lead.median()) if len(lead) else np.nan,
            "lead_q1": float(lead.quantile(.25)) if len(lead) else np.nan,
            "lead_q3": float(lead.quantile(.75)) if len(lead) else np.nan,
            "premature": prem, "n_engines": int(len(rtf))}


def run(a):
    X = np.load(os.path.join(a.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(a.prepped, "metadata.csv"))
    units = md["unit"].to_numpy()
    rul = md["rul"].to_numpy(dtype=float)
    tag = os.path.basename(a.prepped.rstrip("/"))
    folds = engine_folds(units, a.folds, a.seed)
    print(f"{tag}: {len(md):,} windows, {len(set(units))} engines, "
          f"{a.folds}-fold over engines, persistence {a.persist}")

    rows, chosen = [], []
    for fi, held in enumerate(folds):
        te = np.isin(units, list(held)); tr = ~te
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X).astype(np.float64)
        tr_i, te_i = np.where(tr)[0], np.where(te)[0]

        # ---- partition arms, via the existing implementation -------------
        km_lab = None
        from sklearn.cluster import KMeans
        m = KMeans(a.capacity, n_init=5, random_state=a.seed).fit(Xs[tr_i])
        root_tr, root_te = m.labels_, m.predict(Xs[te_i])
        for depth, strat in ((a.max_depth, "recursive"), (0, "flat")):
            aa = argparse.Namespace(**{**vars(a), "max_depth": depth})
            counter, log, forest = [0], [], {}
            for c in np.unique(root_tr):
                mem = tr_i[root_tr == c]
                if len(mem) >= a.min_node:
                    forest[int(c)] = _rec.rec.grow(Xs, rul, mem, 0, aa,
                                                   counter, log)
            leaves = [l for t_ in forest.values() for l in _rec.rec.terminals(t_)]
            node = np.full(len(te_i), -1, dtype=int)
            for c, tree in forest.items():
                s = root_te == c
                if s.any():
                    node[s] = _rec.rec.route(tree, Xs, te_i[s])
            for rho in RHO_GRID:
                risk = {l.node_id for l in leaves if l.median_rul <= rho}
                rows += engine_rows(md, te_i, np.isin(node, list(risk)),
                                    tag, strat, rho, fi, a.persist)

        # ---- supervised regression arms ---------------------------------
        models = {
            "ridge": RidgeCV(alphas=np.logspace(-1, 4, 20)),
            "forest": RandomForestRegressor(n_estimators=a.trees,
                                            min_samples_leaf=5,
                                            n_jobs=-1, random_state=a.seed),
        }
        for name, mdl in models.items():
            mdl.fit(Xs[tr_i], rul[tr_i])
            pred_te = mdl.predict(Xs[te_i])
            pred_tr = mdl.predict(Xs[tr_i])
            for rho in RHO_GRID:
                rows += engine_rows(md, te_i, pred_te <= rho,
                                    tag, name, rho, fi, a.persist)
            # operating point chosen on TRAINING engines only
            best = None
            for rho in RHO_GRID:
                trr = pd.DataFrame(engine_rows(md, tr_i, pred_tr <= rho,
                                               tag, name, rho, fi, a.persist))
                e = endpoints(trr, rho)
                if e["premature"] == e["premature"] and \
                        e["premature"] <= a.max_premature:
                    best = rho
            chosen.append({"cohort": tag, "fold": fi, "strategy": name,
                           "rho_selected": best})
        print(f"  fold {fi}: {tr.sum():,} train / {te.sum():,} held-out windows",
              flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"warning_baselines_{tag}_p{a.persist}")
    df.to_csv(stem + "_engines.csv", index=False)

    summ = []
    for (strat, rho), g in df.groupby(["strategy", "rho"]):
        summ.append({"strategy": strat, "rho": rho, **endpoints(g, rho)})
    s = pd.DataFrame(summ).sort_values(["strategy", "rho"])
    s.to_csv(stem + "_summary.csv", index=False)

    print(f"\n{'strategy':<11}{'rho':>5}{'detect':>8}{'lead':>8}{'IQR':>13}"
          f"{'premature':>11}")
    for _, r in s.iterrows():
        iqr = (f"{r.lead_q1:.0f}-{r.lead_q3:.0f}"
               if r.lead_q1 == r.lead_q1 else "--")
        lm = f"{r.lead_median:.1f}" if r.lead_median == r.lead_median else "--"
        print(f"{r.strategy:<11}{int(r.rho):>5}{r.detection:>8.3f}{lm:>8}"
              f"{iqr:>13}{r.premature:>11.3f}")

    if chosen:
        c = pd.DataFrame(chosen)
        c.to_csv(stem + "_selected.csv", index=False)
        print(f"\noperating point selected on training engines "
              f"(premature <= {a.max_premature})")
        for strat, g in c.groupby("strategy"):
            v = g.rho_selected.dropna()
            print(f"  {strat:<9} per fold {list(g.rho_selected)}"
                  + (f", modal {int(v.mode().iloc[0])}" if len(v) else ""))

    print(f"\nwrote {stem}_summary.csv")
    print("\nThe regressors receive the same labels, features, folds and")
    print("persistence rule as the tree. A tree that does not beat them on")
    print("lead time at matched premature rate has no advantage to report,")
    print("and saying so is more useful than omitting the comparison.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--capacity", type=int, default=16)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--branch", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--min-node", type=int, default=15)
    p.add_argument("--rul-iqr-stop", type=float, default=40.0)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--persist", type=int, default=2)
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--max-premature", type=float, default=0.10,
                   help="training premature-warning rate the selected "
                        "operating point must not exceed")
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
