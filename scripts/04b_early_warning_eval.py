#!/usr/bin/env python3
"""
04b_early_warning_eval.py
=========================
Rewrites the early-warning evaluation to remove the four defects the review
identified in the first version.

    pseudo-replication      engines were pooled across five random held-out
                            sets, so a cohort of 100 run-to-failure engines
                            produced 126 observations and some engines were
                            counted more than once. Here the engines are
                            partitioned into K folds, each engine is held out
                            exactly once, and every engine contributes one
                            out-of-fold prediction.

    post-hoc operating point  a single rho was reported after inspecting the
                            FD001 curve. Here the entire curve is reported for
                            every cohort, and no operating point is selected.

    no baseline             recursive refinement was reported without anything
                            to beat. Two baselines are added: the root
                            partition without recursion, and a distance rule
                            that flags a window when its distance to the
                            training terminal centroid falls below a threshold
                            calibrated on training engines to raise the same
                            share of alarms.

    incomplete endpoints    only lead time was tabulated across cohorts. All
                            four endpoints are reported at every operating
                            point: detection rate, lead time, premature-flag
                            rate and missed rate.

The procedure remains supervised in its risk scoring, since remaining useful
life defines the at-risk nodes, and that is stated in the manuscript.

    python 04b_early_warning_eval.py --prepped data/fd001_bal
    python 04b_early_warning_eval.py --prepped data/fd003_bal --method som
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "rec", os.path.join(HERE, "04_recursive_early_prediction.py"))
rec = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rec)

sys.path.insert(0, HERE)
from _common import fit_som, side_of  # noqa: E402

CHANGE_POINT = 125
RHO_GRID = [30, 50, 75, 100, 125]


def engine_folds(units, k, seed):
    """Partition engines into k folds so each is held out exactly once."""
    u = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return [set(x) for x in np.array_split(u, k)]


def build_tree(Xs, rul, tr_idx, root_lab, args):
    counter, log, forest = [0], [], {}
    for c in np.unique(root_lab):
        member = tr_idx[root_lab == c]
        if len(member) < args.min_node:
            continue
        forest[int(c)] = rec.grow(Xs, rul, member, 0, args, counter, log)
    return forest, log


def endpoints(eng, rho_label, horizon=None):
    """Engine-level endpoints from one out-of-fold prediction per engine.

    A premature warning is an alert raised while the engine still has more
    remaining useful life than the operating horizon. It is NOT simply any
    alert on a truncated engine: truncated trajectories stop before failure but
    are not therefore healthy, and some of them do enter the severe health
    state before their record ends. Judging them by their actual remaining
    useful life credits a warning raised close to failure instead of counting
    it as a false positive.

    The horizon defaults to the operating point rho, so an alert counts as
    premature when it fires further from failure than the threshold that
    defined the at-risk nodes.
    """
    H = rho_label if horizon is None else horizon
    rtf = eng[eng.run_to_failure]
    det = float(rtf.flagged.mean()) if len(rtf) else np.nan
    lead = rtf.loc[rtf.flagged, "lead_time"]
    prem = eng.loc[eng.flagged, "lead_time"] > H
    return {"rho": rho_label, "n_engines": int(len(rtf)),
            "detection": det, "missed": 1 - det if det == det else np.nan,
            "lead_median": float(lead.median()) if len(lead) else np.nan,
            "lead_q1": float(lead.quantile(.25)) if len(lead) else np.nan,
            "lead_q3": float(lead.quantile(.75)) if len(lead) else np.nan,
            "premature": float(prem.mean()) if len(prem) else np.nan}


def run(args):
    global RHO_GRID
    if args.rho_grid:
        RHO_GRID = list(args.rho_grid)
    X = np.load(os.path.join(args.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(args.prepped, "metadata.csv"))
    tag = os.path.basename(args.prepped.rstrip("/"))
    units = md["unit"].to_numpy()
    rul = md["rul"].to_numpy(dtype=float)
    folds = engine_folds(units, args.folds, args.seed)
    print(f"{tag}: {len(md):,} windows, {len(set(units))} engines, "
          f"{args.folds}-fold, root {args.method} at capacity {args.capacity}")

    # one row per engine per strategy per rho; each engine appears once per fold
    recs, stage_rows = [], []
    for fi, held in enumerate(folds):
        te = np.isin(units, list(held)); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = ((X - mu) / sd).astype(np.float32)
        tr_idx = np.where(tr)[0]
        te_idx = np.where(te)[0]

        if args.method == "kmeans":
            m = KMeans(n_clusters=args.capacity, n_init=1,
                       random_state=args.seed).fit(Xs[tr_idx])
            root_tr, root_te = m.labels_, m.predict(Xs[te_idx])
        else:
            side = side_of(args.capacity)
            som = fit_som(Xs[tr_idx], side, args.seed, args.iters)
            def bmu(rows):
                return np.array([np.ravel_multi_index(som.winner(x), (side, side))
                                 for x in Xs[rows]])
            root_tr, root_te = bmu(tr_idx), bmu(te_idx)

        # -- strategy A: recursive tree; strategy B: root only (depth 0) ----
        for depth, name in ((args.max_depth, "recursive"), (0, "flat")):
            a2 = argparse.Namespace(**{**vars(args), "max_depth": depth})
            forest, _ = build_tree(Xs, rul, tr_idx, root_tr, a2)
            leaves = [l for t_ in forest.values() for l in rec.terminals(t_)]
            node_of = np.full(len(te_idx), -1, dtype=int)
            for c, tree in forest.items():
                sel = root_te == c
                if sel.any():
                    node_of[sel] = rec.route(tree, Xs, te_idx[sel])
            nmap, K = stage_index(leaves)
            stage_rows.extend(collect_stage(md, te_idx, node_of, nmap, K,
                                            tag, name, fi))
            for rho in RHO_GRID:
                risk = {l.node_id for l in leaves if l.median_rul <= rho}
                flag = np.isin(node_of, list(risk))
                recs += collect(md, te_idx, flag, tag, name, rho, fi, args.persist)

        # -- strategy C: distance to the training terminal centroid ---------
        imm = tr_idx[md.loc[tr_idx, "health_state"].to_numpy() == "imminent"]
        if len(imm):
            cen = Xs[imm].mean(0)
            d_tr = np.linalg.norm(Xs[tr_idx] - cen, axis=1)
            d_te = np.linalg.norm(Xs[te_idx] - cen, axis=1)
            for rho in RHO_GRID:
                # match the alarm budget: flag the same share of training
                # windows that the recursive tree would flag at this rho
                share = float((rul[tr_idx] <= rho).mean())
                thr = np.quantile(d_tr, share)
                recs += collect(md, te_idx, d_te <= thr, tag, "centroid", rho, fi,
                                args.persist)

        print(f"  fold {fi}: {tr.sum():,} train / {te.sum():,} held-out windows",
              flush=True)

    df = pd.DataFrame(recs)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out,
                        f"early_oof_{tag}_{args.method}"
                        f"_p{args.persist}")
    df.to_csv(stem + "_engines.csv", index=False)

    if stage_rows:
        sd = pd.DataFrame(stage_rows)
        sd.drop(columns=["first_arrival"]).to_csv(stem + "_stage.csv", index=False)
        rtf_stage = sd[(sd.strategy == "recursive") & sd.run_to_failure]
        if len(rtf_stage):
            # Pair every (stage reached, remaining life on arrival) observation.
            # Correlating the final stage against the final RUL is uninformative
            # because every run-to-failure engine ends near zero.
            pairs = [(k_, v_) for _, r_ in rtf_stage.iterrows()
                     for k_, v_ in r_.first_arrival.items()]
            pr = pd.DataFrame(pairs, columns=["N", "rul"])
            pr.to_csv(stem + "_stage_arrivals.csv", index=False)
            K = int(rtf_stage.K.iloc[0])
            if pr.N.nunique() > 1:
                from scipy.stats import spearmanr
                rho_s, p_s = spearmanr(pr.N, pr.rul)
                print(f"\nstage index N: {K} stages, "
                      f"Spearman(N, remaining life on arrival) = {rho_s:+.3f} "
                      f"(p={p_s:.1e}, n={len(pr)})")
                q = pd.qcut(pr.N, min(5, pr.N.nunique()), duplicates="drop")
                g = pr.groupby(q, observed=True).agg(
                    n=("rul", "size"), median_rul=("rul", "median"))
                print(f"{'stage band':>16}{'arrivals':>10}{'median RUL':>13}")
                for idx, r_ in g.iterrows():
                    print(f"{str(idx):>16}{int(r_.n):>10}{r_.median_rul:>13.1f}")
                print("A monotone fall in remaining life as N rises means the "
                      "index tracks event maturity.")

    rows = []
    for (strat, rho), g in df.groupby(["strategy", "rho"]):
        e = endpoints(g, rho, args.horizon)
        e["strategy"] = strat
        rows.append(e)
    summ = pd.DataFrame(rows).sort_values(["strategy", "rho"])
    summ.to_csv(stem + "_summary.csv", index=False)

    print(f"\nout-of-fold endpoints, one prediction per engine")
    print(f"{'strategy':<11}{'rho':>5}{'n':>5}{'detect':>8}{'lead med':>10}"
          f"{'IQR':>14}{'premature':>11}")
    for _, r in summ.iterrows():
        iqr = (f"{r.lead_q1:.0f}-{r.lead_q3:.0f}"
               if r.lead_q1 == r.lead_q1 else "--")
        lm = f"{r.lead_median:.1f}" if r.lead_median == r.lead_median else "--"
        print(f"{r.strategy:<11}{int(r.rho):>5}{r.n_engines:>5}"
              f"{r.detection:>8.3f}{lm:>10}{iqr:>14}{r.premature:>11.3f}")
    print(f"\nwrote {stem}_summary.csv")
    print("\nNo operating point is selected. The curve is the result.")


def _first_persistent(flags, k):
    """Index of the first window starting a run of k consecutive alerts."""
    if k <= 1:
        nz = np.flatnonzero(flags)
        return int(nz[0]) if len(nz) else -1
    run = 0
    for i, v in enumerate(flags):
        run = run + 1 if v else 0
        if run >= k:
            return i - k + 1
    return -1


def stage_index(leaves):
    """Order terminal nodes into a maturity index N.

    Nodes are ranked by the median remaining useful life of their training
    windows, descending, so N = 1 is the stage furthest from failure and N = K
    the stage closest to it. The index is the operational read-out: a low N on
    an unseen engine says the event is still far off, a high N that it is near.
    """
    order = sorted(leaves, key=lambda l: -l.median_rul)
    return {l.node_id: i + 1 for i, l in enumerate(order)}, len(order)


def collect_stage(md, te_idx, node_of, nmap, K, tag, strategy, fold):
    """Per engine: the highest stage index reached and the RUL at each arrival."""
    sub = md.loc[te_idx].copy()
    sub["N"] = [nmap.get(x, -1) for x in node_of]
    out = []
    for unit, g in sub.groupby("unit"):
        g = g[g.N > 0].sort_values("cycle_end")
        if not len(g):
            continue
        first = {}
        run_max = 0
        for _, r in g.iterrows():
            if r.N > run_max:
                run_max = int(r.N)
                first[run_max] = float(r.rul)
        out.append({"cohort": tag, "strategy": strategy, "fold": fold,
                    "unit": unit, "K": K,
                    "run_to_failure": g["origin"].iloc[0] == "train",
                    "N_max": run_max, "N_final": int(g.N.iloc[-1]),
                    "rul_final": float(g.rul.iloc[-1]),
                    "first_arrival": first})
    return out


def collect(md, te_idx, flag, tag, strategy, rho, fold, persist=1):
    """One row per held-out engine.

    Windows overlap by 25 of 30 cycles, so a single at-risk window is a noisy
    trigger. `persist` requires that many consecutive at-risk windows before an
    alert is counted, and the lead time is taken at the first window of the
    run. persist=1 reproduces the single-window rule.
    """
    sub = md.loc[te_idx].copy()
    sub["flagged_window"] = flag
    out = []
    for unit, g in sub.groupby("unit"):
        g = g.sort_values("cycle_end")
        i = _first_persistent(g["flagged_window"].to_numpy(), persist)
        out.append({"cohort": tag, "strategy": strategy, "rho": rho,
                    "fold": fold, "unit": unit, "persist": persist,
                    "run_to_failure": g["origin"].iloc[0] == "train",
                    "flagged": i >= 0,
                    "lead_time": float(g["rul"].iloc[i]) if i >= 0 else np.nan})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/metrics")
    p.add_argument("--method", choices=["kmeans", "som"], default="kmeans")
    p.add_argument("--capacity", type=int, default=16)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--branch", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--min-node", type=int, default=15)
    p.add_argument("--rul-iqr-stop", type=float, default=40.0)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--horizon", type=float, default=None,
                   help="remaining-useful-life horizon beyond which an "
                        "alert counts as premature; defaults to rho")
    p.add_argument("--rho-grid", type=int, nargs="+", default=None,
                   help="operating points; defaults to 30 50 75 100 125")
    p.add_argument("--persist", type=int, default=1,
                   help="consecutive at-risk windows required "
                        "before an alert counts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=50_000)
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
