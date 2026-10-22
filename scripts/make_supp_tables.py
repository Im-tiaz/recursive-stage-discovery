#!/usr/bin/env python3
"""
make_supp_tables.py
===================
Emit Supplementary Tables S1 and S2 as LaTeX, from the sweeps produced by
``run_supp_sensitivity.sh``.

S1  Sensitivity of the cohort contrasts to the near-failure threshold
    (10, 12, 15 and 20 cycles, window fixed at 30).
S2  Sensitivity to window length (20, 30 and 40 cycles, threshold fixed
    at 12).

Both tables report, per cohort and per setting: rare-state prevalence, the
held-out SOM-minus-K-means difference in near-failure effective coverage at
400 prototypes with a 95% interval, and the fault-mode contrast formed within
each bootstrap replicate.

The resampling here is identical to ``factorial_contrasts.py``: splits are
drawn with replacement within a cohort, then initialisations within each drawn
split, and the SOM-minus-K-means difference is formed within split and
initialisation before aggregation. Using the same scheme is what makes these
tables comparable with the main-text table rather than merely adjacent to it.

As in the main text, an interval containing zero is reported as a difference
not detected, never as evidence of equivalence.

    python make_supp_tables.py --results outputs/supp --staged data/supp \
        --capacity 400 --out paper
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# cohort tag -> (fault modes, operating conditions, design shorthand)
DESIGN = {
    "fd001_bal": (1, 1, "1c, 1m"),
    "fd002_bal": (1, 6, "6c, 1m"),
    "fd003_bal": (2, 1, "1c, 2m"),
    "fd004_bal": (2, 6, "6c, 2m"),
}
RARE = "imminent"
METRIC = "effcov_holdout_pooled"

# table -> ordered list of (column label, setting directory)
TABLES = {
    "S1": ("near-failure threshold", "cycles",
           [("10", "thr10"), ("12", "base"), ("15", "thr15"), ("20", "thr20")]),
    "S2": ("window length", "cycles",
           [("20", "win20"), ("30", "base"), ("40", "win40")]),
}


# ---------------------------------------------------------------- resampling
def paired(df: pd.DataFrame, metric: str) -> dict:
    """d[split] = array over seeds of (SOM - K-means) for one cohort."""
    out = {}
    for split, g in df.groupby("split"):
        s = g[g.method == "som"].set_index("seed")[metric]
        k = g[g.method == "kmeans"].set_index("seed")[metric]
        common = s.index.intersection(k.index)
        if len(common):
            out[split] = (s.loc[common] - k.loc[common]).to_numpy()
    return out


def draw(d: dict, rng) -> float:
    """One hierarchical bootstrap replicate of a cohort's mean difference."""
    splits = list(d)
    chosen = rng.choice(len(splits), len(splits), replace=True)
    return float(np.mean([rng.choice(d[splits[c]], len(d[splits[c]])).mean()
                          for c in chosen]))


def ci(v, lo=2.5, hi=97.5):
    return np.percentile(v, [lo, hi])


# ------------------------------------------------------------------ loading
def prevalence(staged: str, setting: str, tag: str) -> float | None:
    """Rare-state share of all windows in a staged cohort."""
    f = os.path.join(staged, setting, tag, "metadata.csv")
    if not os.path.exists(f):
        return None
    md = pd.read_csv(f, usecols=["health_state"])
    return float((md["health_state"] == RARE).mean())


def n_windows(staged: str, setting: str, tag: str) -> int | None:
    f = os.path.join(staged, setting, tag, "metadata.csv")
    if not os.path.exists(f):
        return None
    return int(len(pd.read_csv(f, usecols=["health_state"])))


def load_setting(results: str, setting: str, capacity: int):
    """{tag: paired-difference dict} for one setting, or None if incomplete."""
    cells = {}
    for tag in DESIGN:
        f = os.path.join(results, setting, f"holdout_sweep_{tag}.csv")
        if not os.path.exists(f):
            print(f"  MISSING {f}", file=sys.stderr)
            return None
        df = pd.read_csv(f)
        df = df[df.capacity == capacity]
        if df.empty:
            print(f"  EMPTY at capacity {capacity}: {f}", file=sys.stderr)
            return None
        cells[tag] = paired(df, METRIC)
    return cells


# ------------------------------------------------------------------ writing
def fmt(est: float, lo: float, hi: float) -> str:
    star = "^{*}" if (lo > 0 or hi < 0) else ""
    return f"${est:+.3f}{star}$ & $[{lo:+.3f}, {hi:+.3f}]$"


def build(which: str, args, rng) -> str | None:
    varied, unit, columns = TABLES[which]
    fixed = ("window length fixed at 30 cycles" if which == "S1"
             else "near-failure threshold fixed at 12 cycles")

    rows, missing = [], []
    for label, setting in columns:
        cells = load_setting(args.results, setting, args.capacity)
        if cells is None:
            missing.append(f"{varied} = {label}")
            continue
        reps = {t: np.array([draw(cells[t], rng) for _ in range(args.n_boot)])
                for t in DESIGN}
        fm = (reps["fd003_bal"] + reps["fd004_bal"]) / 2 \
           - (reps["fd001_bal"] + reps["fd002_bal"]) / 2

        for tag, (_, _, shorthand) in DESIGN.items():
            obs = float(np.mean([v.mean() for v in cells[tag].values()]))
            lo, hi = ci(reps[tag])
            prev = prevalence(args.staged, setting, tag)
            nwin = n_windows(args.staged, setting, tag)
            rows.append({
                "setting": label, "cohort": tag[:5].upper(),
                "design": shorthand,
                "windows": nwin, "prev": prev,
                "cell": fmt(obs, lo, hi),
            })
        lo, hi = ci(fm)
        rows.append({"setting": label, "cohort": r"\textit{Fault-mode contrast}",
                     "design": "", "windows": None, "prev": None,
                     "cell": fmt(float(fm.mean()), lo, hi)})

    if not rows:
        print(f"{which}: no settings available, nothing written",
              file=sys.stderr)
        return None
    if missing:
        print(f"{which}: WARNING, incomplete -- missing {', '.join(missing)}",
              file=sys.stderr)

    show_n = which == "S2"
    if show_n:
        ncol = "lllrrrl"
        head = (r"Window & Cohort & Design & Windows & Prevalence & "
                r"Difference & 95\% CI \\")
    else:
        ncol = "lllrrl"
        head = (r"Threshold & Cohort & Design & Prevalence & "
                r"Difference & 95\% CI \\")

    cap = {
        "S1": ("Sensitivity of the cohort contrasts to the near-failure "
               "threshold. Difference is SOM minus $K$-means held-out "
               "near-failure effective coverage at 400 prototypes."),
        "S2": ("Sensitivity of the cohort contrasts to window length. "
               "Difference is SOM minus $K$-means held-out near-failure "
               "effective coverage at 400 prototypes."),
    }[which]

    out = [r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{{cap}}}",
           rf"\label{{tab:{which}}}",
           rf"\begin{{tabular}}{{{ncol}}}", r"\toprule", head, r"\midrule"]

    last = None
    for r in rows:
        if last is not None and r["setting"] != last:
            out.append(r"\addlinespace")
        cells_ = [r["setting"] if r["setting"] != last else "",
                  r["cohort"], r["design"]]
        if show_n:
            cells_.append("" if r["windows"] is None
                          else f"{r['windows']:,}".replace(",", "{,}"))
        cells_.append("" if r["prev"] is None else f"{100*r['prev']:.1f}\\%")
        out.append(" & ".join(cells_) + " & " + r["cell"] + r" \\")
        last = r["setting"]

    out += [r"\bottomrule", r"\end{tabular}",
            r"",
            r"\vspace{4pt}",
            r"\begin{minipage}{0.95\linewidth}\footnotesize",
            rf"Each cell is five engine splits with ten initialisations "
            rf"per split, {fixed}. Differences are paired within split and "
            r"initialisation; intervals are 95\% hierarchical bootstrap "
            r"percentiles obtained by resampling splits and then "
            r"initialisations. An asterisk marks an interval excluding zero. "
            r"An interval containing zero is a difference not detected, not "
            r"evidence of equivalence.",
            r"\end{minipage}",
            r"\end{table}"]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="outputs/supp")
    p.add_argument("--staged", default="data/supp")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--out", default="paper")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    wrote = []
    for which in ("S1", "S2"):
        tex = build(which, args, rng)
        if tex is None:
            continue
        f = os.path.join(args.out, f"supp_{which}.tex")
        with open(f, "w") as fh:
            fh.write(tex + "\n")
        wrote.append(f)
        print(f"wrote {f}")

    if not wrote:
        print("\nNothing written. Run './run_supp_sensitivity.sh stage' and "
              "then 'sweep' first.", file=sys.stderr)
        return 1
    print("\nPaste each file into supplementary.tex where its placeholder is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
