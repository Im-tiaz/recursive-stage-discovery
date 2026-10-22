#!/usr/bin/env python3
"""
cmapss_prep.py
==============
Turn NASA C-MAPSS run-to-failure sensor streams into an N x D feature matrix
with health-state labels, shaped for capacity-matched clustering analysis.

Data source
-----------
NASA PCoE "Turbofan Engine Degradation Simulation Data Set" (Saxena & Goebel,
2008). Expects the unzipped .txt files in one directory:

    train_FD001.txt  test_FD001.txt  RUL_FD001.txt   (and FD002-FD004)

Each row: unit, cycle, op_setting_1..3, sensor_1..21  (space separated).

What this produces
------------------
    X.npy         float32, (n_windows, n_features)
    features.npy  feature names, aligned to columns of X
    windows.npy   window ids "unit@cycle_end", aligned to rows of X
    metadata.csv  window_id, unit, subset, cycle_end, rul, health_state, split

`meta` keeps `unit` and `cycle_end` so you can (a) draw a single engine's
trajectory across a fitted SOM and (b) run the snapshot-vs-trajectory
experiment by sampling one window per unit.

Design notes that matter for the paper
--------------------------------------
* Splits are GROUPED BY ENGINE UNIT. Windows from one engine overlap heavily;
  splitting at the window level leaks and inflates every metric. This is the
  single most common way a C-MAPSS result gets torn apart in review.
* FD002/FD004 have six operating regimes. Raw sensor values are dominated by
  regime, not by degradation. `--conditions 6` clusters the op-setting triple
  and z-scores each sensor within its regime. Skip this and your clusters will
  recover flight conditions instead of engine health.
* Health states come from RUL thresholds and are deliberately imbalanced:
  near-failure windows are the rare critical minority class.

Usage
-----
    python cmapss_prep.py --self-test              # no data needed, verifies pipeline
    python cmapss_prep.py --data-dir ./CMAPSSData --subsets FD001 --out ./prepped
    python cmapss_prep.py --data-dir ./CMAPSSData --subsets FD002 --conditions 6 \
        --window 30 --stride 5 --pairwise --out ./prepped_fd002
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
COLUMNS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS

# RUL cut points -> health states. Descending severity, deliberately imbalanced.
DEFAULT_BINS = [20, 50, 100]
DEFAULT_LABELS = ["imminent", "critical", "degrading", "healthy"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_subset(data_dir: str, subset: str) -> pd.DataFrame:
    """Load train + test for one subset, attach true RUL to every row."""
    frames = []

    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    train = _read_raw(train_path)
    # Training trajectories run to failure: RUL = cycles remaining until last.
    last = train.groupby("unit")["cycle"].transform("max")
    train["rul"] = last - train["cycle"]
    train["origin"] = "train"
    frames.append(train)

    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")
    if os.path.exists(test_path) and os.path.exists(rul_path):
        test = _read_raw(test_path)
        # Test trajectories are truncated; RUL_*.txt gives RUL at the final cycle.
        truth = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
        units = np.sort(test["unit"].unique())
        if len(truth) != len(units):
            raise ValueError(
                f"{subset}: RUL file has {len(truth)} rows but test set has "
                f"{len(units)} units."
            )
        truth_map = dict(zip(units, truth))
        last_t = test.groupby("unit")["cycle"].transform("max")
        test["rul"] = (last_t - test["cycle"]) + test["unit"].map(truth_map)
        test["origin"] = "test"
        # Offset unit ids so train and test engines never collide.
        test["unit"] = test["unit"] + 10_000
        frames.append(test)
    else:
        warnings.warn(f"{subset}: no test/RUL files found, using train only.")

    df = pd.concat(frames, ignore_index=True)
    df["subset"] = subset
    # Namespace units across subsets so multi-subset runs stay groupable.
    df["unit"] = subset + "_" + df["unit"].astype(str)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def _read_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Some distributions carry two trailing all-NaN columns from the delimiter.
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path}: expected {len(COLUMNS)} columns, found {df.shape[1]}."
        )
    df.columns = COLUMNS
    return df


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def normalize_by_condition(df: pd.DataFrame, n_conditions: int) -> pd.DataFrame:
    """Z-score each sensor within its operating regime.

    With n_conditions == 1 this is a plain global z-score. With 6 (FD002/FD004)
    the op-setting triple is clustered first, because regime dominates the raw
    signal and will otherwise be what any clustering method recovers.
    """
    df = df.copy()
    if n_conditions > 1:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_conditions, n_init=10, random_state=0)
        df["regime"] = km.fit_predict(df[OP_COLS].to_numpy())
    else:
        df["regime"] = 0

    for col in SENSOR_COLS:
        grp = df.groupby("regime")[col]
        mu, sd = grp.transform("mean"), grp.transform("std")
        df[col] = np.where(sd > 1e-8, (df[col] - mu) / sd, 0.0)
    return df


def subsample_engines(df: pd.DataFrame, max_engines: int,
                      seed: int) -> pd.DataFrame:
    """Keep a random subset of engines, balanced across origin.

    FD002 and FD004 ship roughly 2.5x as many engines as FD001 and FD003. In a
    factorial comparison that confounds operating-condition count with training
    set size, so cohorts are equalised here. Sampling is stratified by origin
    (run-to-failure vs truncated) because the terminal-geometry analysis uses
    run-to-failure engines only, and an unstratified draw would vary their
    number between cohorts.
    """
    units = df[["unit", "origin"]].drop_duplicates()
    if len(units) <= max_engines:
        print(f"  {len(units)} engines <= --max-engines {max_engines}, keeping all")
        return df
    rng = np.random.default_rng(seed)
    keep = []
    groups = sorted(units["origin"].unique())
    per = max_engines // len(groups)
    for i, g in enumerate(groups):
        pool = units.loc[units["origin"] == g, "unit"].to_numpy()
        n = per + (max_engines - per * len(groups) if i == 0 else 0)
        n = min(n, len(pool))
        keep.extend(rng.choice(pool, n, replace=False))
    print(f"  subsampled {len(units)} -> {len(keep)} engines "
          f"(stratified by origin, seed {seed})")
    return df[df["unit"].isin(set(keep))].reset_index(drop=True)


def drop_flat_sensors(df: pd.DataFrame, tol: float = 1e-6,
                      force: list[str] | None = None) -> list[str]:
    """Return sensors that actually vary. FD001 has six that never move.

    `force` overrides the variance screen with an explicit channel list. Use it
    to hold the feature space identical across cohorts: the screened set
    differs between subsets (sensor_10 varies in FD003 but not FD001), so a
    cross-cohort comparison of geometry is otherwise confounded by feature
    dimensionality.
    """
    if force:
        missing = sorted(set(force) - set(SENSOR_COLS))
        if missing:
            raise ValueError(f"unknown sensors: {missing}")
        print(f"  forcing {len(force)} sensors (variance screen bypassed)")
        return list(force)
    keep = [c for c in SENSOR_COLS if df[c].std(skipna=True) > tol]
    dropped = sorted(set(SENSOR_COLS) - set(keep))
    if dropped:
        print(f"  dropping {len(dropped)} constant sensors: {', '.join(dropped)}")
    return keep


# --------------------------------------------------------------------------
# Windowing and features
# --------------------------------------------------------------------------
def _skew(x: np.ndarray) -> float:
    sd = x.std()
    return 0.0 if sd < 1e-12 else float((((x - x.mean()) / sd) ** 3).mean())


def _kurt(x: np.ndarray) -> float:
    sd = x.std()
    return 0.0 if sd < 1e-12 else float((((x - x.mean()) / sd) ** 4).mean() - 3.0)


def _autocorr1(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    a, b = x[:-1], x[1:]
    da, db = a - a.mean(), b - b.mean()
    denom = np.sqrt((da**2).sum() * (db**2).sum())
    return 0.0 if denom < 1e-12 else float((da * db).sum() / denom)


CHANNEL_STATS = [
    "mean", "std", "min", "max", "range", "slope",
    "last", "delta", "rms", "mad", "skew", "kurt", "ac1",
]
N_BANDS = 4


def _channel_features(x: np.ndarray, t: np.ndarray) -> list[float]:
    """13 time-domain statistics + N_BANDS spectral band energies."""
    slope = float(np.polyfit(t, x, 1)[0]) if len(x) > 1 else 0.0
    feats = [
        float(x.mean()), float(x.std()), float(x.min()), float(x.max()),
        float(x.max() - x.min()), slope,
        float(x[-1]), float(x[-1] - x[0]),
        float(np.sqrt((x**2).mean())),
        float(np.abs(np.diff(x)).mean()) if len(x) > 1 else 0.0,
        _skew(x), _kurt(x), _autocorr1(x),
    ]
    # Spectral energy in N_BANDS equal slices of the one-sided spectrum.
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    if len(spec) > 1:
        spec = spec[1:]  # drop DC (already zero after centring)
        for band in np.array_split(spec, N_BANDS):
            feats.append(float(band.sum()) if band.size else 0.0)
    else:
        feats.extend([0.0] * N_BANDS)
    return feats


def build_feature_names(channels: list[str], pairwise: bool) -> list[str]:
    names = []
    for ch in channels:
        names += [f"{ch}__{s}" for s in CHANNEL_STATS]
        names += [f"{ch}__band{b}" for b in range(N_BANDS)]
    if pairwise:
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                names.append(f"corr__{channels[i]}__{channels[j]}")
    return names


def extract_windows(
    df: pd.DataFrame,
    channels: list[str],
    window: int,
    stride: int,
    pairwise: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Slide a fixed-length window along each engine's stream."""
    rows, meta = [], []
    t = np.arange(window, dtype=float)
    skipped = 0

    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        arr = g[channels].to_numpy(dtype=float)
        if len(g) < window:
            skipped += 1
            continue
        ruls = g["rul"].to_numpy()
        cycles = g["cycle"].to_numpy()
        subset = g["subset"].iloc[0]
        origin = g["origin"].iloc[0]

        for end in range(window - 1, len(g), stride):
            seg = arr[end - window + 1 : end + 1]
            feats = []
            for c in range(seg.shape[1]):
                feats.extend(_channel_features(seg[:, c], t))
            if pairwise:
                with np.errstate(invalid="ignore", divide="ignore"):
                    cm = np.corrcoef(seg, rowvar=False)
                cm = np.nan_to_num(cm)
                iu = np.triu_indices(cm.shape[0], k=1)
                feats.extend(cm[iu].tolist())
            rows.append(feats)
            meta.append(
                {
                    "unit": unit,
                    "subset": subset,
                    "origin": origin,
                    "cycle_end": int(cycles[end]),
                    "rul": float(ruls[end]),
                }
            )

    if skipped:
        print(f"  skipped {skipped} engines shorter than window={window}")
    if not rows:
        raise RuntimeError("No windows produced. Lower --window.")
    return np.asarray(rows, dtype=np.float32), pd.DataFrame(meta)


# --------------------------------------------------------------------------
# Labels and splits
# --------------------------------------------------------------------------
def label_health(rul: pd.Series, bins: list[int], labels: list[str]) -> pd.Series:
    edges = [-np.inf] + sorted(bins) + [np.inf]
    return pd.cut(rul, bins=edges, labels=labels, right=True).astype(str)


def grouped_split(meta: pd.DataFrame, test_frac: float, seed: int) -> pd.Series:
    """Hold out whole engines, never individual windows."""
    rng = np.random.default_rng(seed)
    units = np.array(sorted(meta["unit"].unique()))
    rng.shuffle(units)
    n_test = max(1, int(round(len(units) * test_frac)))
    held = set(units[:n_test])
    return meta["unit"].map(lambda u: "test" if u in held else "train")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(args) -> None:
    frames = [load_subset(args.data_dir, s) for s in args.subsets]
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(df):,} cycles from {df['unit'].nunique():,} engines")

    if args.max_engines:
        df = subsample_engines(df, args.max_engines, args.seed)
    channels = drop_flat_sensors(df, force=args.sensors)
    df = normalize_by_condition(df, args.conditions)
    print(f"  {len(channels)} varying sensors, {args.conditions} operating regime(s)")

    X, meta = extract_windows(df, channels, args.window, args.stride, args.pairwise)
    names = build_feature_names(channels, args.pairwise)
    assert X.shape[1] == len(names), (X.shape[1], len(names))

    meta["health_state"] = label_health(meta["rul"], args.bins, DEFAULT_LABELS)
    meta["split"] = grouped_split(meta, args.test_frac, args.seed)

    # Standardise features using TRAINING ENGINES ONLY.
    tr = (meta["split"] == "train").to_numpy()
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd[sd < 1e-8] = 1.0
    X = ((X - mu) / sd).astype(np.float32)

    # Data contract, mirroring Paper 3's expression/genes/samples/metadata:
    #   X.npy (windows x features) | features.npy | windows.npy | metadata.csv
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "X.npy"), X)
    np.save(os.path.join(args.out, "features.npy"), np.array(names, dtype=object),
            allow_pickle=True)
    window_ids = (meta["unit"] + "@" + meta["cycle_end"].astype(str)).to_numpy()
    np.save(os.path.join(args.out, "windows.npy"), window_ids.astype(object),
            allow_pickle=True)
    meta.insert(0, "window_id", window_ids)
    meta.to_csv(os.path.join(args.out, "metadata.csv"), index=False)

    print(f"\nX: {X.shape[0]:,} windows x {X.shape[1]:,} features -> {args.out}")
    counts = meta["health_state"].value_counts()
    print("\nhealth state distribution (this is your imbalance):")
    for lab in DEFAULT_LABELS:
        n = int(counts.get(lab, 0))
        print(f"  {lab:<10} {n:>7,}  {100 * n / len(meta):5.1f}%")
    rarest, biggest = counts.min(), counts.max()
    print(f"\n  imbalance ratio: {biggest / max(rarest, 1):.1f}x")
    print(f"  engines: {meta['unit'].nunique():,} "
          f"({(meta['split'] == 'train').sum():,} train windows / "
          f"{(meta['split'] == 'test').sum():,} held-out windows, split by engine)")


def self_test() -> None:
    """Generate C-MAPSS-shaped synthetic data and run the whole pipeline."""
    import tempfile

    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    rows = []
    for unit in range(1, 26):
        n = int(rng.integers(90, 200))
        for cyc in range(1, n + 1):
            wear = (cyc / n) ** 2
            ops = rng.normal(0, 0.001, 3)
            sens = rng.normal(0, 0.05, 21) + wear * np.linspace(0, 2, 21)
            sens[[0, 4, 9]] = 100.0  # constant sensors, as in FD001
            rows.append([unit, cyc, *ops, *sens])
    pd.DataFrame(rows).to_csv(
        os.path.join(tmp, "train_FD001.txt"), sep=" ", header=False, index=False
    )

    args = argparse.Namespace(
        data_dir=tmp, subsets=["FD001"], conditions=1, window=30, stride=5,
        sensors=None, max_engines=None,
        pairwise=True, bins=DEFAULT_BINS, test_frac=0.25, seed=1,
        out=os.path.join(tmp, "out"),
    )
    run(args)
    X = np.load(os.path.join(args.out, "X.npy"))
    meta = pd.read_csv(os.path.join(args.out, "metadata.csv"))
    assert np.isfinite(X).all(), "non-finite features"
    assert len(X) == len(meta)
    assert set(meta.loc[meta["split"] == "train", "unit"]).isdisjoint(
        set(meta.loc[meta["split"] == "test", "unit"])
    ), "engine leaked across the split"
    print("\nself-test passed: pipeline is sound, no leakage, features finite.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="directory holding the unzipped .txt files")
    p.add_argument("--subsets", nargs="+", default=["FD001"],
                   choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--conditions", type=int, default=1,
                   help="operating regimes: 1 for FD001/FD003, 6 for FD002/FD004")
    p.add_argument("--window", type=int, default=30, help="window length in cycles")
    p.add_argument("--stride", type=int, default=5, help="window stride in cycles")
    p.add_argument("--pairwise", action="store_true",
                   help="add cross-sensor correlation features")
    p.add_argument("--bins", type=int, nargs=3, default=DEFAULT_BINS,
                   help="RUL cut points for health states")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="./prepped")
    p.add_argument("--max-engines", type=int, default=None,
                   help="cap the engine count, stratified by origin; use to "
                        "balance cohorts of unequal size")
    p.add_argument("--sensors", nargs="+", default=None,
                   help="explicit sensor list, e.g. sensor_2 sensor_3 ...; "
                        "bypasses the variance screen so cohorts share a "
                        "feature space")
    p.add_argument("--self-test", action="store_true",
                   help="run end-to-end on synthetic data, no download needed")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.data_dir:
        p.error("--data-dir is required (or use --self-test)")
    run(args)


if __name__ == "__main__":
    sys.exit(main())