#!/usr/bin/env python3
"""
03_som_trajectory.py
====================
The centrepiece figure: a SOM class-dominance map with engine degradation
trajectories drawn across it.

Panel A  one engine's cycle-by-cycle path from the healthy region into the
         imminent patch, with cycle labels.
Panel B  many engines overlaid, showing the path is a shared route rather
         than one engine's accident.

Also emits a TRAJECTORY COHERENCE statistic, so the figure is not merely
decorative. If consecutive windows land on adjacent map nodes far more often
than a shuffled cycle order would produce, the map has organised itself along
the degradation axis. Without this number, a reviewer can fairly say the line
looks smooth only because you drew it.

The dominance background is computed from TRAINING windows only. Trajectories
may be drawn for held-out engines, which then shows the map generalising.

Usage
-----
    python 03_som_trajectory.py --prepped data/fd001
    python 03_som_trajectory.py --prepped data/fd001 --engine FD001_24 --seed 3
    python 03_som_trajectory.py --prepped data/fd003 --out outputs/figures_fd003
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# Ordinal severity, so the palette must be sequential, not categorical.
STATES = ["healthy", "degrading", "critical", "imminent"]
COLORS = ["#2c7fb8", "#7fcdbb", "#fec44f", "#d95f0e"]
UNOWNED = "#f0f0f0"


# --------------------------------------------------------------------------
def fit_som(X: np.ndarray, side: int, seed: int, iters: int):
    from minisom import MiniSom

    som = MiniSom(side, side, X.shape[1], sigma=max(1.0, side / 4),
                  learning_rate=0.5, random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, iters, verbose=False)
    return som


def bmu_coords(som, X: np.ndarray) -> np.ndarray:
    """(n, 2) array of best-matching-unit lattice coordinates."""
    return np.array([som.winner(x) for x in X])


def dominance_grid(coords: np.ndarray, y: np.ndarray, side: int) -> np.ndarray:
    """Per-node plurality class index, or -1 where no window landed."""
    grid = np.full((side, side), -1, dtype=int)
    df = pd.DataFrame({"r": coords[:, 0], "c": coords[:, 1], "y": y})
    for (r, c), g in df.groupby(["r", "c"]):
        grid[r, c] = STATES.index(g["y"].value_counts().idxmax())
    return grid


# --------------------------------------------------------------------------
def trajectory_coherence(coords: np.ndarray, rng, n_null: int = 500) -> dict:
    """Mean lattice step between consecutive windows vs a shuffled-order null.

    A map organised along the degradation axis moves in small steps as cycles
    advance. Shuffling the cycle order destroys that and inflates the step.
    """
    def mean_step(c):
        return float(np.linalg.norm(np.diff(c, axis=0), axis=1).mean())

    obs = mean_step(coords)
    null = np.array([mean_step(coords[rng.permutation(len(coords))])
                     for _ in range(n_null)])
    # One-sided: is the observed path SHORTER than chance?
    p = (np.sum(null <= obs) + 1) / (n_null + 1)
    return {"observed": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "p": float(p),
            "ratio": float(obs / null.mean())}


# --------------------------------------------------------------------------
def draw_background(ax, grid: np.ndarray, side: int) -> None:
    cmap = ListedColormap([UNOWNED] + COLORS)
    ax.imshow(grid + 1, cmap=cmap, vmin=0, vmax=4, origin="lower",
              interpolation="nearest", alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, side - 0.5); ax.set_ylim(-0.5, side - 0.5)
    for s in ax.spines.values():
        s.set_edgecolor("#999999")


def draw_path(ax, coords: np.ndarray, rul: np.ndarray, jitter, lw=1.6,
              alpha=1.0, markers=True) -> None:
    """Lattice coords are integers; jitter separates repeated visits."""
    xy = coords[:, ::-1].astype(float) + jitter
    ax.plot(xy[:, 0], xy[:, 1], "-", color="#111111", lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=3)
    if not markers:
        return
    ax.scatter(xy[:, 0], xy[:, 1], c=rul, cmap="magma_r", s=26,
               edgecolors="white", linewidths=0.5, zorder=4)
    ax.scatter(*xy[0], marker="o", s=110, facecolor="none",
               edgecolor="#111111", lw=1.8, zorder=5)
    ax.scatter(*xy[-1], marker="X", s=130, color="#111111", zorder=5)


# --------------------------------------------------------------------------
def run(args) -> None:
    X = np.load(os.path.join(args.prepped, "X.npy"))
    md = pd.read_csv(os.path.join(args.prepped, "metadata.csv"))
    side = int(round(np.sqrt(args.capacity)))
    if side * side != args.capacity:
        raise ValueError(f"capacity {args.capacity} is not a perfect square")

    tr = (md["split"] == "train").to_numpy()
    print(f"fitting {side}x{side} SOM on {tr.sum():,} training windows "
          f"({args.iters:,} iterations, seed {args.seed})")
    som = fit_som(X[tr], side, args.seed, args.iters)

    coords = bmu_coords(som, X)              # every window, train and held out
    grid = dominance_grid(coords[tr], md.loc[tr, "health_state"].to_numpy(), side)

    owned = (grid >= 0).sum()
    imm = (grid == STATES.index("imminent")).sum()
    print(f"  {owned}/{args.capacity} nodes own a class, "
          f"{imm} dominated by 'imminent'")

    # -- choose engines: prefer complete run-to-failure trajectories ---------
    lengths = md.groupby("unit").size().sort_values(ascending=False)
    if args.engine:
        focus = args.engine
        if focus not in set(md["unit"]):
            raise ValueError(f"unknown engine {focus}")
    else:
        focus = lengths.index[0]
    others = [u for u in lengths.index[: args.n_overlay + 1] if u != focus]
    others = others[: args.n_overlay]
    print(f"  focus engine {focus} ({lengths[focus]} windows), "
          f"{len(others)} overlaid")

    rng = np.random.default_rng(args.seed)

    # -- coherence, on the focus engine and pooled over the overlay ----------
    def eng(u):
        g = md.index[md["unit"] == u].to_numpy()
        order = np.argsort(md.loc[g, "cycle_end"].to_numpy())
        return g[order]

    idx = eng(focus)
    coh = trajectory_coherence(coords[idx], rng)
    print(f"\ntrajectory coherence ({focus}):")
    print(f"  observed mean step {coh['observed']:.3f} lattice units")
    print(f"  shuffled-order null {coh['null_mean']:.3f} +/- {coh['null_sd']:.3f}")
    print(f"  ratio {coh['ratio']:.3f}, p = {coh['p']:.4f}")

    rows = []
    for u in [focus] + others:
        c = trajectory_coherence(coords[eng(u)], rng, n_null=200)
        c["unit"] = u
        c["n_windows"] = len(eng(u))
        rows.append(c)
    coh_df = pd.DataFrame(rows)[["unit", "n_windows", "observed", "null_mean",
                                 "null_sd", "ratio", "p"]]
    print(f"\nacross {len(coh_df)} engines: median ratio "
          f"{coh_df['ratio'].median():.3f}, "
          f"{(coh_df['p'] < 0.05).sum()}/{len(coh_df)} with p < 0.05")

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))

    draw_background(axes[0], grid, side)
    j = rng.normal(0, 0.10, (len(idx), 2))
    draw_path(axes[0], coords[idx], md.loc[idx, "rul"].to_numpy(), j)
    axes[0].set_title(f"A  Engine {focus}: cycles {md.loc[idx,'cycle_end'].min()}"
                      f"\u2013{md.loc[idx,'cycle_end'].max()}",
                      loc="left", fontsize=11)

    draw_background(axes[1], grid, side)
    for u in others:
        i = eng(u)
        jj = rng.normal(0, 0.10, (len(i), 2))
        draw_path(axes[1], coords[i], None, jj, lw=1.0, alpha=0.45,
                  markers=False)
    axes[1].set_title(f"B  {len(others)} engines, same map", loc="left",
                      fontsize=11)

    handles = [Line2D([], [], marker="s", ls="", ms=10, color=c,
                      label=s.capitalize()) for s, c in zip(STATES, COLORS)]
    handles += [
        Line2D([], [], marker="o", ls="", mfc="none", mec="#111", ms=9,
               label="First window"),
        Line2D([], [], marker="X", ls="", color="#111", ms=10,
               label="Failure"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Degradation trajectories across the SOM class-dominance map",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"fig3_som_trajectory_cap{args.capacity}"
                                  f"_seed{args.seed}")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=220, bbox_inches="tight")
    coh_df.to_csv(stem + "_coherence.csv", index=False)
    np.savez_compressed(stem + "_som.npz", coords=coords, grid=grid,
                        weights=som.get_weights())
    print(f"\nwrote {stem}.pdf / .png / _coherence.csv / _som.npz")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepped", required=True)
    p.add_argument("--out", default="./outputs/figures")
    p.add_argument("--capacity", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=50_000)
    p.add_argument("--engine", help="unit id, e.g. FD001_24; default longest")
    p.add_argument("--n-overlay", type=int, default=12)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    sys.exit(main())
