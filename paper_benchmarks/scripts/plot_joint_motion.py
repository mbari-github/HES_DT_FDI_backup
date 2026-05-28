#!/usr/bin/env python3
"""
Plot joint motion from benchmark CSV to confirm the mechanism moves correctly.

Usage:
  python3 plot_joint_motion.py --input <csv> --output <dir> --prefix <prefix> --dt 0.001
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Joint metadata: (column_name, label, unit, is_prismatic)
JOINTS = [
    ("rev_crank",             "rev_crank (θ)",        "rad", False),
    ("rev_body2linkAC",       "rev_body2linkAC",       "rad", False),
    ("rev_crank2shaft",       "rev_crank2shaft",       "rad", False),
    ("slider",                "slider",                "m",   True),
    ("rev_slider2linkBC",     "rev_slider2linkBC",     "rad", False),
    ("rev_linkAC2linkCE",     "rev_linkAC2linkCE",     "rad", False),
    ("rev_palmo2prossimale",  "MCF (palmo→prossimale)","rad", False),
    ("rev_prossimale2mediale","IFP (prossimale→mediale)","rad",False),
    ("slider2",               "slider2",               "m",   True),
]


def plot_joint_motion(df, output_dir, prefix, dt):
    os.makedirs(output_dir, exist_ok=True)
    t = df["step"].values * dt  # time axis in seconds

    # ── 1. theta and theta_dot ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, df["theta"].values, color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("theta (rad)", fontsize=11)
    axes[0].set_title("Reduced DOF: theta (rev_crank)", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, df["theta_dot"].values, color="#ff7f0e", linewidth=0.8)
    axes[1].set_ylabel("theta_dot (rad/s)", fontsize=11)
    axes[1].set_title("Angular velocity: theta_dot", fontsize=12)
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}theta_trajectory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.basename(path)}")

    # ── 2. All joint states in a grid ──────────────────────────────────
    available = [j for j in JOINTS if j[0] in df.columns]
    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), sharex=True)
    axes_flat = axes.flatten() if n > 1 else [axes]

    colors_rev  = "#1f77b4"
    colors_pris = "#2ca02c"

    for i, (col, label, unit, is_pris) in enumerate(available):
        ax = axes_flat[i]
        color = colors_pris if is_pris else colors_rev
        ax.plot(t, df[col].values, color=color, linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (s)", fontsize=9)

        # Stats annotation
        vals = df[col].values
        ax.annotate(
            f"min={vals.min():.3g}  max={vals.max():.3g}",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=7, va="top", color="gray"
        )

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Joint States — Mechanism Motion Verification", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}joint_states.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.basename(path)}")

    # ── 3. Phase portrait: theta vs theta_dot ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["theta"].values, df["theta_dot"].values,
        c=t, cmap="viridis", s=0.5, alpha=0.6
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Time (s)", fontsize=10)
    ax.set_xlabel("theta (rad)", fontsize=11)
    ax.set_ylabel("theta_dot (rad/s)", fontsize=11)
    ax.set_title("Phase portrait: theta vs theta_dot", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}phase_portrait.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.basename(path)}")

    # ── 4. Finger joints (MCF + IFP) zoomed ───────────────────────────
    finger_cols = [j for j in available if j[0] in ("rev_palmo2prossimale", "rev_prossimale2mediale")]
    if finger_cols:
        fig, axes = plt.subplots(len(finger_cols), 1, figsize=(12, 4 * len(finger_cols)), sharex=True)
        if len(finger_cols) == 1:
            axes = [axes]
        for ax, (col, label, unit, _) in zip(axes, finger_cols):
            ax.plot(t, df[col].values, color="#d62728", linewidth=0.8)
            ax.set_ylabel(unit, fontsize=10)
            ax.set_title(f"Finger joint: {label}", fontsize=11)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)", fontsize=10)
        fig.suptitle("Finger joints (passive Fung spring-damper)", fontsize=12)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}finger_joints.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {os.path.basename(path)}")

    # ── 5. Print summary ───────────────────────────────────────────────
    print("\n=== Joint Motion Summary ===")
    print(f"{'Joint':<30} {'min':>10} {'max':>10} {'range':>10} {'unit'}")
    print("-" * 65)
    for col, label, unit, _ in available:
        vals = df[col].values
        print(f"{label:<30} {vals.min():>10.4f} {vals.max():>10.4f} {vals.max()-vals.min():>10.4f}  {unit}")


def main():
    parser = argparse.ArgumentParser(description="Plot joint motion from benchmark CSV")
    parser.add_argument("--input",  required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prefix", default="",    help="Filename prefix")
    parser.add_argument("--dt",     type=float, default=0.001, help="Timestep in seconds")
    args = parser.parse_args()

    if args.prefix and not args.prefix.endswith("_"):
        args.prefix += "_"

    df = pd.read_csv(args.input)

    missing = [j[0] for j in JOINTS if j[0] not in df.columns]
    if missing:
        print(f"WARNING: joint columns missing from CSV: {missing}")
        print("  Run the benchmark with the updated code that records joint states.")
        return

    plot_joint_motion(df, args.output, args.prefix, args.dt)
    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
