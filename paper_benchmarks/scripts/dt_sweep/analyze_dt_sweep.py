#!/usr/bin/env python3
"""
Analyze dt-sweep CSV data: compute RMSE, generate plots and report.

Usage:
    python3 analyze_dt_sweep.py [--data-dir ../../results/dt_sweep]
                                [--output-dir ../../results/dt_sweep]

Output in output-dir:
    - theta_overlay.png        — θ(t) for all dt overlaid
    - theta_error.png          — Δθ(t) for each dt (separate subplots)
    - rmse_vs_dt.png           — RMSE vs dt in log-log
    - magnitude_dynamics.png   — M_eff and proj over time (ref vs nominal)
    - report.txt               — summary statistics table
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ====================================================================
# LOAD
# ====================================================================

def load_data(data_dir: str, dt_values: list) -> dict:
    """Load all CSV files, keyed by dt value."""
    import csv
    data = {}
    for dt_val in dt_values:
        path = os.path.join(data_dir, f'dt_{dt_val:.6f}.csv')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        rows = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items()})
        arr = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}
        data[dt_val] = arr
        print(f"  Loaded: {path}  ({len(rows)} rows)")
    return data


def load_params(data_dir: str) -> dict:
    params_path = os.path.join(data_dir, 'params.json')
    with open(params_path, 'r') as f:
        return json.load(f)


# ====================================================================
# INTERPOLATION
# ====================================================================

def interpolate_to_grid(data: dict, ref_dt: float):
    """Interpolate all datasets onto the reference time grid."""
    ref = data[ref_dt]
    t_ref = ref['time']
    result = {}
    result[ref_dt] = ref  # reference stays as-is
    for dt_val, arr in data.items():
        if dt_val == ref_dt:
            continue
        t = arr['time']
        interp = {}
        for key in arr.keys():
            if key in ('step', 'at_limit', 'nfev'):
                # discrete fields: nearest-neighbor
                indices = np.clip(
                    np.searchsorted(t, t_ref, side='right') - 1, 0, len(t) - 1)
                interp[key] = arr[key][indices]
            elif key.startswith('B_') or key.startswith('Bdot_'):
                indices = np.clip(
                    np.searchsorted(t, t_ref, side='right') - 1, 0, len(t) - 1)
                interp[key] = arr[key][indices]
            elif key == 'time':
                interp[key] = t_ref
            else:
                # linear interpolation for continuous fields
                interp[key] = np.interp(t_ref, t, arr[key])
        result[dt_val] = interp
    return result, t_ref


# ====================================================================
# METRICS
# ====================================================================

def compute_rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def compute_metrics(interp_data: dict, ref_dt: float, dt_values: list,
                    nv: int) -> dict:
    """Compute RMSE and max|error| for each dt vs reference."""
    ref = interp_data[ref_dt]
    metrics = {}
    for dt_val in dt_values:
        if dt_val == ref_dt:
            metrics[dt_val] = {
                'dt_ms': dt_val * 1000,
                'rmse_theta': 0.0,
                'max_dtheta': 0.0,
                'rmse_thetadot': 0.0,
                'max_dthetadot': 0.0,
                'rmse_M_eff': 0.0,
                'rmse_proj': 0.0,
                'rmse_mean_Bdot': 0.0,
                'stable': True,
            }
            continue
        d = interp_data[dt_val]

        err_theta = d['theta'] - ref['theta']
        err_thetadot = d['theta_dot'] - ref['theta_dot']
        err_M_eff = d['M_eff'] - ref['M_eff']
        err_proj = d['proj'] - ref['proj']

        # Bdot RMSE averaged over all components
        bdot_errors = []
        for j in range(nv):
            k = f'Bdot_{j}'
            if k in d and k in ref:
                bdot_errors.append(d[k] - ref[k])
        rmse_mean_bdot = 0.0
        if bdot_errors:
            all_errors = np.array(bdot_errors)
            rmse_per_component = np.sqrt(np.mean(all_errors ** 2, axis=1))
            rmse_mean_bdot = float(np.mean(rmse_per_component))

        stable = not (np.any(np.isnan(d['theta'])) or
                      np.any(np.isinf(d['theta'])))

        metrics[dt_val] = {
            'dt_ms': dt_val * 1000,
            'rmse_theta': compute_rmse(err_theta),
            'max_dtheta': float(np.max(np.abs(err_theta))),
            'rmse_thetadot': compute_rmse(err_thetadot),
            'max_dthetadot': float(np.max(np.abs(err_thetadot))),
            'rmse_M_eff': compute_rmse(err_M_eff),
            'rmse_proj': compute_rmse(err_proj),
            'rmse_mean_Bdot': rmse_mean_bdot,
            'stable': stable,
        }
    return metrics


# ====================================================================
# PLOTS
# ====================================================================

def plot_theta_overlay(interp_data: dict, ref_dt: float, dt_values: list,
                       output_dir: str, t_ref: np.ndarray):
    """θ(t) overlay for all dt — zoom on first 5s."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(dt_values)))

    # Full view
    for i, dt_val in enumerate(dt_values):
        d = interp_data[dt_val]
        label = f'dt={dt_val*1000:.3f}ms' + (' (ref)' if dt_val == ref_dt else '')
        ls = '-' if dt_val == ref_dt else '--'
        lw = 2.0 if dt_val == ref_dt else 0.8
        ax1.plot(t_ref, d['theta'], color=colors[i], linestyle=ls,
                 linewidth=lw, label=label, alpha=0.8)

    ax1.set_ylabel('θ [rad]')
    ax1.set_title('θ(t) at different integration step sizes')
    ax1.legend(fontsize='small', ncol=2)
    ax1.grid(True, alpha=0.3)

    # Zoom first 5s
    zoom_idx = t_ref <= 5.0
    for i, dt_val in enumerate(dt_values):
        d = interp_data[dt_val]
        label = f'dt={dt_val*1000:.3f}ms' + (' (ref)' if dt_val == ref_dt else '')
        ls = '-' if dt_val == ref_dt else '--'
        lw = 2.0 if dt_val == ref_dt else 0.8
        ax2.plot(t_ref[zoom_idx], d['theta'][zoom_idx], color=colors[i],
                 linestyle=ls, linewidth=lw, label=label, alpha=0.8)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('θ [rad]')
    ax2.set_title('Zoom: first 5 seconds')
    ax2.legend(fontsize='small', ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'theta_overlay.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_theta_error(interp_data: dict, ref_dt: float, dt_values: list,
                     output_dir: str, t_ref: np.ndarray):
    """Δθ(t) subplots for each non-reference dt."""
    test_dts = [dt for dt in dt_values if dt != ref_dt]
    n = len(test_dts)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, dt_val in zip(axes, test_dts):
        d = interp_data[dt_val]
        err = d['theta'] - interp_data[ref_dt]['theta']
        ax.plot(t_ref, err, linewidth=0.6, color='crimson')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel(f'dt={dt_val*1000:.3f}ms\nΔθ [rad]')
        ax.grid(True, alpha=0.3)
        rmse = compute_rmse(err)
        ax.text(0.02, 0.95, f'RMSE = {rmse:.3e} rad',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('θ error relative to reference (dt = {:.5f} s)'.format(ref_dt),
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, 'theta_error.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_rmse_vs_dt(metrics: dict, ref_dt: float, output_dir: str):
    """RMSE vs dt in log-log scale."""
    dts = sorted([k for k in metrics.keys() if k != ref_dt])
    dt_ms = np.array([metrics[k]['dt_ms'] for k in dts])

    fig, ax = plt.subplots(figsize=(8, 5))

    fields = [
        ('rmse_theta', 'RMSE(θ)', 's'),
        ('rmse_thetadot', 'RMSE(θ̇)', '^'),
        ('rmse_proj', 'RMSE(proj)', 'D'),
        ('rmse_M_eff', 'RMSE(M_eff)', 'v'),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(fields)))

    for (key, label, marker), color in zip(fields, colors):
        vals = np.array([metrics[k][key] for k in dts])
        ax.loglog(dt_ms, vals, marker=marker, color=color,
                  label=label, linewidth=1.5, markersize=6)

    # Reference line O(dt) — slope 1 in log-log
    if len(dts) >= 2:
        ref_val = metrics[dts[0]]['rmse_theta']
        ref_dt_ms = dt_ms[0]
        # O(dt) line passing through first point
        dt_line = np.logspace(np.log10(dt_ms[0]), np.log10(dt_ms[-1]), 10)
        val_line = ref_val * (dt_line / ref_dt_ms)
        ax.loglog(dt_line, val_line, 'k--', linewidth=1.0, alpha=0.5,
                  label='O(dt) reference')

    ax.set_xlabel('Integration step dt [ms]')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs integration step size')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3, which='both')

    fmt = matplotlib.ticker.ScalarFormatter()
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    plt.tight_layout()
    path = os.path.join(output_dir, 'rmse_vs_dt.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_magnitude_dynamics(interp_data: dict, ref_dt: float, dt_values: list,
                            output_dir: str, t_ref: np.ndarray,
                            nominal_dt: float = 0.001):
    """M_eff(t) and proj(t): reference vs nominal dt."""
    ref = interp_data[ref_dt]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t_ref, ref['M_eff'], '-', linewidth=1.0, color='steelblue',
             label=f'ref (dt={ref_dt*1000:.3f}ms)')
    if nominal_dt in interp_data:
        d = interp_data[nominal_dt]
        ax1.plot(t_ref, d['M_eff'], '--', linewidth=0.8, color='crimson',
                 alpha=0.7, label=f'nominal (dt={nominal_dt*1000:.3f}ms)')
        err = d['M_eff'] - ref['M_eff']
        rmse = compute_rmse(err)
        ax1.text(0.02, 0.95, f'RMSE = {rmse:.3e}',
                 transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_ylabel('M_eff [kg·m²]')
    ax1.set_title('Effective inertia M_eff = B^T M B + Jm')
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_ref, ref['proj'], '-', linewidth=1.0, color='steelblue',
             label=f'ref (dt={ref_dt*1000:.3f}ms)')
    if nominal_dt in interp_data:
        d = interp_data[nominal_dt]
        ax2.plot(t_ref, d['proj'], '--', linewidth=0.8, color='crimson',
                 alpha=0.7, label=f'nominal (dt={nominal_dt*1000:.3f}ms)')
        err = d['proj'] - ref['proj']
        rmse = compute_rmse(err)
        ax2.text(0.02, 0.95, f'RMSE = {rmse:.3e}',
                 transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('proj [Nm]')
    ax2.set_title('Projected dynamics proj = B^T(M·Bdot·θ̇ + h)')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'magnitude_dynamics.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ====================================================================
# REPORT
# ====================================================================

def generate_report(metrics: dict, dt_values: list, ref_dt: float,
                    output_dir: str, params: dict):
    """Write summary report as text file."""
    lines = []
    lines.append("=" * 75)
    lines.append("DT-SWEEP ANALYSIS REPORT — Bdot Finite-Difference Sensitivity")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"Reference dt: {ref_dt*1000:.5f} ms")
    lines.append(f"Torque profile: sine, A={params['amplitude']} Nm, "
                 f"T={params['period']} s")
    lines.append(f"Simulation duration: {params['duration']} s")
    lines.append(f"Warmup: {params['warmup']} s")
    lines.append(f"URDF: {params['urdf']}")
    lines.append(f"Model nv: {params['nv']}")
    lines.append("")
    lines.append("-" * 75)
    lines.append(f"{'dt [ms]':<12} {'RMSE(θ)':<15} {'max|Δθ|':<15} "
                 f"{'RMSE(θ̇)':<15} {'RMSE(M_eff)':<15} {'RMSE(proj)':<15} "
                 f"{'RMSE(Bdot)':<15} {'Stable':<8}")
    lines.append("-" * 75)

    for dt_val in sorted(dt_values):
        m = metrics[dt_val]
        rmse_bdot = m['rmse_mean_Bdot']
        lines.append(
            f"{m['dt_ms']:<12.4f} "
            f"{m['rmse_theta']:<15.3e} "
            f"{m['max_dtheta']:<15.3e} "
            f"{m['rmse_thetadot']:<15.3e} "
            f"{m['rmse_M_eff']:<15.3e} "
            f"{m['rmse_proj']:<15.3e} "
            f"{rmse_bdot:<15.3e} "
            f"{'YES' if m['stable'] else 'DIVERGED':<8}"
        )

    lines.append("-" * 75)
    lines.append("")
    lines.append("Notes:")
    lines.append("  - All RMSE computed relative to reference dt = "
                 f"{ref_dt*1000:.5f} ms")
    lines.append("  - All trajectories interpolated onto reference time grid")
    lines.append("  - RMSE(Bdot) = mean RMSE across all Bdot components")
    lines.append("  - Bdot approximated as first-order finite difference: "
                 "Bdot = (B_k - B_{k-1}) / dt")
    lines.append("")

    report = '\n'.join(lines)
    path = os.path.join(output_dir, 'report.txt')
    with open(path, 'w') as f:
        f.write(report + '\n')
    print(f"  Saved: {path}")
    print()
    print(report)


# ====================================================================
# MAIN
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyze dt-sweep data and generate plots')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir is None:
        args.data_dir = os.path.normpath(
            os.path.join(script_dir, '..', '..', 'results', 'dt_sweep'))
    if args.output_dir is None:
        args.output_dir = args.data_dir

    os.makedirs(args.output_dir, exist_ok=True)

    # Load params
    params = load_params(args.data_dir)
    dt_values = sorted(params['dt_list'])
    ref_dt = dt_values[0]  # smallest dt is the reference
    nv = params['nv']

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    data = load_data(args.data_dir, dt_values)

    print(f"\nInterpolating onto reference time grid (dt={ref_dt*1000:.5f} ms)...")
    interp_data, t_ref = interpolate_to_grid(data, ref_dt)
    print(f"  Reference grid: {len(t_ref)} points, "
          f"t = [{t_ref[0]:.3f}, {t_ref[-1]:.3f}] s")

    print(f"\nComputing metrics...")
    metrics = compute_metrics(interp_data, ref_dt, dt_values, nv)

    print(f"\nGenerating plots...")
    plot_theta_overlay(interp_data, ref_dt, dt_values, args.output_dir, t_ref)
    plot_theta_error(interp_data, ref_dt, dt_values, args.output_dir, t_ref)
    plot_rmse_vs_dt(metrics, ref_dt, args.output_dir)

    nominal_dt = 0.001
    plot_magnitude_dynamics(interp_data, ref_dt, dt_values, args.output_dir,
                            t_ref, nominal_dt)

    print(f"\nGenerating report...")
    generate_report(metrics, dt_values, ref_dt, args.output_dir, params)

    print(f"\n{'=' * 60}")
    print("Analysis complete.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
