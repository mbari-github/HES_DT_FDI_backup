#!/usr/bin/env python3
"""
Stability analysis of the Euler integrator on dt-sweep data.

Detects numerical chattering by counting sign inversions in theta_dot as a
function of dt. Chattering is the signature of Euler-integration instability
on stiff friction terms (Coulomb + tanh). A rate above 50% means theta_dot
changes sign on more than half of consecutive steps — i.e., the system is
oscillating at the Nyquist frequency of the integration grid.

Usage:
    # Analyse nominal-friction sweep (shows chattering onset)
    python3 analyze_dt_stability.py

    # Analyse no-friction sweep (should show 0 chattering for all dt)
    python3 analyze_dt_stability.py --data-dir ../../results/dt_sweep_no_friction

Output (in --output-dir):
    stability_report.txt
    stability_sign_changes.png    — bar chart: sign inversions per dt
    stability_theta_dot_series.png — theta_dot time series, one subplot per dt
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ====================================================================
# DATA LOADING
# ====================================================================

def load_data(data_dir: str, dt_values: list) -> dict:
    data = {}
    for dt_val in dt_values:
        path = os.path.join(data_dir, f'dt_{dt_val:.6f}.csv')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        thetas, tdots = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                thetas.append(float(row['theta']))
                tdots.append(float(row['theta_dot']))
        data[dt_val] = {
            'theta': np.array(thetas),
            'theta_dot': np.array(tdots),
        }
        print(f"  Loaded: dt={dt_val * 1000:.3f} ms  ({len(thetas)} rows)")
    return data


# ====================================================================
# METRICS
# ====================================================================

def sign_changes(arr: np.ndarray) -> int:
    """Count consecutive pairs where theta_dot reverses sign (both non-zero)."""
    if len(arr) < 2:
        return 0
    s = np.sign(arr)
    nonzero = (s[:-1] != 0) & (s[1:] != 0)
    return int(np.sum((s[1:] != s[:-1]) & nonzero))


def compute_metrics(data: dict, dt_values: list) -> dict:
    results = {}
    for dt_val in dt_values:
        if dt_val not in data:
            continue
        td = data[dt_val]['theta_dot']
        th = data[dt_val]['theta']
        n = len(td)
        sc = sign_changes(td)
        rate = sc / max(n - 1, 1)
        results[dt_val] = {
            'n_steps': n,
            'sign_changes': sc,
            'rate': rate,
            'chattering': rate > 0.5,
            'theta_min': float(np.min(th)),
            'theta_max': float(np.max(th)),
            'theta_range': float(np.max(th) - np.min(th)),
            'tdot_max_abs': float(np.max(np.abs(td))),
        }
    return results


# ====================================================================
# PLOTS
# ====================================================================

def plot_sign_changes(results: dict, dt_values: list, output_dir: str) -> None:
    dts = sorted(dt for dt in dt_values if dt in results)
    dt_labels = [f'{dt * 1000:.3f}' for dt in dts]
    sc_counts = [results[dt]['sign_changes'] for dt in dts]
    rates = [results[dt]['rate'] * 100 for dt in dts]
    colors = ['#c0392b' if results[dt]['chattering'] else '#2980b9' for dt in dts]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    bars = ax1.bar(dt_labels, sc_counts, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, sc_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(sc_counts) * 0.02,
                 str(val), ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('θ̇ sign inversions [count]')
    ax1.set_title('Chattering detection: θ̇ sign inversions vs integration step\n'
                  '(red = chattering regime, rate > 50%)')
    ax1.grid(True, alpha=0.3, axis='y')

    bars2 = ax2.bar(dt_labels, rates, color=colors, edgecolor='black', width=0.5)
    ax2.axhline(50, color='#c0392b', linestyle='--', linewidth=1.2,
                label='50% threshold (chattering criterion)')
    ax2.set_xlabel('Integration step dt [ms]')
    ax2.set_ylabel('Sign inversion rate [%]')
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'stability_sign_changes.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_theta_dot_series(data: dict, results: dict, dt_values: list,
                          output_dir: str, max_t: float = 3.0) -> None:
    dts = sorted(dt for dt in dt_values if dt in data)
    n_plots = len(dts)
    fig, axes = plt.subplots(n_plots, 1, figsize=(11, 2.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, dt_val in zip(axes, dts):
        td = data[dt_val]['theta_dot']
        t = np.arange(len(td)) * dt_val
        mask = t <= max_t
        chattering = results[dt_val]['chattering']
        color = '#c0392b' if chattering else '#2980b9'

        ax.plot(t[mask], td[mask], linewidth=0.6, color=color, alpha=0.85)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_ylabel(f'dt={dt_val * 1000:.3f} ms\nθ̇ [rad/s]', fontsize=9)
        ax.grid(True, alpha=0.25)

        sc = results[dt_val]['sign_changes']
        rate = results[dt_val]['rate'] * 100
        status = '⚠ CHATTERING' if chattering else '✓ stable'
        fc = '#fadbd8' if chattering else '#d5e8d4'
        ax.text(0.01, 0.95,
                f"{status}   sign inv.: {sc}  ({rate:.1f}%)",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=fc, alpha=0.8))

    axes[-1].set_xlabel(f'Time [s]   (first {max_t} s shown)')
    fig.suptitle('θ̇(t) at different integration steps — chattering detection', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, 'stability_theta_dot_series.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ====================================================================
# REPORT
# ====================================================================

def generate_report(results: dict, dt_values: list, output_dir: str,
                    params: dict) -> None:
    no_friction = params.get('no_friction', False)
    no_coulomb = params.get('no_coulomb', False)
    amplitude = params.get('amplitude', '?')
    period = params.get('period', '?')

    if no_friction:
        friction_label = 'DISABLED — fric_coul=fric_visc=damping=0'
    elif no_coulomb:
        friction_label = 'Coulomb only disabled — fric_coul=0, fric_visc=2.0, damping=1.0'
    else:
        friction_label = 'nominal — fric_coul=2.0, fric_visc=2.0, damping=1.0'

    lines = [
        '=' * 68,
        'STABILITY ANALYSIS — Euler integrator chattering detection',
        '=' * 68,
        '',
        f"Torque profile : sine,  A = {amplitude} Nm,  T = {period} s",
        f"Friction mode  : {friction_label}",
        f"Chattering criterion: sign inversion rate > 50 %",
        '',
        f"{'dt [ms]':<10} {'Steps':>8} {'Sign inv.':>10} {'Rate [%]':>10} "
        f"{'θ range [rad]':>14} {'Status':>12}",
        '-' * 68,
    ]

    for dt_val in sorted(results.keys()):
        r = results[dt_val]
        status = 'CHATTERING' if r['chattering'] else 'stable'
        lines.append(
            f"{dt_val * 1000:<10.4f} {r['n_steps']:>8} {r['sign_changes']:>10} "
            f"{r['rate'] * 100:>9.1f}% {r['theta_range']:>14.4f} {status:>12}"
        )

    lines += [
        '-' * 68,
        '',
        'Interpretation:',
    ]
    chattering_dts = [dt for dt in sorted(results) if results[dt]['chattering']]
    stable_dts = [dt for dt in sorted(results) if not results[dt]['chattering']]
    if chattering_dts:
        lines.append(
            f"  Chattering onset: dt >= {min(chattering_dts) * 1000:.3f} ms"
        )
        lines.append(
            f"  Safe range for Euler + stiff Coulomb friction: dt <= "
            f"{max(stable_dts) * 1000:.3f} ms"
            if stable_dts else "  No stable dt found in tested range."
        )
    else:
        lines.append('  All tested dt values are stable (no chattering detected).')
        if no_friction or no_coulomb:
            lines.append(
                '  Expected: with Coulomb friction disabled, Euler integration '
                'is unconditionally stable for this smooth system.'
            )

    report = '\n'.join(lines)
    path = os.path.join(output_dir, 'stability_report.txt')
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
        description='Euler integrator stability analysis on dt-sweep CSV data'
    )
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing dt_*.csv and params.json '
                             '(default: ../../results/dt_sweep)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots and report (default: same as data-dir)')
    parser.add_argument('--max-t', type=float, default=3.0,
                        help='Max time shown in theta_dot series plot in seconds (default: 3.0)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir is None:
        args.data_dir = os.path.normpath(
            os.path.join(script_dir, '..', '..', 'results', 'dt_sweep'))
    if args.output_dir is None:
        args.output_dir = args.data_dir

    os.makedirs(args.output_dir, exist_ok=True)

    params_path = os.path.join(args.data_dir, 'params.json')
    if not os.path.exists(params_path):
        print(f"ERROR: params.json not found in {args.data_dir}", flush=True)
        raise SystemExit(1)

    with open(params_path) as f:
        params = json.load(f)
    dt_values = sorted(params['dt_list'])

    print('=' * 60)
    print('Loading data...')
    print('=' * 60)
    data = load_data(args.data_dir, dt_values)

    if not data:
        print('ERROR: no CSV files loaded.')
        raise SystemExit(1)

    print('\nComputing stability metrics...')
    results = compute_metrics(data, dt_values)

    print('\nGenerating plots...')
    plot_sign_changes(results, dt_values, args.output_dir)
    plot_theta_dot_series(data, results, dt_values, args.output_dir, max_t=args.max_t)

    print('\nGenerating report...')
    generate_report(results, dt_values, args.output_dir, params)

    print(f"\n{'=' * 60}")
    print('Analysis complete.')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
