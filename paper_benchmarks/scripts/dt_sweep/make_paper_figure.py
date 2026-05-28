#!/usr/bin/env python3
"""
Paper figure for Reviewer 1 – Point 3:
  "Sensitivity of the Bdot finite-difference approximation and impact on
   integration stability as a function of the time step Δt."

Produces two publication-quality figures in --output-dir:
  fig_dt_rmse.pdf / .png   — log-log RMSE(θ), RMSE(θ̇), RMSE(Ḃ) vs Δt
                             with O(Δt) and O(Δt²) reference lines
  fig_dt_stability.pdf / .png — θ̇(t) subplot per dt, colour-coded stable/chattering

Usage:
    python3 make_paper_figure.py
    python3 make_paper_figure.py --data-dir ../../results/dt_sweep_paper
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   10,
    'legend.fontsize':  8,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'lines.linewidth':  1.4,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'figure.dpi':       150,
})

COLOR_STABLE    = '#2471a3'
COLOR_CHAT      = '#c0392b'
COLOR_REF_O1    = '#7d6608'
COLOR_REF_O2    = '#1d8348'


# ── I/O ────────────────────────────────────────────────────────────────────

def load_csv(path):
    thetas, tdots = [], []
    B_cols = None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    nv = sum(1 for k in rows[0] if k.startswith('B_') and not k.startswith('Bdot'))
    for row in rows:
        thetas.append(float(row['theta']))
        tdots.append(float(row['theta_dot']))
    Bdot = np.zeros((len(rows), nv))
    for j in range(nv):
        Bdot[:, j] = [float(r[f'Bdot_{j}']) for r in rows]
    return {
        'theta':     np.array(thetas),
        'theta_dot': np.array(tdots),
        'Bdot':      Bdot,
        'nv':        nv,
    }


def interp_onto(arr, dt_src, dt_ref, t_start, t_end):
    t_src = np.arange(len(arr)) * dt_src + t_start
    t_ref = np.arange(int(round((t_end - t_start) / dt_ref))) * dt_ref + t_start
    t_ref = t_ref[t_ref <= t_end + 1e-12]
    return np.interp(t_ref, t_src, arr), t_ref


# ── Metrics ────────────────────────────────────────────────────────────────

def sign_changes(arr):
    s = np.sign(arr)
    nz = (s[:-1] != 0) & (s[1:] != 0)
    return int(np.sum((s[1:] != s[:-1]) & nz))


def compute_all(data_dir, dt_values, dt_ref):
    ref = load_csv(os.path.join(data_dir, f'dt_{dt_ref:.6f}.csv'))
    t_start = 0.0
    n_ref = len(ref['theta'])
    t_end = (n_ref - 1) * dt_ref

    metrics = {}
    for dt in dt_values:
        if dt == dt_ref:
            continue
        d = load_csv(os.path.join(data_dir, f'dt_{dt:.6f}.csv'))
        if d is None:
            continue

        th_i, t_i = interp_onto(d['theta'],     dt, dt_ref, t_start, t_end)
        td_i, _   = interp_onto(d['theta_dot'], dt, dt_ref, t_start, t_end)

        n = min(len(th_i), n_ref)
        rmse_th = float(np.sqrt(np.mean((th_i[:n] - ref['theta'][:n])**2)))
        rmse_td = float(np.sqrt(np.mean((td_i[:n] - ref['theta_dot'][:n])**2)))

        nv = d['Bdot'].shape[1]
        bdot_rmse_per_comp = []
        for j in range(nv):
            bd_i, _ = interp_onto(d['Bdot'][:, j], dt, dt_ref, t_start, t_end)
            bdot_rmse_per_comp.append(
                float(np.sqrt(np.mean((bd_i[:n] - ref['Bdot'][:n, j])**2)))
            )
        rmse_bd = float(np.mean(bdot_rmse_per_comp))

        sc   = sign_changes(d['theta_dot'])
        rate = sc / max(len(d['theta_dot']) - 1, 1)

        metrics[dt] = {
            'rmse_theta':    rmse_th,
            'rmse_tdot':     rmse_td,
            'rmse_bdot':     rmse_bd,
            'sign_changes':  sc,
            'rate':          rate,
            'chattering':    rate > 0.5,
            'theta_range':   float(np.max(d['theta']) - np.min(d['theta'])),
        }
    return metrics, ref


# ── Figure 1: RMSE vs Δt (log-log) ─────────────────────────────────────────

def fig_rmse(metrics, params, output_dir):
    dts = sorted(metrics)
    dt_ms   = np.array([d * 1e3 for d in dts])
    rmse_th = np.array([metrics[d]['rmse_theta'] for d in dts])
    rmse_td = np.array([metrics[d]['rmse_tdot']  for d in dts])
    rmse_bd = np.array([metrics[d]['rmse_bdot']  for d in dts])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # ── Left: θ and θ̇ RMSE ──
    ax = axes[0]
    ax.loglog(dt_ms, rmse_th, 'o-', color=COLOR_STABLE,  label=r'RMSE($\theta$)')
    ax.loglog(dt_ms, rmse_td, 's--', color='#1abc9c',    label=r'RMSE($\dot{\theta}$)')

    # O(Δt) reference anchored at coarsest stable point
    i_anch = 0   # finest non-ref dt
    dt0 = dt_ms[i_anch]
    c1  = rmse_th[i_anch] / dt0
    ax.loglog(dt_ms, c1 * dt_ms, ':', color=COLOR_REF_O1, linewidth=1.0, label=r'$O(\Delta t)$')

    ax.set_xlabel(r'Integration step $\Delta t$ [ms]')
    ax.set_ylabel('RMSE [rad  or  rad/s]')
    ax.set_title(r'Trajectory accuracy vs $\Delta t$')
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    _annotate_points(ax, dt_ms, rmse_th)

    # ── Right: Ḃ RMSE ──
    ax2 = axes[1]
    ax2.loglog(dt_ms, rmse_bd, 'D-', color=COLOR_CHAT, label=r'RMSE$(\dot{B})$ [mean over components]')

    c1b = rmse_bd[i_anch] / dt0
    ax2.loglog(dt_ms, c1b * dt_ms, ':', color=COLOR_REF_O1, linewidth=1.0, label=r'$O(\Delta t)$')

    ax2.set_xlabel(r'Integration step $\Delta t$ [ms]')
    ax2.set_ylabel(r'RMSE$(\dot{B})$ [s$^{-1}$]')
    ax2.set_title(r'$\dot{B}(q)$ FD approximation error vs $\Delta t$')
    ax2.legend()
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    _annotate_points(ax2, dt_ms, rmse_bd)

    A       = params.get('amplitude', '?')
    T       = params.get('period',    '?')
    profile = params.get('profile',   'sine')
    th_min  = params.get('theta_min', -2.5)
    th_max  = params.get('theta_max',  2.5)
    if profile == 'half_sine_neg':
        torque_str = rf'$\tau(t)=-|{A}|\,|\sin(\pi t/{T})|$ Nm (always $\leq 0$)'
    else:
        torque_str = rf'$\tau(t)={A}\sin(2\pi t/{T})$ Nm'
    fig.suptitle(
        rf'Sensitivity to $\Delta t$ — {torque_str}, '
        rf'$\theta\in[{th_min},{th_max}]$ rad, '
        r'nominal friction ($f_c=2$ Nm, $b_v=2$ Nm·s/rad, $\beta=1$ Nm·s/rad)',
        fontsize=8, y=1.01
    )
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fig_dt_rmse.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


def _annotate_points(ax, xs, ys):
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.1e}', (x, y),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7, color='#333333')


# ── Figure 2: θ̇(t) stability ───────────────────────────────────────────────

def fig_stability(data_dir, metrics, dt_values, dt_ref, params, output_dir,
                  max_t=5.0):
    dts = sorted(d for d in dt_values if d != dt_ref
                 and os.path.exists(os.path.join(data_dir, f'dt_{d:.6f}.csv')))

    n_plots = len(dts)
    fig, axes = plt.subplots(n_plots, 1,
                             figsize=(8.5, 2.6 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, dt in zip(axes, dts):
        d    = load_csv(os.path.join(data_dir, f'dt_{dt:.6f}.csv'))
        td   = d['theta_dot']
        t    = np.arange(len(td)) * dt
        mask = t <= max_t

        m    = metrics.get(dt, {})
        chat = m.get('chattering', False)
        rate = m.get('rate', 0.0) * 100
        sc   = m.get('sign_changes', 0)

        color  = COLOR_CHAT if chat else COLOR_STABLE
        status = '[!] CHATTERING' if chat else '[ok] stable'
        fc     = '#fadbd8'      if chat else '#d5e8d4'

        ax.plot(t[mask], td[mask], linewidth=0.7, color=color, alpha=0.9)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_ylabel(fr'$\Delta t={dt*1e3:.2g}$ ms' '\n' r'$\dot{\theta}$ [rad/s]',
                      fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.text(0.01, 0.95,
                f'{status}   sign-inv.: {sc}  ({rate:.1f}%)',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=fc, alpha=0.8))

    axes[-1].set_xlabel(f'Time [s]  (first {max_t} s shown)')
    A       = params.get('amplitude', '?')
    T       = params.get('period',    '?')
    profile = params.get('profile',   'sine')
    if profile == 'half_sine_neg':
        torque_str = rf'$\tau(t)=-|{A}||\sin(\pi t/{T})|$ Nm'
    else:
        torque_str = rf'$\tau(t)={A}\sin(2\pi t/{T})$ Nm'
    fig.suptitle(
        rf'Integration stability: $\dot{{\theta}}(t)$ at different $\Delta t$ — {torque_str}',
        fontsize=10
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fig_dt_stability.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


# ── Figure 3: θ̇(t) vs reference — chattering overlay ──────────────────────

def _find_chattering_window(tdot, dt, window=2.0):
    """Return (t_start, t_end) of the window with the most sign inversions."""
    n_win = max(1, int(window / dt))
    s = np.sign(tdot)
    inv = np.zeros(len(s), dtype=int)
    inv[1:] = ((s[1:] != s[:-1]) & (s[:-1] != 0) & (s[1:] != 0)).astype(int)
    cum = np.cumsum(inv)
    # sliding window count
    counts = cum[n_win:] - cum[:-n_win]
    if len(counts) == 0:
        return 0.0, window
    i_best = int(np.argmax(counts))
    return i_best * dt, (i_best + n_win) * dt


def fig_thetadot_vs_ref(data_dir, metrics, dt_values, dt_ref, params, output_dir,
                        zoom_window=2.0):
    """
    For each non-reference dt: θ̇(t) overlaid with reference.
    Left column = full signal; right column = auto-zoom on chattering region.
    """
    dts = sorted(d for d in dt_values if d != dt_ref
                 and os.path.exists(os.path.join(data_dir, f'dt_{d:.6f}.csv')))
    if not dts:
        return

    ref = load_csv(os.path.join(data_dir, f'dt_{dt_ref:.6f}.csv'))
    t_ref = np.arange(len(ref['theta_dot'])) * dt_ref

    n = len(dts)
    fig, axes = plt.subplots(n, 2, figsize=(11.0, 2.8 * n),
                             gridspec_kw={'width_ratios': [2.5, 1]})
    if n == 1:
        axes = axes[np.newaxis, :]

    COLOR_REF = '#999999'

    for row, dt in enumerate(dts):
        d    = load_csv(os.path.join(data_dir, f'dt_{dt:.6f}.csv'))
        t    = np.arange(len(d['theta_dot'])) * dt
        td   = d['theta_dot']
        tdR  = ref['theta_dot']

        m    = metrics.get(dt, {})
        sc   = m.get('sign_changes', 0)
        rate = m.get('rate', 0.0) * 100
        chat = m.get('chattering', False)
        color = COLOR_CHAT if chat else COLOR_STABLE
        fc    = '#fadbd8' if chat else '#d5e8d4'
        label_dt  = rf'$\Delta t={dt*1e3:.2g}$ ms'
        label_ref = rf'ref ($\Delta t={dt_ref*1e3:.2g}$ ms)'

        # ── left: full signal ──
        ax = axes[row, 0]
        ax.plot(t_ref, tdR, color=COLOR_REF, linewidth=0.7, zorder=1, label=label_ref)
        ax.plot(t,     td,  color=color,     linewidth=0.6, alpha=0.85, zorder=2, label=label_dt)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel(r'$\dot{\theta}$ [rad/s]', fontsize=9)
        ax.legend(loc='lower right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.25)

        status = '[!] chattering' if chat else '[ok] stable'
        ax.text(0.01, 0.97,
                f'{status}   inversions: {sc} ({rate:.1f}%)',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor=fc, alpha=0.8))

        # ── right: zoom on chattering region ──
        ax2 = axes[row, 1]
        t0, t1 = _find_chattering_window(td, dt, window=zoom_window)
        # extend slightly for context
        t0 = max(0.0, t0 - 0.1)
        t1 = min(t[-1], t1 + 0.1)

        mask_z  = (t     >= t0) & (t     <= t1)
        mask_rz = (t_ref >= t0) & (t_ref <= t1)

        ax2.plot(t_ref[mask_rz], tdR[mask_rz],
                 color=COLOR_REF, linewidth=0.9, zorder=1, label=label_ref)
        ax2.plot(t[mask_z],  td[mask_z],
                 color=color, linewidth=0.7, alpha=0.9, zorder=2, label=label_dt)
        ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax2.set_ylabel(r'$\dot{\theta}$ [rad/s]', fontsize=8)
        ax2.set_title(f'Zoom [{t0:.1f}–{t1:.1f}] s', fontsize=8)
        ax2.grid(True, alpha=0.25)
        ax2.tick_params(labelsize=8)

        # shade chattering intervals in zoom
        s = np.sign(td)
        inv_idx = np.where(np.diff(s) != 0)[0]
        for ii in inv_idx:
            ti = t[ii]
            if t0 <= ti <= t1:
                ax2.axvline(ti, color=COLOR_CHAT, linewidth=0.4, alpha=0.35)

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')

    A       = params.get('amplitude', '?')
    T       = params.get('period',    '?')
    profile = params.get('profile',   'sine')
    if profile == 'half_sine_neg':
        torque_str = rf'$\tau(t)=-|{A}|\,|\sin(\pi t/{T})|$ Nm'
    else:
        torque_str = rf'$\tau(t)={A}\sin(2\pi t/{T})$ Nm'
    fig.suptitle(
        r'$\dot{\theta}(t)$ vs reference: numerical chattering from stiff Coulomb friction'
        '\n' + rf'({torque_str}; left = full signal, right = auto-zoom on highest-inversion window)',
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fig_dt_thetadot_ref.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures for Reviewer Point 3 (Bdot sensitivity)'
    )
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--max-t', type=float, default=5.0,
                        help='Max time shown in stability plot [s]')
    parser.add_argument('--zoom-window', type=float, default=2.0,
                        help='Duration [s] of the chattering zoom window (default: 2.0)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir is None:
        args.data_dir = os.path.normpath(
            os.path.join(script_dir, '..', '..', 'results', 'dt_sweep_paper'))
    if args.output_dir is None:
        args.output_dir = args.data_dir

    os.makedirs(args.output_dir, exist_ok=True)

    params_path = os.path.join(args.data_dir, 'params.json')
    with open(params_path) as f:
        params = json.load(f)

    dt_values = sorted(params['dt_list'])
    dt_ref    = dt_values[0]   # finest = reference

    print('Computing metrics...')
    metrics, _ = compute_all(args.data_dir, dt_values, dt_ref)

    print('Generating RMSE figure...')
    fig_rmse(metrics, params, args.output_dir)

    print('Generating stability figure...')
    fig_stability(args.data_dir, metrics, dt_values, dt_ref, params,
                  args.output_dir, max_t=args.max_t)

    print('Generating θ̇ vs reference figure...')
    fig_thetadot_vs_ref(args.data_dir, metrics, dt_values, dt_ref, params,
                        args.output_dir, zoom_window=args.zoom_window)

    print('\nSummary table:')
    print(f"{'dt [ms]':>9} {'RMSE(θ) [rad]':>15} {'RMSE(θ̇)':>12} "
          f"{'RMSE(Ḃ)':>12} {'Chattering':>12}")
    print('-' * 65)
    for dt in sorted(metrics):
        m = metrics[dt]
        s = 'YES' if m['chattering'] else 'no'
        print(f"{dt*1e3:>9.3f} {m['rmse_theta']:>15.3e} {m['rmse_tdot']:>12.3e} "
              f"{m['rmse_bdot']:>12.3e} {s:>12}")

    print('\nDone.')


if __name__ == '__main__':
    main()
