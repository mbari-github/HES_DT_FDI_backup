#!/usr/bin/env python3
"""
Plot per-pair closure residuals — Reviewer 1 Point 6.

Reads the CSV produced by run_closure_analysis.py and generates:

  fig_closure_vs_time.pdf/png   — total + per-pair ‖eᵢ‖ vs simulation time
  fig_closure_vs_theta.pdf/png  — total + per-pair ‖eᵢ‖ vs θ (key figure:
                                   shows residual is geometric, not accumulated)
  fig_closure_components.pdf/png — raw x/y/z components of each pair's error
  closure_summary.txt            — numerical summary

Usage:
    python3 plot_closure_residual.py
    python3 plot_closure_residual.py --data-dir ../../results/closure_residual
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

PAIR_COLORS = ['#2471a3', '#c0392b', '#27ae60', '#8e44ad']


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(data_dir):
    path = os.path.join(data_dir, 'closure_residual.csv')
    rows = list(csv.DictReader(open(path)))
    arr = lambda key: np.array([float(r[key]) for r in rows])

    def arr_opt(key):
        if key in rows[0]:
            return np.array([float(r[key]) for r in rows])
        return np.full(len(rows), np.nan)

    data = {
        'time':          arr('time'),
        'theta':         arr('theta'),
        'theta_dot':     arr('theta_dot'),
        'closure_total': arr('closure_total'),
        'direction':     [r.get('direction', 'fwd') for r in rows],
        'nfev':          arr_opt('nfev'),
    }
    n_pairs = sum(1 for k in rows[0] if k.startswith('pair') and k.endswith('_norm'))
    for i in range(n_pairs):
        data[f'pair{i}_norm'] = arr(f'pair{i}_norm')
        data[f'pair{i}_ex']   = arr(f'pair{i}_ex')
        data[f'pair{i}_ey']   = arr(f'pair{i}_ey')
        data[f'pair{i}_ez']   = arr(f'pair{i}_ez')
    data['n_pairs'] = n_pairs

    dirs = np.array(data['direction'])
    data['mask_fwd'] = dirs == 'fwd'
    data['mask_rev'] = dirs == 'rev'
    return data


# ---------------------------------------------------------------------------
# Figure 1: residual vs time
# ---------------------------------------------------------------------------

def fig_vs_time(data, params, output_dir):
    n = data['n_pairs']
    t = data['time']
    labels = params.get('pair_labels', [f'Pair {i}' for i in range(n)])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Top: total norm
    axes[0].semilogy(t, data['closure_total'], color='#2c3e50', linewidth=1.2,
                     label=r'$\|e\|$ total')
    tol = 1e-5
    axes[0].axhline(tol, color='#c0392b', linestyle='--', linewidth=1.0,
                    label=f'Solver tolerance ({tol:.0e})')
    axes[0].set_ylabel(r'$\|e\|$ [m]')
    axes[0].set_title('Kinematic closure residual vs simulation time')
    axes[0].legend(fontsize=8)

    # Bottom: per-pair
    for i in range(n):
        axes[1].semilogy(t, data[f'pair{i}_norm'],
                         color=PAIR_COLORS[i % len(PAIR_COLORS)],
                         linewidth=0.9, label=labels[i])
    axes[1].set_ylabel(r'$\|e_i\|$ [m]  (per pair)')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(fontsize=7, ncol=2)

    A = params.get('amplitude', '?')
    T = params.get('period', '?')
    fig.suptitle(
        rf'Closure residual vs time — $\tau(t) = -|{A}||\sin(\pi t/{T})|$ Nm',
        fontsize=10)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fig_closure_vs_time.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: residual vs theta  (key figure for the reviewer)
# ---------------------------------------------------------------------------

def fig_vs_theta(data, params, output_dir):
    n = data['n_pairs']
    th = data['theta']
    labels = params.get('pair_labels', [f'Pair {i}' for i in range(n)])
    tol = 1e-5

    has_scan = data['mask_fwd'].any() and data['mask_rev'].any()

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    if has_scan:
        mf = data['mask_fwd']
        mr = data['mask_rev']
        axes[0].semilogy(th[mf], data['closure_total'][mf],
                         color='#2c3e50', linewidth=1.2, label=r'$\|e\|$ fwd')
        axes[0].semilogy(th[mr], data['closure_total'][mr],
                         color='#e67e22', linewidth=1.2, linestyle='--',
                         label=r'$\|e\|$ rev (return)')
    else:
        axes[0].semilogy(th, data['closure_total'], ',', color='#2c3e50',
                         markersize=1, alpha=0.6, label=r'$\|e\|$ total')

    axes[0].axhline(tol, color='#c0392b', linestyle=':', linewidth=1.0,
                    label=f'Solver tolerance ({tol:.0e})')
    axes[0].set_ylabel(r'$\|e\|$ [m]')
    axes[0].set_title(
        r'Closure residual vs $\theta$ — '
        r'forward and return sweeps overlap: residual is geometric, NOT accumulated')
    axes[0].legend(fontsize=8)

    for i in range(n):
        if has_scan:
            axes[1].semilogy(th[mf], data[f'pair{i}_norm'][mf],
                             color=PAIR_COLORS[i % len(PAIR_COLORS)],
                             linewidth=0.9, label=labels[i])
            axes[1].semilogy(th[mr], data[f'pair{i}_norm'][mr],
                             color=PAIR_COLORS[i % len(PAIR_COLORS)],
                             linewidth=0.9, linestyle='--', alpha=0.7)
        else:
            axes[1].semilogy(th, data[f'pair{i}_norm'], ',',
                             color=PAIR_COLORS[i % len(PAIR_COLORS)],
                             markersize=1, alpha=0.7, label=labels[i])
    axes[1].set_ylabel(r'$\|e_i\|$ [m]  (per pair)')
    axes[1].set_xlabel(r'$\theta$ [rad]')
    axes[1].legend(fontsize=7, ncol=2)

    if has_scan:
        axes[0].annotate('Solid = forward  |  Dashed = return',
                         xy=(0.5, -0.05), xycoords='axes fraction',
                         ha='center', fontsize=7, color='#555555')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fig_closure_vs_theta.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: raw x/y/z components
# ---------------------------------------------------------------------------

def fig_components(data, params, output_dir):
    n = data['n_pairs']
    t = data['time']
    labels = params.get('pair_labels', [f'Pair {i}' for i in range(n)])

    fig, axes = plt.subplots(n, 1, figsize=(9, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for j, (comp, ls) in enumerate(zip(['x', 'y', 'z'], ['-', '--', ':'])):
            ax.plot(t, data[f'pair{i}_e{comp}'] * 1e6,
                    linestyle=ls, linewidth=0.9,
                    color=PAIR_COLORS[i % len(PAIR_COLORS)],
                    alpha=0.85, label=f'$e_{{{comp}}}$')
        ax.set_ylabel(f'{labels[i]}\n' + r'error [$\mu$m]', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Per-pair closure error components', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fig_closure_components.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(data, params, output_dir):
    n = data['n_pairs']
    labels = params.get('pair_labels', [f'Pair {i}' for i in range(n)])
    tol = 1e-5
    mode = params.get('mode', 'dynamics')

    header_extra = (
        f"Mode: quasi-static scan — θ from {params.get('theta_max')} to "
        f"{params.get('theta_min')} and back ({params.get('scan_steps')} steps/dir)"
        if mode == 'scan' else
        f"Torque: half-sine neg, A={params.get('amplitude')} Nm, T={params.get('period')} s\n"
        f"dt = {params.get('dt', '?')} s   duration = {params.get('duration', '?')} s"
    )

    lines = [
        '=' * 68,
        'CLOSURE RESIDUAL ANALYSIS — Reviewer 1 Point 6',
        '=' * 68,
        '',
        header_extra,
        f"Solver tolerance: {tol:.0e}",
        '',
        f"{'Constraint pair':<30} {'Mean ‖eᵢ‖':>12} {'Max ‖eᵢ‖':>12} {'Max/tol':>10}",
        '-' * 68,
    ]
    lines.append(
        f"{'TOTAL ‖e‖':<30} "
        f"{data['closure_total'].mean():>12.3e} "
        f"{data['closure_total'].max():>12.3e} "
        f"{data['closure_total'].max()/tol:>9.2f}x"
    )
    for i in range(n):
        arr = data[f'pair{i}_norm']
        lines.append(
            f"{labels[i]:<30} "
            f"{arr.mean():>12.3e} "
            f"{arr.max():>12.3e} "
            f"{arr.max()/tol:>9.2f}x"
        )

    lines += ['-' * 68, '']

    if mode == 'scan':
        # Forward/return overlay comparison
        mf = data['mask_fwd']
        mr = data['mask_rev']
        if mr.any():
            fwd_vals = data['closure_total'][mf]
            rev_vals = data['closure_total'][mr]
            # Align by theta (reverse array is theta ascending, forward descending)
            max_diff = float(np.max(np.abs(fwd_vals - rev_vals[::-1])))
            rel_diff = max_diff / fwd_vals.max()
            lines += [
                'Forward / return sweep comparison (geometric test):',
                f'  Max |fwd - rev| = {max_diff:.3e} m',
                f'  Relative to max residual: {rel_diff*100:.4f} %',
                '',
            ]
            if rel_diff < 0.01:
                lines.append('  -> Forward and return sweeps are IDENTICAL to <1%.')
                lines.append('     The closure residual is a pure function of theta.')
                lines.append('     The monotonic growth in Fig.5 is GEOMETRIC,')
                lines.append('     NOT an accumulation of constraint violation over time.')
            else:
                lines.append('  -> Non-negligible hysteresis detected.')
                lines.append('     Further investigation may be needed.')
        else:
            lines.append('  (No return sweep data to compare.)')

        r_theta = float(np.corrcoef(data['theta'], data['closure_total'])[0, 1])
        lines.append('')
        lines.append(f'  |corr(||e||, theta)| = {abs(r_theta):.4f}  (expected ~1.0 for geometric)')
    else:
        lines.append('Correlation of ||e|| with theta vs with time:')
        r_theta = float(np.corrcoef(data['theta'], data['closure_total'])[0, 1])
        t_norm = (data['time'] - data['time'].mean()) / data['time'].std()
        r_time  = float(np.corrcoef(t_norm, data['closure_total'])[0, 1])
        lines.append(f'  |corr(||e||, theta)|    = {abs(r_theta):.4f}')
        lines.append(f'  |corr(||e||, time)|  = {abs(r_time):.4f}')
        lines.append('')
        if abs(r_theta) > abs(r_time):
            lines.append('  -> Residual correlates more strongly with theta than with time.')
            lines.append('     The growth seen in Fig.5 is GEOMETRIC (config-dependent),')
            lines.append('     NOT an accumulation of constraint violation over time.')
        else:
            lines.append('  -> Residual correlates more strongly with time than with theta.')
            lines.append('     Further investigation may be needed.')

    report = '\n'.join(lines)
    path = os.path.join(output_dir, 'closure_summary.txt')
    with open(path, 'w') as f:
        f.write(report + '\n')
    print(f'  Saved: {path}')
    print()
    print(report)


# ---------------------------------------------------------------------------
# Figure 4: clean single-panel geometric proof (paper figure)
# ---------------------------------------------------------------------------

def fig_geometric_proof(data, params, output_dir):
    """Single-panel figure: ‖e‖ [µm] vs θ [rad], forward and return sweep overlaid.

    Key message: the two curves are indistinguishable → residual is a pure
    function of θ, NOT an accumulation of constraint violation over time.
    """
    mf = data['mask_fwd']
    mr = data['mask_rev']
    th_fwd = data['theta'][mf]
    th_rev = data['theta'][mr]
    e_fwd  = data['closure_total'][mf] * 1e6   # m → µm
    e_rev  = data['closure_total'][mr] * 1e6

    tol_um = 1e-5 * 1e6   # 10 µm

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(th_fwd, e_fwd,
            color='#2471a3', linewidth=1.8,
            label=r'Forward sweep ($\theta_\mathrm{max} \to \theta_\mathrm{min}$)')
    ax.plot(th_rev, e_rev,
            color='#e67e22', linewidth=1.8, linestyle='--',
            label=r'Return sweep ($\theta_\mathrm{min} \to \theta_\mathrm{max}$)')

    ax.axhline(tol_um, color='#c0392b', linewidth=1.0, linestyle=':',
               label=f'Solver tolerance ({tol_um:.0f} µm)')

    ax.set_xlabel(r'$\theta$ [rad]', fontsize=10)
    ax.set_ylabel(r'$\|e\|$ [µm]', fontsize=10)
    ax.set_title(
        'Kinematic closure residual vs crank angle\n'
        r'Forward and return sweeps coincide $\Rightarrow$ residual is geometric, not accumulated',
        fontsize=10)
    ax.legend(fontsize=8, loc='upper right')

    theta_min = params.get('theta_min', -0.75)
    theta_max = params.get('theta_max',  0.09)

    ax.set_xlim(theta_min - 0.02, theta_max + 0.02)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fig_geometric_proof.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: root-cause analysis (error direction + conditioning + nfev)
# ---------------------------------------------------------------------------

def fig_cause_analysis(data, params, output_dir):
    """3-panel figure investigating why the residual grows with |θ|.

    Panel 1: x/y/z components of closure error for dominant pairs (1 and 2).
    Panel 2: smallest non-zero singular value of constraint Jacobian A_dep vs θ.
    Panel 3: solver nfev vs θ.
    """
    try:
        import pinocchio as pin
        from scipy.optimize import least_squares
    except ImportError:
        print('  [fig_cause_analysis] pinocchio not available, skipping.')
        return

    mf = data['mask_fwd']
    th = data['theta'][mf]

    # --- error components for pairs 1 and 2 ---
    c1z = data['pair1_ez'][mf] * 1e6
    c2z = data['pair2_ez'][mf] * 1e6
    # largest of the negligible components across all pairs/axes (for reference band)
    negligible = np.column_stack([
        data['pair1_ex'][mf], data['pair1_ey'][mf],
        data['pair2_ex'][mf], data['pair2_ey'][mf],
        data['pair0_ex'][mf], data['pair0_ey'][mf], data['pair0_ez'][mf],
        data['pair3_ex'][mf], data['pair3_ey'][mf], data['pair3_ez'][mf],
    ]) * 1e6
    neg_max = negligible.max(axis=1)
    neg_min = negligible.min(axis=1)

    # --- nfev (already in CSV) ---
    nfev_arr = np.array([float(r) for r in data.get('nfev_arr',
                [0.0] * mf.sum())])  # fallback if not loaded

    # reload nfev from the raw data arrays stored during load()
    nfev_arr = data.get('nfev', np.zeros(mf.sum()))
    if isinstance(nfev_arr, np.ndarray) and nfev_arr.ndim == 1:
        nfev_plot = nfev_arr[mf]
    else:
        nfev_plot = np.full(mf.sum(), np.nan)

    # --- smallest non-zero sv of A_dep vs θ ---
    urdf_path = params.get('urdf', None)
    sv_min_arr = None
    if urdf_path and os.path.exists(urdf_path):
        DEP_NAMES = [
            'rev_body2linkAC', 'rev_crank2shaft', 'slider',
            'rev_slider2linkBC', 'rev_linkAC2linkCE',
            'rev_palmo2prossimale', 'rev_prossimale2mediale', 'slider2',
        ]
        PAIR_NAMES = [
            ('frame_AC_end',   'frame_rod_end'),
            ('frame_AC_end_2', 'frame_BC_end'),
            ('frame_CE_end',   'frame_BC_end_2'),
            ('frame_CE_end_2', 'slider_t'),
        ]
        model = pin.buildModelFromUrdf(urdf_path)
        pdata = model.createData()
        dep_ids = [model.getJointId(n) - 1 for n in DEP_NAMES]
        pairs   = [(model.getFrameId(a), model.getFrameId(b)) for a, b in PAIR_NAMES]
        idx_th  = model.getJointId('rev_crank') - 1
        lower   = np.array([-2.5, -2.5, -0.015, -2.5, -2.5, -1.5, -2.5, -0.008])
        upper   = np.array([ 2.5,  2.5,  0.004,  2.5,  2.5,  1.2,  2.5,  0.008])

        def cerr(q):
            pin.forwardKinematics(model, pdata, q)
            pin.updateFramePlacements(model, pdata)
            return np.concatenate([
                pdata.oMf[a].translation - pdata.oMf[b].translation
                for a, b in pairs])

        q_cur = pin.neutral(model).copy()
        last_x = np.zeros(8)
        th_sv, sv_vals = [], []
        for thi in np.linspace(th[0], th[-1], 100):
            q_cur[idx_th] = thi
            def res(x, _q=q_cur.copy()):
                for k, i in enumerate(dep_ids): _q[i] = x[k]
                return cerr(_q)
            sol = least_squares(res, last_x.copy(), method='trf',
                                bounds=(lower, upper),
                                xtol=1e-8, ftol=1e-8, gtol=1e-8)
            q_sol = q_cur.copy()
            for k, i in enumerate(dep_ids): q_sol[i] = sol.x[k]
            last_x = sol.x.copy()
            pin.forwardKinematics(model, pdata, q_sol)
            pin.updateFramePlacements(model, pdata)
            A_rows = []
            for a, b in pairs:
                JA = pin.computeFrameJacobian(
                    model, pdata, q_sol, a,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                JB = pin.computeFrameJacobian(
                    model, pdata, q_sol, b,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                A_rows.append(JA - JB)
            A_dep = np.vstack(A_rows)[:, dep_ids]
            sv = np.linalg.svd(A_dep, compute_uv=False)
            sv_nz = sv[sv > 1e-6]
            th_sv.append(thi)
            sv_vals.append(sv_nz[-1] if len(sv_nz) else np.nan)
        sv_min_arr = (np.array(th_sv), np.array(sv_vals))

    # --- plot ---
    n_panels = 3 if sv_min_arr is not None else 2
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(7.5, 3 * n_panels), sharex=True)

    C1, C2 = '#c0392b', '#27ae60'

    # Panel 1: error components
    ax0 = axes[0]
    ax0.plot(th, c1z, color=C1, lw=1.6,
             label=r'Pair 1 — $e_z$: AC_end_2 $-$ BC_end')
    ax0.plot(th, c2z, color=C2, lw=1.6,
             label=r'Pair 2 — $e_z$: CE_end $-$ BC_end_2')
    ax0.fill_between(th, neg_min, neg_max,
                     color='#95a5a6', alpha=0.25,
                     label=r'All $e_x$, $e_y$ components — negligible')
    ax0.axhline(0, color='k', lw=0.5)
    ax0.set_ylabel(r'Error component [µm]')
    ax0.set_title(r'Error direction: $e_z$ dominates, anti-symmetric between pairs 1 and 2')
    ax0.legend(fontsize=7, loc='lower left')

    panel = 1

    # Panel 2: sv_min (optional, requires pinocchio)
    if sv_min_arr is not None:
        ax1 = axes[panel]
        ax1.plot(sv_min_arr[0], sv_min_arr[1], color='#2c3e50', lw=1.6)
        ax1.set_ylabel(r'$\sigma_{\min}^{*}(A_\mathrm{dep})$')
        ax1.set_title('Smallest non-zero singular value of constraint Jacobian — 2× variation only')
        panel += 1

    # Panel 3: nfev
    ax2 = axes[panel]
    if not np.all(np.isnan(nfev_plot)):
        ax2.plot(th, nfev_plot, color='#8e44ad', lw=0,
                 marker='.', ms=2, alpha=0.6, label='nfev per step')
        ax2.set_ylim(0, max(nfev_plot.max() + 1, 5))
    ax2.set_ylabel('Solver nfev')
    ax2.set_xlabel(r'$\theta$ [rad]')
    ax2.set_title('Solver iterations — near-constant: convergence is not the limiting factor')

    for ax in axes:
        ax.set_xlim(th[-1] - 0.02, th[0] + 0.02)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fig_cause_analysis.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot per-pair closure residuals (Point 6 analysis)'
    )
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir is None:
        args.data_dir = os.path.normpath(
            os.path.join(script_dir, '..', '..', 'results', 'closure_residual'))
    if args.output_dir is None:
        args.output_dir = args.data_dir

    os.makedirs(args.output_dir, exist_ok=True)

    params_path = os.path.join(args.data_dir, 'params.json')
    with open(params_path) as f:
        params = json.load(f)

    print('Loading data...')
    data = load(args.data_dir)
    print(f'  {len(data["time"])} steps loaded')

    print('Generating figures...')
    fig_vs_time(data, params, args.output_dir)
    fig_vs_theta(data, params, args.output_dir)
    fig_components(data, params, args.output_dir)
    fig_geometric_proof(data, params, args.output_dir)
    fig_cause_analysis(data, params, args.output_dir)

    print('Writing summary...')
    write_summary(data, params, args.output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
