#!/usr/bin/env python3
"""
plot_baseline.py — visualization of the healthy baseline bag
================================================================

Reads a baseline bag and produces a multi-page PDF (or per-page PNGs)
with diagnostic plots:

  Page 1 — Joint state (theta, theta_dot) over time
  Page 2 — Trajectory reference vs measured theta (admittance tracking)
  Page 3 — External wrench input
  Page 4 — Torque (commanded by trajectory_controller)
  Page 5 — Observer residuals (raw)
  Page 6 — Distributions of all three residuals (histograms)
  Page 7 — Per-regime statistics (one bar chart per signal)

Usage
-----
    python3 plot_baseline.py --bag /path/to/bag_dir
    python3 plot_baseline.py --bag /path/to/bag_dir --manifest baseline_manifest.json
    python3 plot_baseline.py --bag /path/to/bag_dir --output my_plots.pdf
    python3 plot_baseline.py --bag /path/to/bag_dir --png   # produces individual PNGs

The PDF (or PNGs) is/are written to the bag directory by default.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')                       # non-interactive, no DISPLAY needed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import rclpy.serialization as ser
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f'ERROR: ROS 2 Python bindings not available ({e})', file=sys.stderr)
    print('       Source the workspace before running this script.', file=sys.stderr)
    sys.exit(1)


JOINT_NAME = 'rev_crank'

# Topics we want, mapped to a (signal_name -> extractor lambda) entry that
# returns one or more (key, value) tuples per message.
def _extract_joint_state(msg):
    try:
        idx = list(msg.name).index(JOINT_NAME)
    except ValueError:
        return []
    out = []
    if idx < len(msg.position):
        out.append(('theta', float(msg.position[idx])))
    if idx < len(msg.velocity):
        out.append(('theta_dot', float(msg.velocity[idx])))
    return out


def _extract_traj(msg):
    d = list(msg.data)
    out = []
    if len(d) >= 1: out.append(('theta_v',     float(d[0])))
    if len(d) >= 2: out.append(('theta_dot_v', float(d[1])))
    if len(d) >= 3: out.append(('theta_ddot_v',float(d[2])))
    return out


def _extract_wrench(msg):
    # geometry_msgs/WrenchStamped — we extract just force.x as 'wrench_fx'
    return [
        ('wrench_fx', float(msg.wrench.force.x)),
        ('wrench_fy', float(msg.wrench.force.y)),
        ('wrench_fz', float(msg.wrench.force.z)),
    ]


TOPIC_EXTRACTORS = {
    '/joint_states':                ('sensor_msgs/msg/JointState',
                                     _extract_joint_state),
    '/trajectory_ref':              ('exoskeleton_safety_msgs/msg/Float64ArrayStamped',
                                     _extract_traj),
    '/torque':                      ('exoskeleton_safety_msgs/msg/Float64Stamped',
                                     lambda m: [('tau_m', float(m.data))]),
    '/exo_dynamics/tau_ext_theta':  ('exoskeleton_safety_msgs/msg/Float64Stamped',
                                     lambda m: [('tau_ext', float(m.data))]),
    '/exo_dynamics/external_wrench':('geometry_msgs/msg/WrenchStamped',
                                     _extract_wrench),
    '/fdi/debug':                  ('exoskeleton_safety_msgs/msg/Float64ArrayStamped',
                                     lambda m: [
                                         ('r_force',        float(m.data[6]) if len(m.data) > 6 else 0.0),
                                         ('r_encoder',      float(m.data[7]) if len(m.data) > 7 else 0.0),
                                         ('r_force_filt',   float(m.data[8]) if len(m.data) > 8 else 0.0),
                                         ('r_encoder_filt', float(m.data[9]) if len(m.data) > 9 else 0.0),
                                     ]),
}


def _detect_storage_id(bag_path: str) -> str:
    p = Path(bag_path)
    if any(p.glob('*.mcap')):
        return 'mcap'
    if any(p.glob('*.db3')):
        return 'sqlite3'
    return 'mcap'


def read_bag(bag_path: str) -> dict:
    """Returns dict signal_name -> (times[s], values), times relative to bag start."""
    storage = StorageOptions(uri=bag_path, storage_id=_detect_storage_id(bag_path))
    converter = ConverterOptions(input_serialization_format='cdr',
                                 output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage, converter)
    msg_classes = {t.name: get_message(t.type)
                   for t in reader.get_all_topics_and_types()}

    series_t = defaultdict(list)
    series_v = defaultdict(list)
    t0 = None

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic not in TOPIC_EXTRACTORS:
            continue
        cls = msg_classes.get(topic)
        if cls is None:
            continue
        msg = ser.deserialize_message(raw, cls)
        ts = float(t_ns) * 1e-9
        if t0 is None:
            t0 = ts
        ts -= t0

        for name, value in TOPIC_EXTRACTORS[topic][1](msg):
            series_t[name].append(ts)
            series_v[name].append(value)

    return {n: (np.array(series_t[n]), np.array(series_v[n]))
            for n in series_v}


# ════════════════════════════════════════════════════════════════════
# Plotting helpers
# ════════════════════════════════════════════════════════════════════

def downsample(t: np.ndarray, v: np.ndarray, max_points: int = 50000):
    """
    Visual downsampling to keep PDF file size reasonable. Picks every
    Nth sample. Statistics shown in titles still come from the full data.
    """
    if len(t) <= max_points:
        return t, v
    step = len(t) // max_points + 1
    return t[::step], v[::step]


def add_regime_markers(ax, manifest):
    """Draw vertical lines at regime boundaries with labels at the top."""
    if not manifest:
        return
    # Sort by 't_start_relative' or fall back to 't_start' shifted to start
    if 't_start_relative' in manifest[0]:
        starts = [(m['name'], float(m['t_start_relative'])) for m in manifest]
    else:
        # Older manifests: convert wall-clock to relative
        t0 = min(float(m['t_start']) for m in manifest)
        starts = [(m['name'], float(m['t_start']) - t0) for m in manifest]
    starts.sort(key=lambda x: x[1])

    ymin, ymax = ax.get_ylim()
    for i, (name, ts) in enumerate(starts):
        ax.axvline(ts, color='gray', linestyle=':', linewidth=0.5, alpha=0.6)
        # Label every regime; smaller font, alternating high/low so they don't overlap
        y_label = ymax - (ymax - ymin) * (0.05 + 0.04 * (i % 3))
        ax.text(ts, y_label, name, fontsize=5, rotation=90,
                verticalalignment='top', alpha=0.6,
                family='monospace')


def plot_state(data, manifest, ax_pair):
    """Page 1 — theta & theta_dot."""
    ax_th, ax_thd = ax_pair

    if 'theta' in data:
        t, v = data['theta']
        td, vd = downsample(t, v)
        ax_th.plot(td, vd, linewidth=0.5, color='C0')
        ax_th.set_ylabel('theta [rad]')
        ax_th.set_title(f'theta (plant)   '
                        f'mean={v.mean():+.4f}, std={v.std():.4f}, '
                        f'range=[{v.min():+.4f}, {v.max():+.4f}]')
        ax_th.grid(True, alpha=0.3)
        # Reference lines for the wall limits
        ax_th.axhline(-0.75, color='red', linestyle='--', linewidth=0.7,
                      alpha=0.5, label='theta_min')
        ax_th.axhline( 0.09, color='red', linestyle='--', linewidth=0.7,
                      alpha=0.5, label='theta_max')
        ax_th.legend(fontsize=7, loc='lower right')
        add_regime_markers(ax_th, manifest)

    if 'theta_dot' in data:
        t, v = data['theta_dot']
        td, vd = downsample(t, v)
        ax_thd.plot(td, vd, linewidth=0.5, color='C0')
        ax_thd.set_ylabel('theta_dot [rad/s]')
        ax_thd.set_xlabel('time [s]')
        ax_thd.set_title(f'theta_dot   '
                         f'mean={v.mean():+.4f}, std={v.std():.4f}, '
                         f'max|.|={np.abs(v).max():.4f}')
        ax_thd.grid(True, alpha=0.3)
        add_regime_markers(ax_thd, manifest)


def plot_tracking(data, manifest, ax):
    """Page 2 — admittance tracking: theta_v vs theta."""
    if 'theta' in data:
        t, v = data['theta']
        td, vd = downsample(t, v)
        ax.plot(td, vd, linewidth=0.5, color='C0', label='theta (plant)')
    if 'theta_v' in data:
        t, v = data['theta_v']
        td, vd = downsample(t, v)
        ax.plot(td, vd, linewidth=0.5, color='C1', label='theta_v (admittance)',
                alpha=0.7)
    ax.set_ylabel('angle [rad]')
    ax.set_xlabel('time [s]')
    ax.set_title('Admittance tracking — theta_v (virtual) vs theta (plant)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(-0.75, color='red', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.axhline( 0.09, color='red', linestyle='--', linewidth=0.7, alpha=0.5)
    add_regime_markers(ax, manifest)


def plot_wrench(data, manifest, ax):
    """Page 3 — external wrench input (force.x, the dominant component)."""
    if 'wrench_fx' in data:
        t, v = data['wrench_fx']
        td, vd = downsample(t, v)
        ax.plot(td, vd, linewidth=0.5, color='C2', label='F_x')
    if 'wrench_fy' in data:
        t, v = data['wrench_fy']
        td, vd = downsample(t, v)
        ax.plot(td, vd, linewidth=0.5, color='C3', label='F_y', alpha=0.5)
    if 'wrench_fz' in data:
        t, v = data['wrench_fz']
        td, vd = downsample(t, v)
        ax.plot(td, vd, linewidth=0.5, color='C4', label='F_z', alpha=0.5)
    ax.set_ylabel('force [N]')
    ax.set_xlabel('time [s]')
    ax.set_title('External wrench (input)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    add_regime_markers(ax, manifest)


def plot_torque(data, manifest, ax):
    """Page 4 — commanded torque."""
    if 'tau_m' in data:
        t, v = data['tau_m']
        td, vd = downsample(t, v)
        ax.plot(td, vd, linewidth=0.5, color='C5')
        ax.set_ylabel('tau_m [Nm]')
        ax.set_xlabel('time [s]')
        ax.set_title(f'Commanded torque   '
                     f'mean={v.mean():+.4f}, std={v.std():.4f}, '
                     f'range=[{v.min():+.4f}, {v.max():+.4f}]')
        ax.grid(True, alpha=0.3)
        add_regime_markers(ax, manifest)


def plot_residuals_time(data, manifest, axes):
    """Page 5 — FDI residuals over time, filtered."""
    for (name, color, ylabel), ax in zip(
        [('r_force_filt',   'C0', 'r_force_filt [Nm]'),
         ('r_encoder_filt', 'C1', 'r_encoder_filt [rad]')],
        axes,
    ):
        if name not in data:
            ax.set_visible(False)
            continue
        t, v = data[name]
        td, vd = downsample(t, v)
        ax.plot(td, vd, linewidth=0.5, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name}   '
                     f'mean={v.mean():+.4g}, std={v.std():.4g}, '
                     f'p99.9(|dev|)={np.percentile(np.abs(v - v.mean()), 99.9):.4g}')
        ax.grid(True, alpha=0.3)
        add_regime_markers(ax, manifest)
    axes[-1].set_xlabel('time [s]')


def plot_residual_distributions(data, axes):
    """Page 7 — distributions of FDI filtered residuals."""
    for (name, color), ax in zip(
        [('r_force_filt', 'C0'), ('r_encoder_filt', 'C1')],
        axes,
    ):
        if name not in data:
            ax.set_visible(False)
            continue
        _, v = data[name]
        ax.hist(v, bins=120, density=True, alpha=0.7, color=color,
                edgecolor='none')
        ax.set_xlabel(name)
        ax.set_ylabel('density')
        # Add reference lines for percentiles
        m = v.mean()
        for q, c in [(0.5, '0.5'), (99.5, '0.5')]:
            x = np.percentile(v, q)
            ax.axvline(x, color='gray', linestyle=':', linewidth=0.7)
        p999 = np.percentile(np.abs(v - m), 99.9)
        ax.axvline(m + p999, color='red', linestyle='--', linewidth=0.7,
                   label=f'p99.9 = ±{p999:.4g}')
        ax.axvline(m - p999, color='red', linestyle='--', linewidth=0.7)
        ax.set_title(f'{name}   mean={m:+.4g}, std={v.std():.4g}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)


def plot_per_regime(data, manifest, axes):
    """Page 8 — per-regime mean ± std bar chart for selected signals."""
    if not manifest:
        for ax in axes:
            ax.set_visible(False)
        return

    # Decide regime intervals
    # Manifests have either 't_start_relative' (new) or 't_start' (old)
    if 't_start_relative' in manifest[0]:
        starts = sorted([(m['name'], float(m['t_start_relative']),
                          float(m.get('sub_index', 0)))
                         for m in manifest], key=lambda x: x[1])
    else:
        t0 = min(float(m['t_start']) for m in manifest)
        starts = sorted([(m['name'], float(m['t_start']) - t0,
                          float(m.get('sub_index', 0)))
                         for m in manifest], key=lambda x: x[1])

    intervals = []
    for i, (name, ts, sub) in enumerate(starts):
        t_end = starts[i + 1][1] if i + 1 < len(starts) else float('inf')
        intervals.append((f'{name}#{int(sub)}', ts, t_end))

    # For each signal, compute per-regime stats
    target_signals = [
        ('theta',         'C0', axes[0]),
        ('r_force_filt',  'C1', axes[1]),
        ('r_encoder_filt', 'C2', axes[2]),
    ]

    for sig_name, color, ax in target_signals:
        if sig_name not in data:
            ax.set_visible(False)
            continue
        t, v = data[sig_name]
        labels, means, stds = [], [], []
        for label, t0_i, t1_i in intervals:
            mask = (t >= t0_i) & (t < t1_i)
            if mask.sum() < 50:
                continue
            labels.append(label)
            means.append(v[mask].mean())
            stds.append(v[mask].std())
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=color, alpha=0.7,
               error_kw={'linewidth': 0.5, 'capsize': 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=4)
        ax.set_ylabel(sig_name)
        ax.set_title(f'{sig_name}: per-regime mean ± std '
                     f'(N={len(labels)} regimes)')
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5)


# ════════════════════════════════════════════════════════════════════
# Page assembly
# ════════════════════════════════════════════════════════════════════

def build_pdf(data, manifest, output_path):
    with PdfPages(output_path) as pdf:

        # Page1 — joint state (theta, theta_dot)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        plot_state(data, manifest, axes)
        fig.suptitle('Page 1 — Joint state', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig); plt.close(fig)

        # Page2 — tracking
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        plot_tracking(data, manifest, ax)
        fig.suptitle('Page 2 — Admittance tracking', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); plt.close(fig)

        # Page3 — wrench
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        plot_wrench(data, manifest, ax)
        fig.suptitle('Page 3 — External wrench input', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); plt.close(fig)

        # Page4 — torque
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        plot_torque(data, manifest, ax)
        fig.suptitle('Page 4 — Commanded torque', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); plt.close(fig)

        # Page5 — FDI residuals filtered
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        plot_residuals_time(data, manifest, axes)
        fig.suptitle('Page 5 — FDI residuals (time series)', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig); plt.close(fig)

        # Page6 — distributions
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_residual_distributions(data, axes)
        fig.suptitle('Page 6 — FDI residual distributions', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig); plt.close(fig)

        # Page7 — per-regime
        fig, axes = plt.subplots(3, 1, figsize=(13, 11))
        plot_per_regime(data, manifest, axes)
        fig.suptitle('Page 7 — Per-regime statistics', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig); plt.close(fig)

        # Page 2 — tracking
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        plot_tracking(data, manifest, ax)
        fig.suptitle('Page 2 — Admittance tracking', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); plt.close(fig)

        # Page 3 — wrench
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        plot_wrench(data, manifest, ax)
        fig.suptitle('Page 3 — External wrench input', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); plt.close(fig)

        # Page 4 — torque
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        plot_torque(data, manifest, ax)
        fig.suptitle('Page 4 — Commanded torque', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); plt.close(fig)

        # Page 5 — residuals raw
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        plot_residuals_time(data, manifest, axes)
        fig.suptitle('Page 5 — Observer residuals (time series)', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig); plt.close(fig)

        # Page 6 — distributions
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        plot_residual_distributions(data, axes)
        fig.suptitle('Page 6 — Residual distributions', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig); plt.close(fig)

        # Page 7 — per-regime
        fig, axes = plt.subplots(3, 1, figsize=(13, 11))
        plot_per_regime(data, manifest, axes)
        fig.suptitle('Page 7 — Per-regime statistics', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig); plt.close(fig)


def build_pngs(data, manifest, out_dir: Path):
    """Same content as build_pdf but split into individual PNG files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = [
        ('01_joint_state',         (2, 1), (11, 7),
         lambda axes: plot_state(data, manifest, axes)),
        ('02_tracking',            (1, 1), (11, 5),
         lambda axes: plot_tracking(data, manifest, axes)),
        ('03_wrench',              (1, 1), (11, 5),
         lambda axes: plot_wrench(data, manifest, axes)),
        ('04_torque',              (1, 1), (11, 5),
         lambda axes: plot_torque(data, manifest, axes)),
        ('05_residuals_time',  (3, 1), (11, 9),
         lambda axes: plot_residuals_time(data, manifest, axes)),
        ('06_distributions',   (1, 3), (13, 5),
         lambda axes: plot_residual_distributions(data, axes)),
        ('07_per_regime',      (3, 1), (13, 11),
         lambda axes: plot_per_regime(data, manifest, axes)),
    ]
    for name, shape, size, plotter in pages:
        nr, nc = shape
        if nr == 1 and nc == 1:
            fig, ax = plt.subplots(1, 1, figsize=size)
            plotter(ax)
        elif nr == 1:
            fig, axes = plt.subplots(1, nc, figsize=size)
            plotter(axes)
        elif nc == 1:
            fig, axes = plt.subplots(nr, 1, figsize=size, sharex=True)
            plotter(axes)
        else:
            fig, axes = plt.subplots(nr, nc, figsize=size)
            plotter(axes)
        fig.tight_layout()
        out = out_dir / f'{name}.png'
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f'  wrote {out}')


# ════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', required=True, help='rosbag2 directory')
    ap.add_argument('--manifest', default=None,
                    help='baseline_manifest.json (optional, for regime markers)')
    ap.add_argument('--output', default=None,
                    help='output PDF path (default: <bag_dir>/baseline_plots.pdf)')
    ap.add_argument('--png', action='store_true',
                    help='produce individual PNG files instead of one PDF')
    args = ap.parse_args()

    bag = Path(args.bag).resolve()
    if not bag.exists():
        sys.exit(f'Bag not found: {bag}')

    manifest = []
    if args.manifest:
        mp = Path(args.manifest).resolve()
        if mp.exists():
            with open(mp) as f:
                manifest = json.load(f)
            print(f'Loaded manifest with {len(manifest)} regime entries')
        else:
            print(f'WARNING: manifest not found at {mp}; '
                  f'plots will be unsegmented')

    print(f'Reading bag: {bag}')
    print(f'(this may take a couple of minutes on a large bag)')
    data = read_bag(str(bag))
    print(f'  Extracted {len(data)} signals: {sorted(data.keys())}')

    if args.png:
        out_dir = Path(args.output) if args.output else bag / 'plots'
        print(f'Writing PNGs to: {out_dir}')
        build_pngs(data, manifest, out_dir)
    else:
        out_path = Path(args.output) if args.output else bag / 'baseline_plots.pdf'
        print(f'Writing PDF to: {out_path}')
        build_pdf(data, manifest, out_path)
        print(f'  done. {out_path.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()