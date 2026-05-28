#!/usr/bin/env python3
"""
plot_fault_sweep.py -- visualization of the fault sweep bag (FDI version)

Reads a fault sweep bag and produces a multi-page PDF with one page
per fault injection window. Each page shows:

  - The FDI residuals (r_force_filt, r_encoder_filt)
    during a time interval [t_arm - 2s, t_disarm + 2s]
  - Vertical lines marking t_arm and t_disarm
  - The injected fault signature (delta from /fault_injector/status)
  - Window metadata in the title (channel, type, magnitude, name)

A summary page at the start shows the overall sensitivity matrix as
a heatmap, and an overview page at the end shows the residuals over
the entire bag duration with all windows highlighted.

Usage
-----
    python3 plot_fault_sweep.py \\
        --bag      ./bags/fault_sweep_<TIMESTAMP> \\
        --manifest ./fault_sweep_manifest.json
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import rclpy.serialization as ser
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f'ERROR: ROS 2 Python bindings not available ({e})', file=sys.stderr)
    sys.exit(1)

JOINT_NAME = 'rev_crank'

# Pre/post padding around each window (seconds) for the per-window plots
PADDING_SEC = 2.0


# ----------------------------------------------------------------
# Bag reading
# ----------------------------------------------------------------
def _detect_storage_id(bag_path: str) -> str:
    p = Path(bag_path)
    if any(p.glob('*.mcap')):
        return 'mcap'
    if any(p.glob('*.db3')):
        return 'sqlite3'
    return 'mcap'


def read_bag(bag_path: str) -> tuple:
    storage = StorageOptions(uri=bag_path, storage_id=_detect_storage_id(bag_path))
    converter = ConverterOptions(input_serialization_format='cdr',
                               output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage, converter)

    msg_classes = {t.name: get_message(t.type)
                   for t in reader.get_all_topics_and_types()}

    series = defaultdict(lambda: ([], []))
    t_first = None

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        cls = msg_classes.get(topic)
        if cls is None:
            continue
        ts = float(t_ns) * 1e-9
        if t_first is None:
            t_first = ts

        if topic == '/joint_states':
            msg = ser.deserialize_message(raw, cls)
            try:
                idx = list(msg.name).index(JOINT_NAME)
                series['theta'][0].append(ts)
                series['theta'][1].append(float(msg.position[idx]))
            except ValueError:
                pass
        elif topic == '/fdi/debug':
            msg = ser.deserialize_message(raw, cls)
            d = list(msg.data)
            if len(d) > 8:
                series['r_force_filt'][0].append(ts)
                series['r_force_filt'][1].append(float(d[8]))
            if len(d) > 9:
                series['r_encoder_filt'][0].append(ts)
                series['r_encoder_filt'][1].append(float(d[9]))
        elif topic == '/fault_injector/status':
            msg = ser.deserialize_message(raw, cls)
            d = list(msg.data)
            if len(d) >= 8:
                series['fi_fault_active'][0].append(ts)
                series['fi_fault_active'][1].append(float(d[2]))
                series['fi_delta'][0].append(ts)
                series['fi_delta'][1].append(float(d[7]))

    return ({k: (np.array(ts), np.array(vs))
             for k, (ts, vs) in series.items()},
            t_first)


def slice_signal(series, t0, t1):
    if series is None:
        return np.array([]), np.array([])
    ts, vs = series
    if len(ts) == 0:
        return np.array([]), np.array([])
    mask = (ts >= t0) & (ts < t1)
    return ts[mask], vs[mask]


# ----------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------
def downsample(t, v, max_points=10000):
    if len(t) <= max_points:
        return t, v
    step = len(t) // max_points + 1
    return t[::step], v[::step]


def plot_window(window: dict, data: dict, fig, t_first: float):
    name = window.get('name', '?')
    channel = window.get('channel', '?')
    ftype = window.get('type', '?')
    magnitude = float(window.get('magnitude', 0.0))
    t_arm = float(window['t_arm'])
    t_disarm = float(window['t_disarm'])

    t_lo = t_arm - PADDING_SEC
    t_hi = t_disarm + PADDING_SEC

    axes = fig.subplots(3, 1, sharex=True)

    plot_specs = [
        ('r_force_filt',   'C0', 'r_force_filt [Nm]'),
        ('r_encoder_filt', 'C1', 'r_encoder_filt [rad]'),
    ]
    for ax, (sig_name, color, ylabel) in zip(axes[:2], plot_specs):
        ts, v = slice_signal(data.get(sig_name), t_lo, t_hi)
        if len(ts) > 0:
            t_rel = ts - t_arm
            ax.plot(t_rel, v, linewidth=0.7, color=color)
            pre_mask = ts < t_arm
            if pre_mask.sum() > 50:
                pre_mean = float(np.mean(v[pre_mask]))
                pre_std = float(np.std(v[pre_mask]))
                ax.axhline(pre_mean, color='gray', linestyle=':', linewidth=0.6, alpha=0.7)
                ax.axhline(pre_mean + 3 * pre_std, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
                ax.axhline(pre_mean - 3 * pre_std, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        else:
            ax.text(0.5, 0.5, '(no samples)', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
        ax.axvline(0, color='red', linewidth=0.7, alpha=0.7)
        ax.axvline(t_disarm - t_arm, color='red', linewidth=0.7, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    ax_d = axes[2]
    ts, v = slice_signal(data.get('fi_delta'), t_lo, t_hi)
    if len(ts) > 0:
        t_rel = ts - t_arm
        ax_d.plot(t_rel, v, linewidth=0.7, color='C3')
    else:
        ax_d.text(0.5, 0.5, '(no samples)', ha='center', va='center',
                  transform=ax_d.transAxes, color='gray')
    ax_d.axvline(0, color='red', linewidth=0.7, alpha=0.7)
    ax_d.axvline(t_disarm - t_arm, color='red', linewidth=0.7, alpha=0.7)
    ax_d.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax_d.set_ylabel('fi_delta\n[ground truth]', fontsize=8)
    ax_d.set_xlabel('time relative to t_arm [s]', fontsize=8)
    ax_d.grid(True, alpha=0.3)
    ax_d.tick_params(labelsize=7)

    fig.suptitle(f'Window: {name}   ch={channel}   type={ftype}   '
                 f'magnitude={magnitude:+.4g}   duration={t_disarm-t_arm:.1f}s',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def plot_overview(manifest: list, data: dict, fig):
    axes = fig.subplots(2, 1, sharex=True)

    plot_specs = [
        ('r_force_filt',   'C0', 'r_force_filt [Nm]'),
        ('r_encoder_filt', 'C1', 'r_encoder_filt [rad]'),
    ]
    for ax, (sig_name, color, ylabel) in zip(axes, plot_specs):
        if sig_name not in data:
            ax.set_visible(False)
            continue
        t, v = data[sig_name]
        td, vd = downsample(t, v, max_points=20000)
        t0 = t.min() if len(t) else 0
        ax.plot(td - t0, vd, linewidth=0.4, color=color)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        for w in manifest:
            t_arm = float(w['t_arm']) - t0
            t_disarm = float(w['t_disarm']) - t0
            ax.axvspan(t_arm, t_disarm, color='red', alpha=0.08)
    axes[-1].set_xlabel('time [s]', fontsize=8)
    fig.suptitle(f'Bag overview -- {len(manifest)} fault windows highlighted',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def plot_summary_heatmap(manifest: list, data: dict, fig):
    residuals = ['r_force_filt', 'r_encoder_filt']
    rows = []
    labels = []
    for w in manifest:
        t_arm = float(w['t_arm'])
        t_disarm = float(w['t_disarm'])
        row = []
        for r in residuals:
            ts, v = slice_signal(data.get(r), t_arm - 5.0, t_arm)
            if len(v) > 50:
                pre_mean = float(np.mean(v))
            else:
                pre_mean = 0.0
            ts_w, v_w = slice_signal(data.get(r), t_arm, t_disarm)
            if len(v_w) > 0:
                max_dev = float(np.max(np.abs(v_w - pre_mean)))
            else:
                max_dev = 0.0
            row.append(max_dev)
        rows.append(row)
        labels.append(f'#{len(rows)-1} {w.get("name","?")[:18]}')

    if not rows:
        return
    arr = np.array(rows)
    arr_norm = arr.copy()
    for j in range(arr.shape[1]):
        col = arr[:, j]
        norm = np.percentile(col, 95) if np.any(col > 0) else 1.0
        if norm > 0:
            arr_norm[:, j] = col / norm

    ax = fig.subplots(1, 1)
    im = ax.imshow(arr_norm, cmap='hot_r', aspect='auto', vmin=0, vmax=1.5)
    ax.set_xticks(range(len(residuals)))
    ax.set_xticklabels(residuals, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=4)
    fig.colorbar(im, ax=ax, label='max|r-pre_mean| / column p95')
    ax.set_title('Sensitivity overview -- color = max residual deviation '
                 '(normalized per column)', fontsize=10)
    fig.tight_layout()


# ----------------------------------------------------------------
# PDF assembly
# ----------------------------------------------------------------
def build_pdf(manifest: list, data: dict, t_first: float,
              out_path: Path, max_windows: int = None,
              window_only: int = None):
    with PdfPages(out_path) as pdf:
        # Page1 -- bag overview
        fig = plt.figure(figsize=(13, 9))
        plot_overview(manifest, data, fig)
        pdf.savefig(fig)
        plt.close(fig)

        # Page2 -- sensitivity heatmap
        fig = plt.figure(figsize=(8, max(6, 0.18 * len(manifest))))
        plot_summary_heatmap(manifest, data, fig)
        pdf.savefig(fig)
        plt.close(fig)

        windows_to_plot = manifest
        if window_only is not None:
            if 0 <= window_only < len(manifest):
                windows_to_plot = [manifest[window_only]]
            else:
                print(f'WARNING: --window-only {window_only} out of range; '
                      f'plotting all instead')
        elif max_windows is not None:
            windows_to_plot = manifest[:max_windows]

        for i, w in enumerate(windows_to_plot):
            fig = plt.figure(figsize=(11, 9))
            plot_window(w, data, fig, t_first)
            pdf.savefig(fig)
            plt.close(fig)
            if (i + 1) % 10 == 0:
                print(f'  rendered {i + 1}/{len(windows_to_plot)} pages')

    print(f'  done. {out_path.stat().st_size / 1024:.0f} KB')


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', default=None)
    ap.add_argument('--max-windows', type=int, default=None,
                    help='render at most N per-window pages')
    ap.add_argument('--window-only', type=int, default=None,
                    help='render only one specific window (0-based index)')
    args = ap.parse_args()

    bag = Path(args.bag).resolve()
    if not bag.exists():
        sys.exit(f'Bag not found: {bag}')

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        sys.exit(f'Manifest not found: {manifest_path}')

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f'Manifest has {len(manifest)} windows')

    print(f'Reading bag: {bag}')
    print(f'(this will take a couple of minutes on a large bag)')
    data, t_first = read_bag(str(bag))
    print(f'  extracted signals: {sorted(data.keys())}')

    out_path = Path(args.output) if args.output else bag / 'fault_sweep_plots.pdf'
    print(f'Writing PDF to: {out_path}')
    build_pdf(manifest, data, t_first, out_path,
              max_windows=args.max_windows,
              window_only=args.window_only)


if __name__ == '__main__':
    main()
