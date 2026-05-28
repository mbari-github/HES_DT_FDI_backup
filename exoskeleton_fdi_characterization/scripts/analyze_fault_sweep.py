#!/usr/bin/env python3
"""
analyze_fault_sweep.py — offline analysis of the fault-injection bag (fdi_node)

Inputs
------
    --bag             path to the rosbag2 directory (./bags/fault_sweep_*)
    --manifest        path to fault_sweep_manifest.json (window metadata)
    --baseline-stats  baseline_stats.json from analyze_baseline.py
    [--threshold-percentile  99.9 (default)]

Outputs (written to --out-dir, default = bag dir)
-------------------------------------------------
    fault_sweep_report.md     human-readable per-window summary + detection table
    fault_sweep_table.json    machine-readable per-window stats
                              (consumed by compute_thresholds.py)
    sensitivity_matrix.csv    compact (channel, type, magnitude) × residual table

For each fault window the script computes:
  - max excursion of each filtered residual above the healthy mean
  - whether that excursion exceeds the healthy p99.9 threshold (triggered)
  - detection latency: time from t_arm to first threshold crossing
  - mean fault_injector delta over the active window (ground truth Δ)

Additionally, for each (channel, type) pair the script fits a
detection-probability curve across magnitudes and reports:
  - P_det(magnitude): fraction of window the residual was above threshold
  - MDL (minimum detectable level) at 90% detection probability

Usage
-----
    python3 analyze_fault_sweep.py \\
        --bag            ./bags/fault_sweep_<TIMESTAMP> \\
        --manifest       ./fault_sweep_manifest.json \\
        --baseline-stats ./bags/healthy_baseline_<TIMESTAMP>/baseline_stats.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import rclpy.serialization as ser
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f'ERROR: ROS 2 Python bindings not available ({e}).', file=sys.stderr)
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════
# Topic extractors
# ════════════════════════════════════════════════════════════════════

def _extract_status(msg):
    """fault_injector/status layout (Float64ArrayStamped)."""
    d = list(msg.data)
    return [
        ('active_channel', float(d[0]) if len(d) > 0 else 0.0),
        ('fault_type_id',  float(d[1]) if len(d) > 1 else 0.0),
        ('fault_active',   float(d[2]) if len(d) > 2 else 0.0),
        ('magnitude',      float(d[3]) if len(d) > 3 else 0.0),
        ('n_injections',   float(d[4]) if len(d) > 4 else 0.0),
        ('last_raw',       float(d[5]) if len(d) > 5 else 0.0),
        ('last_faulted',   float(d[6]) if len(d) > 6 else 0.0),
        ('delta',          float(d[7]) if len(d) > 7 else 0.0),
    ]


TOPIC_EXTRACTORS = {
    '/fdi/debug': (
        'exoskeleton_safety_msgs/msg/Float64ArrayStamped',
        lambda m: [
            # Filtered residuals — compared against thresholds
            ('r_force_filt',   float(m.data[8])  if len(m.data) > 8  else 0.0),
            ('r_encoder_filt', float(m.data[9])  if len(m.data) > 9  else 0.0),
            # Raw residuals — for latency computation (unfiltered, faster)
            ('r_force',        float(m.data[6])  if len(m.data) > 6  else 0.0),
            ('r_encoder',      float(m.data[7])  if len(m.data) > 7  else 0.0),
            ('force_valid',    float(m.data[10]) if len(m.data) > 10 else 0.0),
        ],
    ),
    '/fault_injector/status': (
        'exoskeleton_safety_msgs/msg/Float64ArrayStamped',
        _extract_status,
    ),
}

# Residuals to analyse (filtered versions, which thresholds are applied to)
RESIDUALS = ['r_force_filt', 'r_encoder_filt']

# Short names for display
SHORT = {'r_force_filt': 'r_force', 'r_encoder_filt': 'r_enc'}


def _detect_storage_id(bag_path: str) -> str:
    p = Path(bag_path)
    if any(p.glob('*.mcap')):
        return 'mcap'
    if any(p.glob('*.db3')):
        return 'sqlite3'
    return 'mcap'


def read_bag(bag_path: str) -> dict:
    storage_id = _detect_storage_id(bag_path)
    storage = StorageOptions(uri=bag_path, storage_id=storage_id)
    converter = ConverterOptions(input_serialization_format='cdr',
                                 output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage, converter)

    msg_classes = {t.name: get_message(t.type)
                   for t in reader.get_all_topics_and_types()}

    series = {}
    while reader.has_next():
        topic, raw, t = reader.read_next()
        if topic not in TOPIC_EXTRACTORS:
            continue
        cls = msg_classes.get(topic)
        if cls is None:
            continue
        msg = ser.deserialize_message(raw, cls)
        ts = float(t) * 1e-9
        for name, value in TOPIC_EXTRACTORS[topic][1](msg):
            series.setdefault(name, ([], []))
            series[name][0].append(ts)
            series[name][1].append(value)

    return {n: (np.array(ts_list), np.array(vs))
            for n, (ts_list, vs) in series.items()}


# ════════════════════════════════════════════════════════════════════
# Per-window analysis
# ════════════════════════════════════════════════════════════════════

def slice_signal(ts, vs, t0, t1):
    mask = (ts >= t0) & (ts < t1)
    return ts[mask], vs[mask]


def detection_latency(ts, vs, t_arm, threshold, mean_healthy):
    """Time from t_arm to first sample where |v − mean_healthy| > threshold."""
    if len(ts) == 0:
        return None
    abs_dev = np.abs(vs - mean_healthy)
    after_arm = ts >= t_arm
    idx = np.argmax((abs_dev > threshold) & after_arm)
    if not ((abs_dev > threshold) & after_arm).any():
        return None
    return float(ts[idx] - t_arm)


def analyze_window(window: dict, data: dict, baseline: dict,
                   pct_key: str = 'p99_9') -> dict:
    t_arm  = window['t_arm']
    t_dis  = window['t_disarm']

    out = {
        'name':      window['name'],
        'channel':   window['channel'],
        'type':      window['type'],
        'magnitude': window['magnitude'],
        'fault_window_duration': t_dis - t_arm,
        'residuals': {},
        'fault_delta_steady': None,
    }

    for r in RESIDUALS:
        if r not in data:
            continue
        ts, vs = data[r]
        b = baseline.get(r, {})
        mean_h = float(b.get('mean', 0.0))
        # Threshold = mean + p99.9(|deviation|)  (same formula as thresholds_template)
        thr = mean_h + float(b.get(pct_key, 0.0))

        ts_w, vs_w = slice_signal(ts, vs, t_arm, t_dis)
        if len(vs_w) == 0:
            out['residuals'][r] = None
            continue

        max_val     = float(np.max(vs_w))
        max_dev     = max_val - mean_h
        p_detect    = float(np.mean(vs_w > thr))   # fraction of window above threshold
        latency     = detection_latency(ts, vs, t_arm, thr - mean_h, mean_h)

        out['residuals'][r] = {
            'max_val':     max_val,
            'max_dev':     max_dev,
            'threshold':   thr,
            'mean_h':      mean_h,
            'mean_w':      float(np.mean(vs_w)),
            'std_w':       float(np.std(vs_w)),
            'p_detect':    p_detect,
            'triggered':   bool(max_dev > (thr - mean_h)),
            'latency_s':   latency,
        }

    # Ground-truth injection delta (skip first 0.5 s for transient)
    if 'delta' in data:
        ts, vs = data['delta']
        _, vs_w = slice_signal(ts, vs, t_arm + 0.5, t_dis)
        if len(vs_w) > 0:
            out['fault_delta_steady'] = float(np.mean(vs_w))

    return out


# ════════════════════════════════════════════════════════════════════
# Detection-probability curve per (channel, type)
# ════════════════════════════════════════════════════════════════════

def detection_curves(per_window: list) -> dict:
    """Group windows by (channel, type); return sorted (magnitude, p_detect) lists."""
    curves = {}
    for w in per_window:
        key = (w['channel'], w['type'])
        curves.setdefault(key, [])
        for r, d in w['residuals'].items():
            if d is None:
                continue
            curves[key].append({
                'residual':  r,
                'magnitude': w['magnitude'],
                'p_detect':  d['p_detect'],
                'triggered': d['triggered'],
                'latency_s': d['latency_s'],
            })

    # Sort by magnitude
    for key in curves:
        curves[key].sort(key=lambda x: (x['residual'], x['magnitude']))

    return curves


def mdl_90(curve_points: list, residual: str):
    """Minimum detectable level: smallest magnitude with p_detect >= 0.9."""
    pts = [(p['magnitude'], p['p_detect'])
           for p in curve_points if p['residual'] == residual and p['p_detect'] >= 0.9]
    if not pts:
        return None
    return min(m for m, _ in pts)


# ════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════

def build_report(per_window: list, curves: dict) -> str:
    text = ['# Fault Sweep Report', '']
    text.append(f'Total windows: {len(per_window)}')
    text.append('')

    by_ch = {}
    for w in per_window:
        by_ch.setdefault(w['channel'], []).append(w)

    present = [r for r in RESIDUALS
               if any(r in w['residuals'] for w in per_window)]

    # Per-window table
    for ch in sorted(by_ch):
        text.append(f'## Channel {ch}')
        text.append('')
        header = ('| Window | Type | Mag | Δ_steady | '
                  + ' | '.join(SHORT.get(r, r) + ' p_det / latency' for r in present) + ' |')
        text.append(header)
        text.append('|' + '|'.join(['---'] * (4 + len(present))) + '|')
        for w in by_ch[ch]:
            row = [w['name'], w['type'], f'{w["magnitude"]:.4g}']
            ds = w['fault_delta_steady']
            row.append(f'{ds:.3g}' if ds is not None else '—')
            for r in present:
                d = w['residuals'].get(r)
                if d is None:
                    row.append('—')
                else:
                    flag = '✓' if d['triggered'] else '·'
                    lat = f'{d["latency_s"]*1000:.0f} ms' if d['latency_s'] is not None else 'no'
                    row.append(f'{flag} {d["p_detect"]*100:.0f}% / {lat}')
            text.append('| ' + ' | '.join(row) + ' |')
        text.append('')

    # Minimum detectable level table
    text.append('## Minimum detectable level (MDL at 90% detection probability)')
    text.append('')
    text.append('| Channel | Type | Residual | MDL-90 |')
    text.append('|---|---|---|---|')
    for (ch, t), pts in sorted(curves.items()):
        for r in present:
            mdl = mdl_90(pts, r)
            mdl_str = f'{mdl:.4g}' if mdl is not None else '> max tested'
            text.append(f'| {ch} | {t} | {SHORT.get(r, r)} | {mdl_str} |')
    text.append('')

    text.append('Legend: ✓ triggered (max_val > healthy_mean + p99.9), · not triggered. '
                'p_det = fraction of fault window above threshold.')

    return '\n'.join(text) + '\n'


def write_sensitivity_csv(per_window: list, csv_path: Path):
    """One row per window, columns per residual."""
    fields = ['name', 'channel', 'type', 'magnitude', 'fault_delta_steady']
    for r in RESIDUALS:
        fields += [f'{r}_p_detect', f'{r}_triggered', f'{r}_latency_ms', f'{r}_max_dev']

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for win in per_window:
            row = {
                'name':               win['name'],
                'channel':            win['channel'],
                'type':               win['type'],
                'magnitude':          win['magnitude'],
                'fault_delta_steady': win['fault_delta_steady'],
            }
            for r in RESIDUALS:
                d = win['residuals'].get(r)
                if d is None:
                    row[f'{r}_p_detect']   = ''
                    row[f'{r}_triggered']  = ''
                    row[f'{r}_latency_ms'] = ''
                    row[f'{r}_max_dev']    = ''
                else:
                    row[f'{r}_p_detect']   = f'{d["p_detect"]:.4f}'
                    row[f'{r}_triggered']  = int(d['triggered'])
                    row[f'{r}_latency_ms'] = (f'{d["latency_s"]*1000:.1f}'
                                              if d['latency_s'] is not None else '')
                    row[f'{r}_max_dev']    = f'{d["max_dev"]:.6g}'
            w.writerow(row)


# ════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag',      required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--baseline-stats', required=True,
                    help='baseline_stats.json from analyze_baseline.py')
    ap.add_argument('--threshold-percentile', type=float, default=99.9,
                    choices=[99.0, 99.9, 99.99])
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    pct_map = {99.0: 'p99', 99.9: 'p99_9', 99.99: 'p99_99'}
    pct_key = pct_map[args.threshold_percentile]

    bag_path      = Path(args.bag).resolve()
    manifest_path = Path(args.manifest).resolve()
    baseline_path = Path(args.baseline_stats).resolve()
    out_dir       = Path(args.out_dir).resolve() if args.out_dir else bag_path

    print(f'Reading bag: {bag_path}')
    data = read_bag(str(bag_path))
    print(f'  {len(data)} signals')

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f'Loaded manifest: {len(manifest)} windows')

    with open(baseline_path) as f:
        baseline = json.load(f).get('global', {})
    print(f'Loaded baseline stats: {len(baseline)} signals')

    print('Analyzing windows...')
    per_window = [analyze_window(w, data, baseline, pct_key) for w in manifest]

    curves = detection_curves(per_window)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'fault_sweep_report.md'
    json_path   = out_dir / 'fault_sweep_table.json'
    csv_path    = out_dir / 'sensitivity_matrix.csv'

    with open(report_path, 'w') as f:
        f.write(build_report(per_window, curves))
    with open(json_path, 'w') as f:
        json.dump(per_window, f, indent=2)
    write_sensitivity_csv(per_window, csv_path)

    print(f'Wrote {report_path}')
    print(f'Wrote {json_path}')
    print(f'Wrote {csv_path}')

    n_trig = sum(1 for w in per_window
                 for d in w['residuals'].values() if d and d['triggered'])
    n_total = sum(len(w['residuals']) for w in per_window)
    print(f'\nQuick summary: {n_trig}/{n_total} (residual, window) pairs triggered')
    print()
    print('Next step:')
    print(f'  python3 compute_thresholds.py \\')
    print(f'      --baseline-stats {baseline_path} \\')
    print(f'      --sweep-table    {json_path}')


if __name__ == '__main__':
    main()
