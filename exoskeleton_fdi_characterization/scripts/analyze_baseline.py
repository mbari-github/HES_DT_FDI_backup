#!/usr/bin/env python3
"""
analyze_baseline.py — offline analysis of the healthy baseline bag (fdi_node)

Inputs
------
    --bag       path to the rosbag2 directory (e.g. ./bags/healthy_baseline_*)
    --manifest  path to baseline_manifest.json (regime transitions)

Outputs (written to --out-dir, default = bag dir)
-------------------------------------------------
    baseline_report.md          human-readable summary per-regime and global
    baseline_stats.json         machine-readable stats (consumed by
                                analyze_fault_sweep.py and compute_thresholds.py)
    thresholds_template.yaml    first-pass thresholds; refine with
                                compute_thresholds.py --baseline-stats

Signals extracted
-----------------
From /fdi/debug (12-field Float64ArrayStamped):
    r_force       [6]  — raw admittance-inversion residual  [Nm]
    r_encoder     [7]  — raw Luenberger residual            [rad]
    r_force_filt  [8]  — EMA(|r_force|)                    [Nm]   ← threshold input
    r_encoder_filt[9]  — EMA(|r_encoder|)                  [rad]  ← threshold input
    force_valid   [10] — 1 when R_force channel is valid

From /joint_states:
    theta, theta_dot   (rev_crank joint)

From /exo_dynamics/tau_ext_theta:
    tau_ext

From /exo_dynamics/ff_terms:
    M_eff, proj, g_proj, tau_pass_theta

Statistics computed
-------------------
For each signal: n_samples, mean, std, min, max,
                 p99 / p99.9 / p99.99 of |signal − mean|, max_abs.

Usage
-----
    python3 analyze_baseline.py \\
        --bag      ./bags/healthy_baseline_<TIMESTAMP> \\
        --manifest ./baseline_manifest.json
"""
import argparse
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
    print('       Source the workspace before running this script.', file=sys.stderr)
    sys.exit(1)


JOINT_NAME = 'rev_crank'


def _extract_joint_state(msg, joint_name=JOINT_NAME):
    try:
        idx = list(msg.name).index(joint_name)
    except ValueError:
        return []
    out = []
    if idx < len(msg.position):
        out.append(('theta',     float(msg.position[idx])))
    if idx < len(msg.velocity):
        out.append(('theta_dot', float(msg.velocity[idx])))
    return out


def _extract_float_stamped(msg, name):
    return [(name, float(msg.data))]


TOPIC_EXTRACTORS = {
    '/joint_states': (
        'sensor_msgs/msg/JointState',
        lambda m: _extract_joint_state(m),
    ),
    '/exo_dynamics/tau_ext_theta': (
        'exoskeleton_safety_msgs/msg/Float64Stamped',
        lambda m: _extract_float_stamped(m, 'tau_ext'),
    ),
    '/torque_post_inject': (
        'exoskeleton_safety_msgs/msg/Float64Stamped',
        lambda m: _extract_float_stamped(m, 'tau_m'),
    ),
    '/exo_dynamics/ff_terms': (
        'exoskeleton_safety_msgs/msg/Float64ArrayStamped',
        lambda m: [
            ('M_eff',          float(m.data[0]) if len(m.data) > 0 else 0.0),
            ('proj',           float(m.data[1]) if len(m.data) > 1 else 0.0),
            ('g_proj',         float(m.data[2]) if len(m.data) > 2 else 0.0),
            ('tau_pass_theta', float(m.data[3]) if len(m.data) > 3 else 0.0),
        ],
    ),
    '/fdi/debug': (
        'exoskeleton_safety_msgs/msg/Float64ArrayStamped',
        lambda m: [
            # Raw residuals (pre-EMA) — useful for alpha calibration
            ('r_force',        float(m.data[6])  if len(m.data) > 6  else 0.0),
            ('r_encoder',      float(m.data[7])  if len(m.data) > 7  else 0.0),
            # Filtered residuals — these are what thresholds are compared against
            ('r_force_filt',   float(m.data[8])  if len(m.data) > 8  else 0.0),
            ('r_encoder_filt', float(m.data[9])  if len(m.data) > 9  else 0.0),
            # Validity flag (1 = R_force channel active, 0 = saturated / deadband)
            ('force_valid',    float(m.data[10]) if len(m.data) > 10 else 0.0),
        ],
    ),
}


def _detect_storage_id(bag_path: str) -> str:
    p = Path(bag_path)
    if any(p.glob('*.mcap')):
        return 'mcap'
    if any(p.glob('*.db3')):
        return 'sqlite3'
    return 'mcap'


def read_bag(bag_path: str) -> dict:
    storage_id = _detect_storage_id(bag_path)
    storage_options = StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    msg_classes = {}
    for topic_meta in reader.get_all_topics_and_types():
        msg_classes[topic_meta.name] = get_message(topic_meta.type)

    series = {}
    while reader.has_next():
        topic, raw, t = reader.read_next()
        if topic not in TOPIC_EXTRACTORS:
            continue
        msg_class = msg_classes.get(topic)
        if msg_class is None:
            continue
        _, extractor = TOPIC_EXTRACTORS[topic]
        msg = ser.deserialize_message(raw, msg_class)
        ts = float(t) * 1e-9
        for name, value in extractor(msg):
            if name not in series:
                series[name] = ([], [])
            series[name][0].append(ts)
            series[name][1].append(value)

    return {
        name: (np.array(ts), np.array(vs))
        for name, (ts, vs) in series.items()
    }


def compute_global_stats(data: dict) -> dict:
    out = {}
    for name, (_, vs) in data.items():
        if len(vs) == 0:
            continue
        abs_dev = np.abs(vs - np.mean(vs))
        out[name] = {
            'n_samples': int(len(vs)),
            'mean':      float(np.mean(vs)),
            'std':       float(np.std(vs)),
            'min':       float(np.min(vs)),
            'max':       float(np.max(vs)),
            'p99':       float(np.percentile(abs_dev, 99.0)),
            'p99_9':     float(np.percentile(abs_dev, 99.9)),
            'p99_99':    float(np.percentile(abs_dev, 99.99)),
            'max_abs':   float(np.max(np.abs(vs))),
        }
    return out


def compute_per_regime_stats(data: dict, manifest: list) -> dict:
    if not manifest:
        return {}
    starts = sorted(
        [(m['name'], float(m['t_start']), float(m.get('sub_index', 0)))
         for m in manifest],
        key=lambda x: x[1],
    )
    intervals = []
    for i, (name, ts, sub) in enumerate(starts):
        t_end = starts[i + 1][1] if i + 1 < len(starts) else float('inf')
        intervals.append((f'{name}#{int(sub)}', ts, t_end))

    out = {}
    for label, t0, t1 in intervals:
        out[label] = {}
        for name, (ts, vs) in data.items():
            mask = (ts >= t0) & (ts < t1)
            if not mask.any():
                continue
            sub_vs = vs[mask]
            out[label][name] = {
                'mean':    float(np.mean(sub_vs)),
                'std':     float(np.std(sub_vs)),
                'max_abs': float(np.max(np.abs(sub_vs))),
                'n':       int(len(sub_vs)),
            }
    return out


def _recommend_alpha(std_raw: float, thresh: float, mean_h: float = 0.0) -> float:
    """EMA alpha so that std(filtered) < (thresh - mean_h) / 3 (SNR >= 3).

    The critical quantity is the *excursion*: how far the threshold sits above
    the healthy mean.  Using thresh/3 instead of (thresh-mean_h)/3 would give
    a wrong (too high) alpha when mean_h is a significant fraction of thresh.

    EMA variance: sigma2_out = sigma2_in * alpha / (2 - alpha)
    Solving for alpha given sigma2_out = ((thresh-mean_h)/3)^2:
        excursion = thresh - mean_h
        x = (excursion/3)^2 / sigma2_in
        alpha = 2x / (1 + x)
    """
    excursion = thresh - mean_h
    if std_raw <= 0.0 or excursion <= 0.0:
        return 0.1
    x = (excursion / 3.0) ** 2 / (std_raw ** 2)
    alpha = 2.0 * x / (1.0 + x)
    return float(min(max(alpha, 0.01), 0.5))


def _recommend_debounce(far_per_sample: float, p_target: float = 1e-9) -> int:
    """Minimum debounce_n so P(n consecutive false alarms) < p_target.

    The leaky-integrator debounce in fdi_node requires debounce_n
    consecutive alerts to trigger from a zero state.  Each sample is a
    false alert with probability far_per_sample (independent).

    n = ceil(log(p_target) / log(far_per_sample))
    """
    if far_per_sample <= 0.0 or far_per_sample >= 1.0:
        return 10
    import math
    n = math.ceil(math.log(p_target) / math.log(far_per_sample))
    return max(3, min(n, 50))


def make_thresholds_template(stats: dict, publish_rate: float = 200.0) -> str:
    """Generate a fdi_node YAML snippet from baseline statistics.

    Threshold = mean(healthy) + p99.9(|r - mean|)
    This gives a false alarm rate ≈ 0.1% per sample (1 / 1000 samples
    = 1 every 5 s at 200 Hz) BEFORE debounce.
    """
    def g(name, key, default=0.0):
        return stats.get(name, {}).get(key, default)

    # Thresholds — 99.9-th percentile of the actual signal value
    thresh_force   = g('r_force_filt', 'mean', 0.0) + g('r_force_filt', 'p99_9', 0.5)
    thresh_encoder = g('r_encoder_filt', 'mean', 0.0) + g('r_encoder_filt', 'p99_9', 0.02)

    # FAR per sample (probability each sample exceeds threshold in healthy)
    far_force   = 1.0 - 0.999   # p99.9  → 0.001
    far_encoder = 1.0 - 0.999

    # Recommended EMA alpha from raw residual std and computed threshold
    std_r_force   = g('r_force',   'std', 0.1)
    std_r_encoder = g('r_encoder', 'std', 0.005)
    mean_rf = g('r_force_filt',   'mean', 0.0)
    mean_re = g('r_encoder_filt', 'mean', 0.0)
    alpha_force   = _recommend_alpha(std_r_force,   thresh_force,   mean_rf)
    alpha_encoder = _recommend_alpha(std_r_encoder, thresh_encoder, mean_re)
    # Use the more conservative (smaller) alpha so both residuals are adequately filtered
    alpha = round(min(alpha_force, alpha_encoder), 3)
    alpha = max(alpha, 0.05)  # floor: avoid extreme lag

    # Debounce: P(false alarm for debounce_n consecutive samples) < 1e-9
    debounce_n = _recommend_debounce(max(far_force, far_encoder), p_target=1e-9)

    lines = [
        '# thresholds_template.yaml',
        '# First-pass thresholds from healthy baseline.',
        '# Refine with: python3 compute_thresholds.py --baseline-stats baseline_stats.json',
        '#              [--sweep-table fault_sweep_table.json]',
        '#',
        f'# Healthy residual stats (global):',
        f'#   r_force_filt : mean={g("r_force_filt","mean",0):.4f}'
        f'  std={g("r_force_filt","std",0):.4f}'
        f'  max={g("r_force_filt","max",0):.4f}  p99.9={g("r_force_filt","p99_9",0):.4f} Nm',
        f'#   r_encoder_filt: mean={g("r_encoder_filt","mean",0):.5f}'
        f'  std={g("r_encoder_filt","std",0):.5f}'
        f'  max={g("r_encoder_filt","max",0):.5f}  p99.9={g("r_encoder_filt","p99_9",0):.5f} rad',
        '#',
        'fdi_node:',
        '  ros__parameters:',
        '',
        f'    thresh_force:   {thresh_force:.4f}  '
        f'# mean + p99.9 of r_force_filt  (FAR ~ 0.1% / sample)',
        f'    thresh_encoder: {thresh_encoder:.5f}  '
        f'# mean + p99.9 of r_encoder_filt (FAR ~ 0.1% / sample)',
        '',
        f'    residual_alpha: {alpha:.3f}  '
        f'# EMA alpha: std_out < thresh/3 for both residuals (SNR >= 3)',
        '',
        f'    debounce_count: {debounce_n}  '
        f'# P(n consecutive false alarms) < 1e-9 at FAR=0.1%'
        f'  ({debounce_n * 1000.0 / publish_rate:.0f} ms at {publish_rate:.0f} Hz)',
    ]
    return '\n'.join(lines) + '\n'


def make_report(global_stats: dict, regime_stats: dict, manifest: list) -> str:
    text = ['# Healthy Baseline Report', '']
    text.append(f'Total regime entries: {len(manifest)}')
    text.append('')

    text.append('## Global statistics')
    text.append('')
    text.append('| Signal | Samples | Mean | Std | Min | Max | p99.9(|dev|) | MaxAbs |')
    text.append('|---|---|---|---|---|---|---|---|')
    for name, s in sorted(global_stats.items()):
        text.append(
            f'| {name} | {s["n_samples"]} | {s["mean"]:.4g} | {s["std"]:.4g} | '
            f'{s["min"]:.4g} | {s["max"]:.4g} | {s["p99_9"]:.4g} | {s["max_abs"]:.4g} |'
        )
    text.append('')

    if regime_stats:
        text.append('## Per-regime statistics (key signals)')
        text.append('')
        signals_of_interest = [
            'theta', 'theta_dot', 'tau_ext', 'tau_m',
            'r_force_filt', 'r_encoder_filt',
        ]
        present = [s for s in signals_of_interest
                   if any(s in sub for sub in regime_stats.values())]
        if present:
            header = '| Regime | ' + ' | '.join(present) + ' |'
            text.append(header)
            text.append('|' + '|'.join(['---'] * (len(present) + 1)) + '|')
            for label, sub in regime_stats.items():
                row = [label]
                for s in present:
                    if s in sub:
                        row.append(f'{sub[s]["mean"]:.3g} ± {sub[s]["std"]:.3g}')
                    else:
                        row.append('—')
                text.append('| ' + ' | '.join(row) + ' |')
            text.append('')

    return '\n'.join(text) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag',      required=True, help='rosbag2 directory')
    ap.add_argument('--manifest', required=True, help='baseline_manifest.json path')
    ap.add_argument('--out-dir',  default=None,
                    help='output directory (default: same as --bag)')
    args = ap.parse_args()

    bag_path = Path(args.bag).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else bag_path

    if not bag_path.exists():
        sys.exit(f'Bag directory not found: {bag_path}')

    print(f'Reading bag: {bag_path}')
    data = read_bag(str(bag_path))
    print(f'  Extracted {len(data)} signals')

    manifest = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f'  Loaded manifest: {len(manifest)} regime entries')
    else:
        print(f'  WARNING: manifest not found; per-regime stats will be skipped')

    print('Computing global statistics...')
    global_stats = compute_global_stats(data)

    print('Computing per-regime statistics...')
    regime_stats = compute_per_regime_stats(data, manifest)

    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path    = out_dir / 'baseline_stats.json'
    report_path   = out_dir / 'baseline_report.md'
    template_path = out_dir / 'thresholds_template.yaml'

    with open(stats_path, 'w') as f:
        json.dump({'global': global_stats, 'per_regime': regime_stats}, f, indent=2)
    with open(report_path, 'w') as f:
        f.write(make_report(global_stats, regime_stats, manifest))
    with open(template_path, 'w') as f:
        f.write(make_thresholds_template(global_stats))

    print(f'Wrote {stats_path}')
    print(f'Wrote {report_path}')
    print(f'Wrote {template_path}')
    print()
    print('Next steps:')
    print('  1. Run the fault sweep: ros2 launch exoskeleton_fdi_characterization fault_sweep.launch.py')
    print('  2. Analyze sweep:       python3 analyze_fault_sweep.py --bag ./bags/fault_sweep_* ...')
    print('  3. Derive all params:   python3 compute_thresholds.py --baseline-stats baseline_stats.json \\')
    print('                              --sweep-table ./bags/fault_sweep_*/fault_sweep_table.json')


if __name__ == '__main__':
    main()
