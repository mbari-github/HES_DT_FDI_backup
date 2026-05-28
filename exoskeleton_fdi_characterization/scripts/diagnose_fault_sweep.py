#!/usr/bin/env python3
"""
diagnose_fault_sweep.py -- sanity checks on the fault sweep bag (FDI version)

Verifies that the recorded fault sweep is suitable for downstream
analysis. Specifically checks:

  1. Every window in the manifest is covered by the bag (start and
     end timestamps fall within the bag's time range).

  2. The fault_injector actually armed and disarmed within each
     window (we expect fault_active to be True between t_arm and
     t_disarm, False outside).

  3. The "delta steady" -- i.e. the average size of the fault
     framework's correction during the window -- has the expected
     sign and order of magnitude relative to the requested
     magnitude.

  4. theta stays within reasonable bounds even under faults. If
     theta saturates at the plant's hard limits during a window,
     the residuals from that window may be contaminated by the
     saturation dynamics rather than the fault dynamics; we flag
     those windows.

  5. Per-window ratio of valid samples on each FDI residual.

Usage
-----
    python3 diagnose_fault_sweep.py \\
        --bag      ./bags/fault_sweep_<TIMESTAMP> \\
        --manifest ./fault_sweep_manifest.json

Output is plain text on stdout.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import rclpy.serialization as ser
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f'ERROR: ROS 2 Python bindings not available ({e})', file=sys.stderr)
    print('       Source the workspace before running this script.', file=sys.stderr)
    sys.exit(1)

# Configuration: physical limits of the plant.
THETA_MIN     = -0.75
THETA_MAX     =  0.09
WALL_BUFFER   =  0.05
JOINT_NAME    = 'rev_crank'

# Fault types codes (must match fault_framework.py)
FAULT_TYPE_NAMES = {
    0: 'NONE',
    1: 'offset',
    2: 'scale',
    3: 'noise',
    4: 'freeze',
    5: 'drift_linear',
}


def _detect_storage_id(bag_path: str) -> str:
    p = Path(bag_path)
    if any(p.glob('*.mcap')):
        return 'mcap'
    if any(p.glob('*.db3')):
        return 'sqlite3'
    return 'mcap'


def read_bag(bag_path: str) -> tuple:
    """
    Returns (data, t_first, t_last) where data is a dict
    signal_name -> (times: np.ndarray, values: np.ndarray) and
    t_first / t_last are the absolute bag timestamps in seconds.
    """
    storage = StorageOptions(uri=bag_path, storage_id=_detect_storage_id(bag_path))
    converter = ConverterOptions(input_serialization_format='cdr',
                               output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage, converter)

    msg_classes = {t.name: get_message(t.type)
                   for t in reader.get_all_topics_and_types()}

    series = defaultdict(lambda: ([], []))
    t_first = None
    t_last = None

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        cls = msg_classes.get(topic)
        if cls is None:
            continue
        ts = float(t_ns) * 1e-9
        if t_first is None:
            t_first = ts
        t_last = ts

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
                series['fi_active_channel'][0].append(ts)
                series['fi_active_channel'][1].append(float(d[0]))
                series['fi_type_id'][0].append(ts)
                series['fi_type_id'][1].append(float(d[1]))
                series['fi_fault_active'][0].append(ts)
                series['fi_fault_active'][1].append(float(d[2]))
                series['fi_magnitude'][0].append(ts)
                series['fi_magnitude'][1].append(float(d[3]))
                series['fi_delta'][0].append(ts)
                series['fi_delta'][1].append(float(d[7]))

    return ({k: (np.array(ts), np.array(vs))
             for k, (ts, vs) in series.items()},
            t_first, t_last)


def slice_signal(series, t0, t1):
    if series is None:
        return np.array([]), np.array([])
    ts, vs = series
    if len(ts) == 0:
        return np.array([]), np.array([])
    mask = (ts >= t0) & (ts < t1)
    return ts[mask], vs[mask]


def diagnose_window(window: dict, data: dict, idx: int, total: int) -> dict:
    """Run all the per-window checks. Returns a dict of findings."""
    name      = window.get('name', f'win_{idx}')
    channel   = window.get('channel', '?')
    ftype     = window.get('type', '?')
    magnitude = float(window.get('magnitude', 0.0))
    t_arm     = float(window['t_arm'])
    t_disarm  = float(window['t_disarm'])

    findings = {
        'idx': idx,
        'name': name,
        'channel': channel,
        'type': ftype,
        'magnitude': magnitude,
        't_arm': t_arm,
        't_disarm': t_disarm,
        'duration': t_disarm - t_arm,
        'issues': [],
    }

    # Check 1: window is covered by the bag (relies on global t_first/t_last)

    # Check 2: fault_injector actually armed and disarmed
    _, fa_in = slice_signal(data.get('fi_fault_active'), t_arm, t_disarm)
    if len(fa_in) == 0:
        findings['issues'].append('NO fi_fault_active samples in window')
    else:
        frac_active = float(np.mean(fa_in > 0.5))
        findings['frac_fault_active_in_window'] = frac_active
        if frac_active < 0.90:
            findings['issues'].append(
                f'fault_active was True only {100*frac_active:.0f}% '
                f'of the window (expected >=90%)')

    # Check 3: delta steady has expected order of magnitude
    _, delta_in = slice_signal(data.get('fi_delta'), t_arm + 0.5, t_disarm)
    if len(delta_in) > 50:
        delta_mean = float(np.mean(delta_in))
        delta_max  = float(np.max(np.abs(delta_in)))
        findings['delta_mean'] = delta_mean
        findings['delta_max']  = delta_max
        if ftype == 'offset' and abs(magnitude) > 1e-9:
            ratio = abs(delta_mean) / abs(magnitude)
            if ratio < 0.2 or ratio > 5.0:
                findings['issues'].append(
                    f'offset delta_mean={delta_mean:+.4g} '
                    f'far from magnitude={magnitude:+.4g} (ratio {ratio:.2f})')
        elif ftype == 'scale' and abs(magnitude - 1.0) > 1e-9:
            if abs(delta_mean) < 1e-6:
                findings['issues'].append(
                    f'scale fault: delta_mean is essentially zero '
                    f'(scale={magnitude}, signal might be near zero)')
    else:
        findings['issues'].append(
            f'NO fi_delta samples in window (only {len(delta_in)})')

    # Check 4: theta saturation
    _, th_in = slice_signal(data.get('theta'), t_arm, t_disarm)
    if len(th_in) > 0:
        th_min = float(np.min(th_in))
        th_max = float(np.max(np.abs(th_in)))
        findings['theta_range'] = (float(np.min(th_in)), float(np.max(th_in)))
        free_lo = THETA_MIN + WALL_BUFFER
        free_hi = THETA_MAX - WALL_BUFFER
        n_in_buffer = int(np.sum((th_in < free_lo) | (th_in > free_hi)))
        n_total = len(th_in)
        if n_in_buffer > 0:
            findings['frac_in_buffer'] = n_in_buffer / n_total
            if n_in_buffer / n_total > 0.05:
                findings['issues'].append(
                    f'theta in wall buffer for {100*n_in_buffer/n_total:.1f}% '
                    f'of window (range: [{th_min:.3f}, '
                    f'{np.max(th_in):.3f}])')
        if np.min(th_in) < THETA_MIN - 1e-3 or np.max(th_in) > THETA_MAX + 1e-3:
            findings['issues'].append(
                f'theta exceeded wall limits: range '
                f'[{np.min(th_in):.3f}, {np.max(th_in):.3f}]')
    else:
        findings['issues'].append('NO theta samples in window')

    # Check 5: FDI residual sample counts in window
    for resname in ('r_force_filt', 'r_encoder_filt'):
        _, vals = slice_signal(data.get(resname), t_arm, t_disarm)
        n = len(vals)
        findings[f'{resname}_n_samples'] = n
        expected = max(1, int(180 * (t_disarm - t_arm)))
        if n < expected:
            findings['issues'].append(
                f'{resname}: only {n} samples '
                f'(expected >= {expected} for {t_disarm-t_arm:.1f}s @ 200Hz)')

    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', required=True, help='rosbag2 directory')
    ap.add_argument('--manifest', required=True, help='fault_sweep_manifest.json')
    args = ap.parse_args()

    bag = Path(args.bag).resolve()
    if not bag.exists():
        sys.exit(f'Bag not found: {bag}')

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        sys.exit(f'Manifest not found: {manifest_path}')

    print(f'Reading bag: {bag}')
    print(f'(this will take a couple of minutes on a large bag -- be patient)')
    data, t_first, t_last = read_bag(str(bag))
    bag_duration = t_last - t_first
    print(f'  Bag time range: {bag_duration:.1f} s '
          f'({bag_duration/60:.1f} min)')
    print(f'  Extracted signals: {sorted(data.keys())}')

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f'  Manifest has {len(manifest)} fault windows')
    print()

    # -- Check 1: every window covered by the bag --
    print('=' * 76)
    print('CHECK 1 -- Manifest windows vs bag coverage')
    print('=' * 76)
    print(f'  bag t_first = {t_first:.3f}')
    print(f'  bag t_last  = {t_last:.3f}')
    out_of_bounds = []
    for i, w in enumerate(manifest):
        t_arm = float(w['t_arm'])
        t_disarm = float(w['t_disarm'])
        if t_arm < t_first or t_disarm > t_last:
            out_of_bounds.append((i, w))
    if out_of_bounds:
        print(f'  {len(out_of_bounds)} window(s) out of bag bounds:')
        for i, w in out_of_bounds:
            print(f'    [{i}] {w.get("name","?")}  '
                  f't_arm={float(w["t_arm"]):.3f}, '
                  f't_disarm={float(w["t_disarm"]):.3f}')
    else:
        print(f'  ✓ All {len(manifest)} windows are within bag time bounds.')
    print()

    # -- Per-window diagnostics --
    print('=' * 76)
    print(f'CHECK 2..5 -- Per-window diagnostics ({len(manifest)} windows)')
    print('=' * 76)
    print()
    print(f'{"#":>3}  {"name":<24} {"ch":>2} {"type":<14} {"mag":>10} '
          f'{"dur":>6} {"%active":>8}  notes')
    print('-' * 76)

    all_findings = []
    n_ok = 0
    n_with_issues = 0
    for i, w in enumerate(manifest):
        f = diagnose_window(w, data, i, len(manifest))
        all_findings.append(f)
        active = f.get('frac_fault_active_in_window', 0.0)
        flag = '✓' if not f['issues'] else '!'
        if f['issues']:
            n_with_issues += 1
        else:
            n_ok += 1
        notes = '; '.join(f['issues']) if f['issues'] else 'OK'
        if len(notes) > 60:
            notes = notes[:57] + '...'
        print(f'{flag} {i:>3}  {f["name"]:<24} {str(f["channel"]):>2} '
              f'{f["type"]:<14} {f["magnitude"]:>10.4g} {f["duration"]:>6.1f} '
              f'{100*active:>6.1f}%  {notes}')

    print()

    # -- Aggregate summary --
    print('=' * 76)
    print('SUMMARY')
    print('=' * 76)
    print(f'  Total windows         : {len(manifest)}')
    print(f'  Windows OK            : {n_ok}')
    print(f'  Windows with issues   : {n_with_issues}')
    print()

    issue_counts = defaultdict(int)
    for f in all_findings:
        for issue in f['issues']:
            key = issue.split(':')[0] if ':' in issue else issue.split('=')[0]
            issue_counts[key] += 1
    if issue_counts:
        print(f'  Most common issues:')
        for k, c in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f'    {c:>3}x {k}')
    print()

    print(f'  Coverage:')
    by_combo = defaultdict(list)
    for f in all_findings:
        by_combo[(f['channel'], f['type'])].append(f)
    for k in sorted(by_combo):
        n = len(by_combo[k])
        n_clean = sum(1 for f in by_combo[k] if not f['issues'])
        print(f'    Ch{k[0]}  {k[1]:<14} : {n_clean}/{n} clean')
    print()

    # Final verdict
    print('=' * 76)
    print('VERDICT')
    print('=' * 76)
    pct_ok = 100.0 * n_ok / len(manifest) if manifest else 0
    if pct_ok >= 95:
        print(f'  {pct_ok:.1f}% of windows are clean.')
        print(f'  -> Bag is fully USABLE for sensitivity matrix calibration.')
    elif pct_ok >= 80:
        print(f'  {pct_ok:.1f}% of windows are clean.')
        print(f'  -> Bag is USABLE with caveats. Some windows may produce')
        print(f'    unreliable sensitivity values (look at the per-channel')
        print(f'    coverage above to decide if any fault category is')
        print(f'    under-represented).')
    else:
        print(f'  Only {pct_ok:.1f}% of windows are clean.')
        print(f'  -> Bag has SIGNIFICANT issues. Review the failures above')
        print(f'    before running analyze_fault_sweep.py.')


if __name__ == '__main__':
    main()
